# tests/test_ontology_extractor.py

from unittest.mock import patch

import pandas as pd  # type: ignore

from cyteonto.ontology.extractor import OntologyExtractor


class TestOntologyExtractor:
    """Test OntologyExtractor functionality."""

    def test_init(self, sample_ontology_csv_file):
        """Test OntologyExtractor initialization."""
        extractor = OntologyExtractor(sample_ontology_csv_file)
        assert extractor.mapping_csv_path == sample_ontology_csv_file
        assert extractor._mapping_df is None
        assert extractor._ontology_to_labels is None
        assert extractor._label_to_ontology is None

    def test_load_mapping_success(self, sample_ontology_csv_file):
        """Test successful mapping loading."""
        extractor = OntologyExtractor(sample_ontology_csv_file)
        result = extractor.load_mapping()

        assert result is True
        assert extractor._mapping_df is not None
        assert len(extractor._mapping_df) == 3
        assert "ontology_id" in extractor._mapping_df.columns
        assert "label" in extractor._mapping_df.columns

    def test_load_mapping_file_not_found(self, temp_dir):
        """Test mapping loading with non-existent file."""
        non_existent_file = temp_dir / "non_existent.csv"
        extractor = OntologyExtractor(non_existent_file)

        result = extractor.load_mapping()
        assert result is False
        assert extractor._mapping_df is None

    @patch("pandas.read_csv")
    def test_load_mapping_invalid_csv(self, mock_read_csv, sample_ontology_csv_file):
        """Test mapping loading with invalid CSV data."""
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")
        extractor = OntologyExtractor(sample_ontology_csv_file)

        result = extractor.load_mapping()
        assert result is False
        assert extractor._mapping_df is None

    def test_get_all_ontology_terms(self, sample_ontology_csv_file):
        """Test getting all ontology terms."""
        extractor = OntologyExtractor(sample_ontology_csv_file)
        terms = extractor.get_all_ontology_terms()

        expected_terms = ["CL:0000001", "CL:0000002", "CL:0000003"]
        assert set(terms) == set(expected_terms)
        assert len(terms) == 3

    def test_get_all_ontology_terms_no_data(self, temp_dir):
        """Test getting ontology terms when no data is loaded."""
        non_existent_file = temp_dir / "non_existent.csv"
        extractor = OntologyExtractor(non_existent_file)

        terms = extractor.get_all_ontology_terms()
        assert terms == []

    def test_get_all_labels(self, sample_ontology_csv_file):
        """Test getting all labels."""
        extractor = OntologyExtractor(sample_ontology_csv_file)
        labels = extractor.get_all_labels()

        expected_labels = ["T cell", "B cell", "NK cell"]
        assert set(labels) == set(expected_labels)
        assert len(labels) == 3

    def test_build_mappings(self, temp_dir):
        """Test building bidirectional mappings."""
        # Create more complex test data
        data = {
            "ontology_id": ["CL:0000001", "CL:0000001", "CL:0000002", "CL:0000003"],
            "label": ["T cell", "T lymphocyte", "B cell", "NK cell"],
        }
        df = pd.DataFrame(data)
        csv_path = temp_dir / "test_mapping.csv"
        df.to_csv(csv_path, index=False)

        extractor = OntologyExtractor(csv_path)
        ontology_to_labels, label_to_ontology = extractor.build_mappings()

        # Test ontology_to_labels mapping (one-to-many)
        assert "CL:0000001" in ontology_to_labels
        assert set(ontology_to_labels["CL:0000001"]) == {"T cell", "T lymphocyte"}
        assert ontology_to_labels["CL:0000002"] == ["B cell"]
        assert ontology_to_labels["CL:0000003"] == ["NK cell"]

        # Test label_to_ontology mapping (many-to-one)
        assert label_to_ontology["T cell"] == "CL:0000001"
        assert label_to_ontology["T lymphocyte"] == "CL:0000001"
        assert label_to_ontology["B cell"] == "CL:0000002"
        assert label_to_ontology["NK cell"] == "CL:0000003"

    def test_build_mappings_no_data(self, temp_dir):
        """Test building mappings when no data is loaded."""
        non_existent_file = temp_dir / "non_existent.csv"
        extractor = OntologyExtractor(non_existent_file)

        ontology_to_labels, label_to_ontology = extractor.build_mappings()
        assert ontology_to_labels == {}
        assert label_to_ontology == {}

    def test_get_ontology_id_for_label(self, sample_ontology_csv_file):
        """Test getting ontology ID for a specific label."""
        extractor = OntologyExtractor(sample_ontology_csv_file)

        # This should trigger build_mappings internally
        ontology_id = extractor.get_ontology_id_for_label("T cell")
        assert ontology_id == "CL:0000001"

        ontology_id = extractor.get_ontology_id_for_label("Nonexistent cell")
        assert ontology_id is None

    def test_get_labels_for_ontology_id(self, sample_ontology_csv_file):
        """Test getting labels for a specific ontology ID."""
        extractor = OntologyExtractor(sample_ontology_csv_file)

        # This should trigger build_mappings internally
        labels = extractor.get_labels_for_ontology_id("CL:0000001")
        assert labels == ["T cell"]

        labels = extractor.get_labels_for_ontology_id("CL:9999999")
        assert labels == []

    def test_get_ontology_terms_for_description_generation(self, temp_dir):
        """Test getting ontology terms for description generation."""
        # Create test data with duplicate ontology IDs
        data = {
            "ontology_id": ["CL:0000001", "CL:0000001", "CL:0000002", "CL:0000002"],
            "label": ["T cell", "T lymphocyte", "B cell", "B lymphocyte"],
        }
        df = pd.DataFrame(data)
        csv_path = temp_dir / "test_mapping.csv"
        df.to_csv(csv_path, index=False)

        extractor = OntologyExtractor(csv_path)
        ontology_ids, ontology_terms = (
            extractor.get_ontology_terms_for_description_generation()
        )

        # Should return unique ontology IDs with concatenated labels
        assert len(ontology_ids) == 2
        assert "CL:0000001" in ontology_ids
        assert "CL:0000002" in ontology_ids

        # Check that labels are properly concatenated with semicolons
        idx_cl1 = ontology_ids.index("CL:0000001")
        idx_cl2 = ontology_ids.index("CL:0000002")

        assert "T cell;T lymphocyte" == ontology_terms[idx_cl1]
        assert "B cell;B lymphocyte" == ontology_terms[idx_cl2]

    def test_get_ontology_terms_for_description_generation_no_data(self, temp_dir):
        """Test getting ontology terms when no data is available."""
        non_existent_file = temp_dir / "non_existent.csv"
        extractor = OntologyExtractor(non_existent_file)

        ontology_ids, ontology_terms = (
            extractor.get_ontology_terms_for_description_generation()
        )
        assert ontology_ids == []
        assert ontology_terms == []

    def test_cached_mappings(self, sample_ontology_csv_file):
        """Test that mappings are cached after first build."""
        extractor = OntologyExtractor(sample_ontology_csv_file)

        # First call should build mappings
        ont_to_labels1, label_to_ont1 = extractor.build_mappings()

        # Second call should use cached mappings - verify internal cache is used
        assert extractor._ontology_to_labels is not None
        assert extractor._label_to_ontology is not None

        # Second call should return same content
        ont_to_labels2, label_to_ont2 = extractor.build_mappings()

        assert ont_to_labels1 == ont_to_labels2
        assert label_to_ont1 == label_to_ont2

        # Verify cached data is used in helper methods
        ontology_id1 = extractor.get_ontology_id_for_label("T cell")
        ontology_id2 = extractor.get_ontology_id_for_label("T cell")
        assert ontology_id1 == ontology_id2 == "CL:0000001"
