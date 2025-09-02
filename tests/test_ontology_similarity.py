from unittest.mock import Mock, patch

from cyteonto.ontology.similarity import OntologySimilarity


class TestOntologySimilarity:
    """Test OntologySimilarity functionality."""

    def test_init_with_path(self, temp_dir):
        """Test initialization with OWL file path."""
        owl_path = temp_dir / "test.owl"
        owl_path.touch()  # Create empty file

        similarity = OntologySimilarity(owl_path)
        assert similarity.owl_file_path == owl_path
        assert similarity._ontology is None
        assert similarity._ontology_loaded is False

    def test_init_without_path(self):
        """Test initialization without OWL file path."""
        similarity = OntologySimilarity()
        assert similarity.owl_file_path is None
        assert similarity._ontology is None
        assert similarity._ontology_loaded is False

    def test_compute_simple_similarity_identical(self):
        """Test simple similarity computation for identical terms."""
        similarity = OntologySimilarity()
        result = similarity.compute_simple_similarity("T cell", "T cell")
        assert result == 1.0

    def test_compute_simple_similarity_different_case(self):
        """Test simple similarity with different case."""
        similarity = OntologySimilarity()
        result = similarity.compute_simple_similarity("T Cell", "t cell")
        assert result == 1.0  # Should be identical after normalization

    def test_compute_simple_similarity_with_underscores_and_hyphens(self):
        """Test simple similarity with underscores and hyphens."""
        similarity = OntologySimilarity()
        result = similarity.compute_simple_similarity("T_cell", "T-cell")
        assert result == 1.0  # Should be identical after normalization

    def test_compute_simple_similarity_partial_match(self):
        """Test simple similarity for partial matches."""
        similarity = OntologySimilarity()
        result = similarity.compute_simple_similarity("T helper cell", "T cell")
        assert 0 < result < 1  # Should be partially similar

    def test_compute_simple_similarity_no_match(self):
        """Test simple similarity for completely different terms."""
        similarity = OntologySimilarity()
        result = similarity.compute_simple_similarity("T cell", "neuron")
        assert 0 <= result < 1  # Should be low similarity

    @patch("cyteonto.ontology.similarity.get_ontology")
    def test_load_ontology_from_local_file(self, mock_get_ontology, temp_dir):
        """Test loading ontology from local OWL file."""
        owl_path = temp_dir / "test.owl"
        owl_path.touch()

        mock_ontology = Mock()
        mock_get_ontology.return_value = mock_ontology

        similarity = OntologySimilarity(owl_path)
        result = similarity._load_ontology()

        assert result is True
        assert similarity._ontology_loaded is True
        mock_get_ontology.assert_called_once_with(f"file://{owl_path.absolute()}")
        mock_ontology.load.assert_called_once()

    @patch("cyteonto.ontology.similarity.get_ontology")
    def test_load_ontology_from_url(self, mock_get_ontology):
        """Test loading ontology from URL when local file doesn't exist."""
        mock_ontology = Mock()
        mock_get_ontology.return_value = mock_ontology

        similarity = OntologySimilarity()
        result = similarity._load_ontology()

        assert result is True
        assert similarity._ontology_loaded is True
        mock_get_ontology.assert_called_once_with(
            "http://purl.obolibrary.org/obo/cl.owl"
        )
        mock_ontology.load.assert_called_once()

    @patch("cyteonto.ontology.similarity.get_ontology")
    def test_load_ontology_failure(self, mock_get_ontology):
        """Test ontology loading failure."""
        mock_get_ontology.side_effect = Exception("Failed to load ontology")

        similarity = OntologySimilarity()
        result = similarity._load_ontology()

        assert result is False
        assert similarity._ontology_loaded is True
        assert similarity._ontology is None

    def test_get_ancestors(self):
        """Test _get_ancestors method."""
        similarity = OntologySimilarity()

        # Test with None class
        ancestors = similarity._get_ancestors(None)
        assert ancestors == set()

        # Test with mock class
        mock_class = Mock()
        mock_ancestors = {Mock(), Mock(), Mock()}
        mock_class.ancestors.return_value = mock_ancestors

        ancestors = similarity._get_ancestors(mock_class)
        assert ancestors == mock_ancestors

    def test_compute_ontology_similarity_identical_terms(self):
        """Test ontology similarity for identical terms."""
        similarity = OntologySimilarity()
        result = similarity.compute_ontology_similarity("CL:0000001", "CL:0000001")
        assert result == 1.0

    def test_compute_ontology_similarity_invalid_types(self):
        """Test ontology similarity with invalid input types."""
        similarity = OntologySimilarity()
        result = similarity.compute_ontology_similarity(None, "CL:0000001")
        assert result == 0.0

        result = similarity.compute_ontology_similarity("CL:0000001", 123)
        assert result == 0.0

    def test_compute_ontology_similarity_non_cl_ids(self):
        """Test ontology similarity with non-CL format IDs."""
        similarity = OntologySimilarity()

        with patch.object(
            similarity, "compute_simple_similarity", return_value=0.5
        ) as mock_simple:
            result = similarity.compute_ontology_similarity("INVALID:001", "CL:0000001")
            assert result == 0.5
            mock_simple.assert_called_once_with("INVALID:001", "CL:0000001")

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_compute_ontology_similarity_no_ontology_loaded(self, mock_load):
        """Test ontology similarity when ontology fails to load."""
        mock_load.return_value = False

        similarity = OntologySimilarity()
        similarity._ontology = None

        with patch.object(
            similarity, "compute_simple_similarity", return_value=0.3
        ) as mock_simple:
            result = similarity.compute_ontology_similarity("CL:0000001", "CL:0000002")
            assert result == 0.3
            mock_simple.assert_called_once_with("CL:0000001", "CL:0000002")

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_compute_ontology_similarity_terms_not_found(self, mock_load):
        """Test ontology similarity when terms are not found in ontology."""
        mock_load.return_value = True

        similarity = OntologySimilarity()
        similarity._ontology = Mock()
        similarity._ontology.search_one.return_value = None  # Terms not found

        with patch.object(
            similarity, "compute_simple_similarity", return_value=0.2
        ) as mock_simple:
            result = similarity.compute_ontology_similarity("CL:0000001", "CL:0000002")
            assert result == 0.2
            mock_simple.assert_called_once_with("CL:0000001", "CL:0000002")

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_compute_ontology_similarity_with_weighted_ancestors(self, mock_load):
        """Test ontology similarity computation with weighted ancestors."""
        mock_load.return_value = True

        similarity = OntologySimilarity()
        similarity._ontology = Mock()

        # Mock classes found in ontology
        mock_class1 = Mock()
        mock_class2 = Mock()

        # Mock ancestors for weighted similarity calculation
        mock_ancestor_a = Mock()
        mock_ancestor_b = Mock()
        mock_ancestor_c = Mock()

        # Set up ancestor relationships
        mock_class1.ancestors.return_value = {mock_ancestor_a, mock_ancestor_b}
        mock_class2.ancestors.return_value = {mock_ancestor_b, mock_ancestor_c}

        # Set up nested ancestor relationships for weight calculation
        mock_ancestor_a.ancestors.return_value = {mock_ancestor_a}  # Self
        mock_ancestor_b.ancestors.return_value = {mock_ancestor_b}  # Self
        mock_ancestor_c.ancestors.return_value = {mock_ancestor_c}  # Self

        # Mock search_one to return our mock classes
        similarity._ontology.search_one.side_effect = [mock_class1, mock_class2]

        result = similarity.compute_ontology_similarity("CL:0000001", "CL:0000002")

        # Should return a similarity score based on weighted ancestor intersection
        assert 0 <= result <= 1
        assert similarity._ontology.search_one.call_count == 2

    def test_compute_batch_similarities(self):
        """Test batch similarity computation."""
        similarity = OntologySimilarity()

        pairs = [
            ("CL:0000001", "CL:0000001"),  # Identical
            ("CL:0000001", "CL:0000002"),  # Different
            ("invalid", "CL:0000003"),  # Invalid format
        ]

        with patch.object(similarity, "compute_ontology_similarity") as mock_compute:
            mock_compute.side_effect = [1.0, 0.5, 0.2]

            results = similarity.compute_batch_similarities(pairs)

            assert len(results) == 3
            assert results == [1.0, 0.5, 0.2]
            assert mock_compute.call_count == 3

    def test_compute_batch_similarities_empty(self):
        """Test batch similarity computation with empty input."""
        similarity = OntologySimilarity()
        results = similarity.compute_batch_similarities([])
        assert results == []

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_compute_ontology_similarity_exception_handling(self, mock_load):
        """Test ontology similarity computation handles exceptions gracefully."""
        mock_load.return_value = True

        similarity = OntologySimilarity()
        similarity._ontology = Mock()
        similarity._ontology.search_one.side_effect = Exception("Search failed")

        with patch.object(
            similarity, "compute_simple_similarity", return_value=0.1
        ) as mock_simple:
            result = similarity.compute_ontology_similarity("CL:0000001", "CL:0000002")
            assert result == 0.1
            mock_simple.assert_called_once_with("CL:0000001", "CL:0000002")

    def test_ontology_loading_caching(self):
        """Test that ontology loading is cached."""
        similarity = OntologySimilarity()

        with patch.object(similarity, "_load_ontology", return_value=True) as mock_load:
            # Ensure we start with unloaded state
            similarity._ontology_loaded = False

            # First call should load ontology
            _ = similarity.compute_ontology_similarity("CL:0000001", "CL:0000002")

            # Set loaded state manually since our mock returns True
            similarity._ontology_loaded = True

            # Second call should not load ontology again
            _ = similarity.compute_ontology_similarity("CL:0000003", "CL:0000004")

            # Should only be called once due to caching
            assert mock_load.call_count == 1
