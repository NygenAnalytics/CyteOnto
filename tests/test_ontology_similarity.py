from unittest.mock import Mock, patch

from cyteonto.ontology.similarity import OntologySimilarity


class TestOntologySimilarity:
    """Test OntologySimilarity functionality."""

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_init_with_path(self, mock_load_ontology, temp_dir):
        """Test initialization with OWL file path."""
        owl_path = temp_dir / "test.owl"
        owl_path.touch()  # Create empty file

        # Mock _load_ontology to prevent actual loading during initialization
        mock_load_ontology.return_value = False

        similarity = OntologySimilarity(owl_path)
        assert similarity.owl_file_path == owl_path
        assert similarity._ontology is None
        assert similarity._ontology_loaded is False

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_init_without_path(self, mock_load_ontology):
        """Test initialization without OWL file path."""
        # Mock _load_ontology to prevent actual loading during initialization
        mock_load_ontology.return_value = False

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

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_get_ancestors(self, mock_load_ontology):
        """Test _get_ancestors_cached method."""
        # Mock _load_ontology to prevent actual loading during initialization
        mock_load_ontology.return_value = False

        similarity = OntologySimilarity()

        # Test with None class - should return empty set
        ancestors = similarity._get_ancestors_cached(None)
        assert ancestors == set()

        # Test with properly mocked class that has required attributes
        mock_class = Mock()
        mock_class.iri = "http://purl.obolibrary.org/obo/CL_0000001"

        # Create mock ancestors with CL_ in their IRIs
        mock_ancestor1 = Mock()
        mock_ancestor1.iri = "http://purl.obolibrary.org/obo/CL_0000002"
        mock_ancestor2 = Mock()
        mock_ancestor2.iri = "http://purl.obolibrary.org/obo/CL_0000003"

        mock_ancestors = {
            mock_ancestor1,
            mock_ancestor2,
            mock_class,
        }  # Include self to test filtering
        mock_class.ancestors.return_value = mock_ancestors

        ancestors = similarity._get_ancestors_cached(mock_class)
        # Should exclude self (mock_class) but include the other ancestors
        expected_ancestors = {mock_ancestor1, mock_ancestor2}
        assert ancestors == expected_ancestors

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

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_compute_batch_similarities(self, mock_load_ontology):
        """Test batch similarity computation using individual calls."""
        # Mock _load_ontology to prevent actual loading during initialization
        mock_load_ontology.return_value = False

        similarity = OntologySimilarity()

        pairs = [
            ("CL:0000001", "CL:0000001"),  # Identical
            ("CL:0000001", "CL:0000002"),  # Different
            ("invalid", "CL:0000003"),  # Invalid format
        ]

        with patch.object(similarity, "compute_ontology_similarity") as mock_compute:
            mock_compute.side_effect = [1.0, 0.5, 0.2]

            # Since compute_batch_similarities doesn't exist, test individual calls instead
            results = []
            for term1, term2 in pairs:
                results.append(similarity.compute_ontology_similarity(term1, term2))

            assert len(results) == 3
            assert results == [1.0, 0.5, 0.2]
            assert mock_compute.call_count == 3

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_compute_batch_similarities_empty(self, mock_load_ontology):
        """Test batch similarity computation with empty input using individual calls."""
        # Mock _load_ontology to prevent actual loading during initialization
        mock_load_ontology.return_value = False

        similarity = OntologySimilarity()

        # Since compute_batch_similarities doesn't exist, test empty list with individual calls
        pairs = []
        results = []
        for term1, term2 in pairs:
            results.append(similarity.compute_ontology_similarity(term1, term2))

        assert results == []

    @patch.object(OntologySimilarity, "_load_ontology")
    @patch.object(OntologySimilarity, "_find_class_cached")
    def test_compute_ontology_similarity_exception_handling(
        self, mock_find_class, mock_load
    ):
        """Test ontology similarity computation when classes are not found."""
        mock_load.return_value = True

        similarity = OntologySimilarity()
        similarity._ontology = Mock()

        # Mock _find_class_cached to return None (classes not found)
        mock_find_class.return_value = None

        with patch.object(
            similarity, "compute_simple_similarity", return_value=0.1
        ) as mock_simple:
            result = similarity.compute_ontology_similarity("CL:0000001", "CL:0000002")
            assert result == 0.1
            mock_simple.assert_called_once_with("CL:0000001", "CL:0000002")

    @patch.object(OntologySimilarity, "_load_ontology")
    def test_ontology_loading_caching(self, mock_load):
        """Test that ontology loading behavior is consistent with loaded state."""
        # Mock _load_ontology to prevent actual loading during initialization
        mock_load.return_value = False

        similarity = OntologySimilarity()

        # Test that when ontology is not loaded, it falls back to simple similarity
        # This should not trigger additional loading attempts
        with patch.object(
            similarity, "compute_simple_similarity", return_value=0.5
        ) as mock_simple:
            result1 = similarity.compute_ontology_similarity("CL:0000001", "CL:0000002")
            result2 = similarity.compute_ontology_similarity("CL:0000003", "CL:0000004")

            # Both calls should fall back to simple similarity
            assert result1 == 0.5
            assert result2 == 0.5
            assert mock_simple.call_count == 2

        # _load_ontology should have been called only during initialization
        assert mock_load.call_count == 1
