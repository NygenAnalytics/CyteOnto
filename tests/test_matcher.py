from unittest.mock import Mock, patch

import numpy as np

from cyteonto.matcher.matcher import CyteOntoMatcher
from cyteonto.ontology.extractor import OntologyExtractor
from cyteonto.ontology.similarity import OntologySimilarity
from cyteonto.storage.file_utils import FileManager
from cyteonto.storage.vector_store import VectorStore


class TestCyteOntoMatcher:
    """Test CyteOntoMatcher functionality."""

    @patch.object(VectorStore, "load_embeddings")
    def test_init_success(
        self, mock_load_embeddings, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test successful CyteOntoMatcher initialization."""
        embeddings_file = temp_dir / "test_embeddings.npz"
        base_data_path = str(temp_dir)

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            sample_embeddings,
            sample_ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, base_data_path)

        assert matcher.embeddings_file_path == embeddings_file
        assert isinstance(matcher.file_manager, FileManager)
        assert isinstance(matcher.vector_store, VectorStore)
        assert matcher.embeddings_ready is True
        assert matcher._ontology_embeddings is not None
        assert matcher._ontology_ids is not None

    @patch.object(VectorStore, "load_embeddings")
    def test_init_no_embeddings_file(self, mock_load_embeddings, temp_dir):
        """Test CyteOntoMatcher initialization without embeddings file."""
        base_data_path = str(temp_dir)

        matcher = CyteOntoMatcher(None, base_data_path)

        assert matcher.embeddings_file_path is None
        assert matcher.embeddings_ready is False
        assert matcher._ontology_embeddings is None
        assert matcher._ontology_ids is None

    @patch.object(VectorStore, "load_embeddings")
    def test_init_failed_loading(self, mock_load_embeddings, temp_dir):
        """Test CyteOntoMatcher initialization with failed embedding loading."""
        embeddings_file = temp_dir / "test_embeddings.npz"
        base_data_path = str(temp_dir)

        # Mock failed embedding loading
        mock_load_embeddings.return_value = None

        matcher = CyteOntoMatcher(embeddings_file, base_data_path)

        assert matcher.embeddings_ready is False
        assert matcher._ontology_embeddings is None
        assert matcher._ontology_ids is None

    @patch.object(VectorStore, "load_embeddings")
    def test_load_ontology_embeddings_cached(
        self, mock_load_embeddings, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test that ontology embeddings are cached after first load."""
        embeddings_file = temp_dir / "test_embeddings.npz"
        base_data_path = str(temp_dir)

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            sample_embeddings,
            sample_ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, base_data_path)

        # First call should load embeddings
        result1 = matcher._load_ontology_embeddings()

        # Second call should use cached embeddings
        result2 = matcher._load_ontology_embeddings()

        assert result1 is True
        assert result2 is True
        # Should only be called once during initialization
        mock_load_embeddings.assert_called_once()

    def test_get_ontology_extractor_cached(self, temp_dir, sample_ontology_csv_file):
        """Test that ontology extractor is cached."""
        matcher = CyteOntoMatcher(None, str(temp_dir))

        with patch.object(
            matcher.file_manager,
            "get_ontology_mapping_path",
            return_value=sample_ontology_csv_file,
        ):
            # First call should create extractor
            extractor1 = matcher._get_ontology_extractor()

            # Second call should return cached extractor
            extractor2 = matcher._get_ontology_extractor()

            assert extractor1 is extractor2
            assert isinstance(extractor1, OntologyExtractor)

    def test_get_ontology_extractor_load_failure(self, temp_dir):
        """Test ontology extractor when mapping load fails."""
        matcher = CyteOntoMatcher(None, str(temp_dir))

        non_existent_file = temp_dir / "non_existent.csv"
        with patch.object(
            matcher.file_manager,
            "get_ontology_mapping_path",
            return_value=non_existent_file,
        ):
            extractor = matcher._get_ontology_extractor()
            assert extractor is None

    def test_get_ontology_similarity_cached(self, temp_dir):
        """Test that ontology similarity calculator is cached."""
        matcher = CyteOntoMatcher(None, str(temp_dir))

        owl_file = temp_dir / "test.owl"
        with patch.object(
            matcher.file_manager, "get_ontology_owl_path", return_value=owl_file
        ):
            # First call should create similarity calculator
            similarity1 = matcher._get_ontology_similarity()

            # Second call should return cached calculator
            similarity2 = matcher._get_ontology_similarity()

            assert similarity1 is similarity2
            assert isinstance(similarity1, OntologySimilarity)

    @patch.object(VectorStore, "load_embeddings")
    def test_find_closest_ontology_terms_success(
        self, mock_load_embeddings, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test successful closest ontology terms finding."""
        embeddings_file = temp_dir / "test_embeddings.npz"

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            sample_embeddings,
            sample_ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

        # Create query embeddings
        query_embeddings = np.random.rand(2, 768).astype(np.float32)

        results = matcher.find_closest_ontology_terms(
            query_embeddings, top_k=3, min_similarity=0.0
        )

        assert len(results) == 2  # Two query embeddings
        assert len(results[0]) <= 3  # Top-k results
        assert len(results[1]) <= 3

        # Check result structure
        for query_results in results:
            for result in query_results:
                assert "label" in result
                assert "ontology_id" in result
                assert "similarity" in result
                assert isinstance(result["similarity"], float)

    @patch.object(VectorStore, "load_embeddings")
    def test_find_closest_ontology_terms_min_similarity(
        self, mock_load_embeddings, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test closest ontology terms finding with minimum similarity threshold."""
        embeddings_file = temp_dir / "test_embeddings.npz"

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            sample_embeddings,
            sample_ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

        # Create query embeddings
        query_embeddings = np.random.rand(1, 768).astype(np.float32)

        results = matcher.find_closest_ontology_terms(
            query_embeddings, top_k=5, min_similarity=0.9
        )

        assert len(results) == 1
        # With high similarity threshold, might get fewer results
        for result in results[0]:
            assert result["similarity"] >= 0.9

    def test_find_closest_ontology_terms_no_embeddings(self, temp_dir):
        """Test closest ontology terms finding when no embeddings loaded."""
        matcher = CyteOntoMatcher(None, str(temp_dir))

        query_embeddings = np.random.rand(1, 768).astype(np.float32)
        results = matcher.find_closest_ontology_terms(query_embeddings, top_k=3)

        assert results == []

    @patch.object(VectorStore, "load_embeddings")
    def test_find_closest_ontology_terms_1d_input(
        self, mock_load_embeddings, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test closest ontology terms finding with 1D query input."""
        embeddings_file = temp_dir / "test_embeddings.npz"

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            sample_embeddings,
            sample_ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

        # Create 1D query embedding
        query_embedding = np.random.rand(768).astype(np.float32)

        results = matcher.find_closest_ontology_terms(query_embedding, top_k=2)

        assert len(results) == 1  # Should be reshaped to 2D
        assert len(results[0]) <= 2

    def test_compute_ontology_similarity_success(
        self, temp_dir, sample_ontology_csv_file
    ):
        """Test successful ontology similarity computation."""
        matcher = CyteOntoMatcher(None, str(temp_dir))

        with patch.object(matcher, "_get_ontology_extractor") as mock_get_extractor:
            with patch.object(
                matcher, "_get_ontology_similarity"
            ) as mock_get_similarity:
                # Mock extractor
                mock_extractor = Mock()
                mock_extractor.build_mappings.return_value = (
                    {"CL:0000001": ["T cell"], "CL:0000002": ["B cell"]},
                    {"T cell": "CL:0000001", "B cell": "CL:0000002"},
                )
                mock_get_extractor.return_value = mock_extractor

                # Mock similarity calculator
                mock_similarity_calc = Mock()
                mock_similarity_calc.compute_ontology_similarity.return_value = 0.8
                mock_get_similarity.return_value = mock_similarity_calc

                author_terms = ["T cell", "B cell"]
                user_terms = ["T lymphocyte", "B lymphocyte"]

                results = matcher.compute_ontology_similarity(author_terms, user_terms)

                assert len(results) == 2
                assert all(isinstance(score, float) for score in results)
                assert mock_similarity_calc.compute_ontology_similarity.call_count == 2

    def test_compute_ontology_similarity_no_extractor(self, temp_dir):
        """Test ontology similarity computation when extractor fails."""
        matcher = CyteOntoMatcher(None, str(temp_dir))

        with patch.object(matcher, "_get_ontology_extractor", return_value=None):
            author_terms = ["T cell", "B cell"]
            user_terms = ["T lymphocyte", "B lymphocyte"]

            results = matcher.compute_ontology_similarity(author_terms, user_terms)

            assert results == [0.0, 0.0]

    @patch.object(VectorStore, "load_embeddings")
    def test_match_embeddings_to_ontology_success(
        self, mock_load_embeddings, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test successful embedding to ontology matching."""
        embeddings_file = temp_dir / "test_embeddings.npz"

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            sample_embeddings,
            sample_ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

        # Create query embeddings
        query_embeddings = np.random.rand(3, 768).astype(np.float32)

        results = matcher.match_embeddings_to_ontology(
            query_embeddings, top_k=1, min_similarity=0.1
        )

        assert len(results) == 3
        for result in results:
            if result is not None:
                assert "label" in result
                assert "ontology_id" in result
                assert "similarity" in result
                assert result["similarity"] >= 0.1

    @patch.object(VectorStore, "load_embeddings")
    def test_match_embeddings_to_ontology_no_matches(
        self, mock_load_embeddings, temp_dir, sample_embeddings, sample_ontology_ids
    ):
        """Test embedding to ontology matching with no matches above threshold."""
        embeddings_file = temp_dir / "test_embeddings.npz"

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            sample_embeddings,
            sample_ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

        # Create query embeddings
        query_embeddings = np.random.rand(2, 768).astype(np.float32)

        # Set very high similarity threshold
        results = matcher.match_embeddings_to_ontology(
            query_embeddings, top_k=1, min_similarity=0.99
        )

        assert len(results) == 2
        # With high threshold, might get None results
        for result in results:
            if result is not None:
                assert result["similarity"] >= 0.99
            # Some results might be None due to low similarity

    @patch.object(VectorStore, "load_embeddings")
    def test_match_embeddings_to_ontology_integration(
        self, mock_load_embeddings, temp_dir
    ):
        """Test integration of embedding matching with realistic data."""
        embeddings_file = temp_dir / "test_embeddings.npz"

        # Create realistic test data
        ontology_ids = ["CL:0000001", "CL:0000002", "CL:0000003"]
        ontology_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Simple test vectors
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).astype(np.float32)

        # Mock successful embedding loading
        mock_load_embeddings.return_value = (
            ontology_embeddings,
            ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

        # Create query similar to first ontology embedding
        query_embeddings = np.array([[0.9, 0.1, 0.1]]).astype(np.float32)

        results = matcher.match_embeddings_to_ontology(
            query_embeddings, top_k=1, min_similarity=0.1
        )

        assert len(results) == 1
        assert results[0] is not None
        assert results[0]["ontology_id"] == "CL:0000001"  # Should match first embedding
        assert results[0]["similarity"] > 0.8  # High similarity expected

    def test_matcher_initialization_error_handling(self, temp_dir):
        """Test that matcher handles initialization errors gracefully."""
        with patch.object(
            VectorStore,
            "load_embeddings",
            return_value=None,  # Return None instead of raising
        ):
            embeddings_file = temp_dir / "test_embeddings.npz"

            # Should not raise exception
            matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

            # Should be in failed state
            assert matcher.embeddings_ready is False
            assert matcher._ontology_embeddings is None

    @patch.object(VectorStore, "load_embeddings")
    def test_cosine_similarity_calculation(self, mock_load_embeddings, temp_dir):
        """Test that cosine similarity is calculated correctly."""
        embeddings_file = temp_dir / "test_embeddings.npz"

        # Create known embeddings for predictable similarity
        ontology_embeddings = np.array(
            [
                [1.0, 0.0],  # Will have similarity 1.0 with [1,0]
                [0.0, 1.0],  # Will have similarity 0.0 with [1,0]
            ]
        ).astype(np.float32)
        ontology_ids = ["CL:0000001", "CL:0000002"]

        mock_load_embeddings.return_value = (
            ontology_embeddings,
            ontology_ids,
            {"version": "1.0"},
        )

        matcher = CyteOntoMatcher(embeddings_file, str(temp_dir))

        # Query exactly matching first embedding
        query_embeddings = np.array([[1.0, 0.0]]).astype(np.float32)

        results = matcher.find_closest_ontology_terms(
            query_embeddings, top_k=2, min_similarity=0.0
        )

        assert len(results) == 1
        assert len(results[0]) == 2

        # First result should be perfect match
        assert abs(results[0][0]["similarity"] - 1.0) < 1e-6
        assert results[0][0]["ontology_id"] == "CL:0000001"

        # Second result should be orthogonal (similarity = 0)
        assert abs(results[0][1]["similarity"] - 0.0) < 1e-6
        assert results[0][1]["ontology_id"] == "CL:0000002"
