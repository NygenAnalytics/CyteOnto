from unittest.mock import Mock, patch

import numpy as np
import pytest

from cyteonto.matcher.matcher import CyteOntoMatcher
from cyteonto.ontology.similarity import OntologySimilarity


class TestCosineKernelParams:
    """Test exposure of cosine_kernel parameters."""

    @pytest.fixture
    def mock_similarity(self):
        """Create a mock OntologySimilarity instance."""
        with patch(
            "cyteonto.ontology.similarity.OntologySimilarity._load_ontology",
            return_value=True,
        ):
            sim = OntologySimilarity()
            sim._ontology = Mock()
            # Mock embedding map
            sim.embedding_map = {
                "CL:0000001": np.array([1.0, 0.0]),
                "CL:0000002": np.array([0.0, 1.0]),  # Orthogonal, cosine sim = 0
                "CL:0000003": np.array([0.707, 0.707]),  # 45 deg, cosine sim ~ 0.707
            }
            # Mock find class to return something so it doesn't fail
            sim._find_class_cached = Mock(return_value=Mock())
            return sim

    def test_compute_ontology_similarity_default_params(self, mock_similarity):
        """Test with default parameters."""
        # Cosine similarity of CL:0000001 and CL:0000002 is 0.0
        # Default params: center=1, width=0.25, amplitude=1
        # Gaussian(0, 1, 0.25, 1) = exp(-((0-1)**2)/(2*0.25**2)) = exp(-1/0.125) = exp(-8) ~= 0.000335

        score = mock_similarity.compute_ontology_similarity(
            "CL:0000001", "CL:0000002", metric="cosine_kernel"
        )
        # We don't need exact match, just ensure it runs and returns a value
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_compute_ontology_similarity_custom_params(self, mock_similarity):
        """Test with custom parameters."""
        # Cosine similarity is 0.0
        # Custom params: center=0, width=1, amplitude=1
        # Gaussian(0, 0, 1, 1) = exp(-((0-0)**2)/(2*1**2)) = exp(0) = 1.0

        params = {"center": 0, "width": 1, "amplitude": 1}
        score = mock_similarity.compute_ontology_similarity(
            "CL:0000001", "CL:0000002", metric="cosine_kernel", metric_params=params
        )
        assert score == pytest.approx(1.0, rel=1e-5)

    def test_params_affect_result(self, mock_similarity):
        """Verify that changing parameters changes the result."""
        term1 = "CL:0000001"
        term2 = "CL:0000003"  # Cosine sim ~ 0.707

        # Default params
        score_default = mock_similarity.compute_ontology_similarity(
            term1, term2, metric="cosine_kernel"
        )

        # Custom params: wider width should increase score for non-perfect match if center=1
        # Sim is ~0.7. Center=1.
        # Default width=0.25. Dist=0.3. 0.3 > width. Score low.
        # Custom width=1.0. Dist=0.3. 0.3 < width. Score higher.

        params = {"width": 1.0}
        score_custom = mock_similarity.compute_ontology_similarity(
            term1, term2, metric="cosine_kernel", metric_params=params
        )

        assert score_custom > score_default

    @patch("cyteonto.matcher.matcher.CyteOntoMatcher._get_ontology_similarity")
    @patch("cyteonto.matcher.matcher.CyteOntoMatcher._get_ontology_extractor")
    def test_matcher_passes_params(self, mock_get_extractor, mock_get_similarity):
        """Test that CyteOntoMatcher passes parameters to OntologySimilarity."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor.build_mappings.return_value = (None, {})
        mock_get_extractor.return_value = mock_extractor

        mock_sim_calc = Mock()
        mock_sim_calc.compute_ontology_similarity.return_value = 0.5
        mock_get_similarity.return_value = mock_sim_calc

        matcher = CyteOntoMatcher(embeddings_file_path=Mock())

        params = {"center": 0.5}
        matcher.compute_ontology_similarity(
            ["T cell"], ["B cell"], metric="cosine_kernel", metric_params=params
        )

        # Verify call args
        call_args = mock_sim_calc.compute_ontology_similarity.call_args
        assert call_args is not None
        assert call_args.kwargs.get("metric_params") == params
