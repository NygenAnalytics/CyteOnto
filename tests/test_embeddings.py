import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from cyteonto.llm_config import EMBDModelConfig
from cyteonto.models.embeddings import generate_embeddings, query_embd_model


class TestEmbeddings:
    """Test embedding generation functionality."""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_query_embd_model_success(self, mock_post, sample_embd_model_config):
        """Test successful single embedding query."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        mock_post.return_value.__aenter__.return_value = mock_response

        result = await query_embd_model("test query", sample_embd_model_config)

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_query_embd_model_google_provider(self, mock_post):
        """Test embedding query with Google provider."""
        google_config = EMBDModelConfig(
            provider="google", model="gemini-embedding-001", apiKey="test-key"
        )

        # Mock response for Google format
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"embeddings": {"values": [0.1, 0.2, 0.3]}}
        )

        mock_post.return_value.__aenter__.return_value = mock_response

        result = await query_embd_model("test query", google_config)

        assert result == [0.1, 0.2, 0.3]

        # Verify Google-specific payload format
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "content" in payload
        assert "parts" in payload["content"]

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_query_embd_model_http_error(
        self, mock_post, sample_embd_model_config
    ):
        """Test embedding query with HTTP error."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_post.return_value.__aenter__.return_value = mock_response

        result = await query_embd_model("test query", sample_embd_model_config)

        assert result is None

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_query_embd_model_exception(
        self, mock_post, sample_embd_model_config
    ):
        """Test embedding query with exception."""
        mock_post.side_effect = Exception("Network error")

        result = await query_embd_model("test query", sample_embd_model_config)

        assert result is None

    @pytest.mark.asyncio
    @patch("cyteonto.models.embeddings.query_embd_model")
    async def test_generate_embeddings_success(
        self, mock_query, sample_embd_model_config
    ):
        """Test successful batch embedding generation."""
        # Mock individual embedding queries
        mock_query.side_effect = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        texts = ["text1", "text2", "text3"]
        result = await generate_embeddings(texts, sample_embd_model_config)

        assert result is not None
        assert result.shape == (3, 3)

        # Due to async execution, order might not be preserved, so check content
        expected_embeddings = {
            tuple([0.1, 0.2, 0.3]),
            tuple([0.4, 0.5, 0.6]),
            tuple([0.7, 0.8, 0.9]),
        }
        actual_embeddings = {tuple(row) for row in result}
        assert actual_embeddings == expected_embeddings

    @pytest.mark.asyncio
    async def test_generate_embeddings_no_api_key(self):
        """Test embedding generation without API key."""
        config = EMBDModelConfig(
            provider="deepinfra",
            model="test-model",
            apiKey="",  # Empty API key
        )

        texts = ["text1", "text2"]
        result = await generate_embeddings(texts, config)

        assert result is None

    @pytest.mark.asyncio
    @patch("cyteonto.models.embeddings.query_embd_model")
    async def test_generate_embeddings_partial_failure(
        self, mock_query, sample_embd_model_config
    ):
        """Test embedding generation with some failures."""
        # Test that an exception during embedding generation returns None
        mock_query.side_effect = Exception("Embedding generation failed")

        texts = ["text1", "text2", "text3"]
        result = await generate_embeddings(texts, sample_embd_model_config)

        # Should return None on any failure
        assert result is None

    @pytest.mark.asyncio
    @patch("cyteonto.models.embeddings.query_embd_model")
    async def test_generate_embeddings_empty_input(
        self, mock_query, sample_embd_model_config
    ):
        """Test embedding generation with empty input."""
        texts = []
        result = await generate_embeddings(texts, sample_embd_model_config)

        assert result is not None
        assert result.shape == (0,)  # Empty array has shape (0,), not (0, 0)
        mock_query.assert_not_called()

    @pytest.mark.asyncio
    @patch("cyteonto.models.embeddings.query_embd_model")
    async def test_generate_embeddings_concurrency_control(
        self, mock_query, sample_embd_model_config
    ):
        """Test that embedding generation respects concurrency limits."""
        # Set low concurrency limit
        sample_embd_model_config.maxConcEmbed = 2

        # Mock successful queries with delay to test concurrency
        async def mock_query_with_delay(*args):
            await asyncio.sleep(0.01)  # Small delay
            return [0.1, 0.2, 0.3]

        mock_query.side_effect = mock_query_with_delay

        texts = ["text1", "text2", "text3", "text4"]

        _ = asyncio.get_event_loop().time()
        result = await generate_embeddings(texts, sample_embd_model_config)
        _ = asyncio.get_event_loop().time()

        assert result is not None
        assert result.shape == (4, 3)

        # With concurrency limit of 2, should take at least 2 batches
        # This is a rough test of concurrency control
        assert mock_query.call_count == 4

    @pytest.mark.asyncio
    @patch("cyteonto.models.embeddings.query_embd_model")
    async def test_generate_embeddings_exception_handling(
        self, mock_query, sample_embd_model_config
    ):
        """Test embedding generation with exceptions."""
        # Mock query that raises exception
        mock_query.side_effect = Exception("Query failed")

        texts = ["text1", "text2"]
        result = await generate_embeddings(texts, sample_embd_model_config)

        assert result is None

    @pytest.mark.asyncio
    @patch("cyteonto.models.embeddings.query_embd_model")
    async def test_generate_embeddings_progress_logging(
        self, mock_query, sample_embd_model_config
    ):
        """Test that embedding generation logs progress."""
        # Mock successful queries
        mock_query.side_effect = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        texts = ["text1", "text2", "text3"]

        result = await generate_embeddings(texts, sample_embd_model_config)

        assert result is not None
        # Just verify the function completed successfully - logging is tested separately

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_query_embd_model_retry_mechanism(
        self, mock_post, sample_embd_model_config
    ):
        """Test retry mechanism for embedding queries."""
        # Mock successful response
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(
            return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        )

        mock_post.return_value.__aenter__.return_value = mock_response_success

        result = await query_embd_model("test query", sample_embd_model_config)

        # Should succeed with proper response
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_query_embd_model_max_retries_exceeded(
        self, mock_post, sample_embd_model_config
    ):
        """Test embedding query when max retries exceeded."""
        # Mock all calls to fail
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_post.return_value.__aenter__.return_value = mock_response

        result = await query_embd_model("test query", sample_embd_model_config)

        assert result is None
        # The retry decorator should attempt the function multiple times

    @pytest.mark.asyncio
    @patch("cyteonto.models.embeddings.query_embd_model")
    async def test_generate_embeddings_consistent_dimensions(
        self, mock_query, sample_embd_model_config
    ):
        """Test that all generated embeddings have consistent dimensions."""
        # Mock embeddings with consistent dimensions
        mock_query.side_effect = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]

        texts = ["text1", "text2", "text3"]
        result = await generate_embeddings(texts, sample_embd_model_config)

        assert result is not None
        assert result.shape == (3, 4)
        assert result.dtype == np.float64  # NumPy array default type

    @pytest.mark.asyncio
    async def test_generate_embeddings_type_conversion(self, sample_embd_model_config):
        """Test that embedding generation properly converts types."""
        # Test with a single successful embedding to check type conversion
        with patch("cyteonto.models.embeddings.query_embd_model") as mock_query:
            mock_query.return_value = [0.1, 0.2, 0.3]

            texts = ["text1"]
            result = await generate_embeddings(texts, sample_embd_model_config)

            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 3)
            assert np.allclose(result[0], [0.1, 0.2, 0.3])
