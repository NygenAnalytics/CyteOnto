"""Tests for cyteonto.embed (HTTP embedding helpers and orchestration)."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from cyteonto import embed
from cyteonto.embed import (
    _build_payload,
    _extract_embedding,
    _headers,
    embed_texts,
)
from cyteonto.models import EmbdConfig


class TestHeaders:
    def test_ollama_no_headers(self):
        cfg = EmbdConfig(provider="ollama", model="m")
        assert _headers(cfg) == {}

    def test_google_uses_goog_key(self):
        cfg = EmbdConfig(provider="google", model="m", apiKey="k")
        headers = _headers(cfg)
        assert headers["x-goog-api-key"] == "k"
        assert headers["Content-Type"] == "application/json"

    def test_bearer_for_openai_style(self):
        cfg = EmbdConfig(provider="deepinfra", model="m", apiKey="k")
        assert _headers(cfg)["Authorization"] == "Bearer k"


class TestBuildPayload:
    def test_google_payload_shape(self):
        cfg = EmbdConfig(provider="google", model="gemini-embedding-001", apiKey="k")
        payload = _build_payload("hello", cfg)
        assert payload["model"] == "gemini-embedding-001"
        assert payload["content"]["parts"][0]["text"] == "hello"

    def test_openai_style_payload(self):
        cfg = EmbdConfig(provider="deepinfra", model="m", apiKey="k")
        payload = _build_payload("hello", cfg)
        assert payload["input"] == ["hello"]
        assert payload["encoding_format"] == "float"

    def test_openrouter_merges_model_settings(self):
        cfg = EmbdConfig(provider="openrouter", model="m", apiKey="k")
        payload = _build_payload("hello", cfg)
        assert "provider" in payload  # routing merged from modelSettings


class TestExtractEmbedding:
    def test_openai_style(self):
        cfg = EmbdConfig(provider="deepinfra", model="m")
        result = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        assert _extract_embedding(result, cfg) == [0.1, 0.2, 0.3]

    def test_google_values(self):
        cfg = EmbdConfig(provider="google", model="m")
        result = {"embeddings": {"values": [0.4, 0.5]}}
        assert _extract_embedding(result, cfg) == [0.4, 0.5]

    def test_empty_response(self):
        cfg = EmbdConfig(provider="deepinfra", model="m")
        assert _extract_embedding({"data": []}, cfg) == []
        assert _extract_embedding({}, cfg) == []


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        cfg = EmbdConfig(provider="deepinfra", model="m", apiKey="k")
        result = await embed_texts([], cfg)
        assert result is not None
        assert result.shape == (0, 0)

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_none(self):
        cfg = EmbdConfig(provider="deepinfra", model="m", apiKey="")
        assert await embed_texts(["x"], cfg) is None

    @pytest.mark.asyncio
    async def test_unsupported_provider_returns_none(self):
        # 'fireworks' is a valid literal but has no embedding endpoint configured.
        cfg = EmbdConfig(provider="fireworks", model="m", apiKey="k")
        assert await embed_texts(["x"], cfg) is None

    @pytest.mark.asyncio
    async def test_success_path(self):
        cfg = EmbdConfig(provider="deepinfra", model="m", apiKey="k", maxConcurrent=2)
        fake = AsyncMock(return_value={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
        with patch.object(embed, "_request_one", fake):
            result = await embed_texts(["a", "b"], cfg)
        assert result is not None
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_failed_request_returns_none(self):
        cfg = EmbdConfig(provider="deepinfra", model="m", apiKey="k")
        fake = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(embed, "_request_one", fake):
            result = await embed_texts(["a"], cfg)
        assert result is None
