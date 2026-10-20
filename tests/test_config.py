"""Tests for cyteonto.config.Config (schema v3 defaults)."""

from typing import get_args

from pydantic_ai import ModelAPIError, ModelHTTPError

from cyteonto.config import Config
from cyteonto.models import EmbdProvider


class TestConfig:
    def test_schema_version(self):
        assert Config.SCHEMA_VERSION == "3.0"

    def test_primary_and_fallback_presets(self):
        cfg = Config()
        assert cfg.PRIMARY_LLM_PROVIDER == "nebius"
        assert cfg.PRIMARY_EMBEDDING_PROVIDER == "nebius"
        assert cfg.FALLBACK_LLM_PROVIDER == "fireworks"
        assert cfg.FALLBACK_EMBEDDING_PROVIDER == "openrouter"

    def test_provider_api_key_env_mapping(self):
        cfg = Config()
        assert cfg.PROVIDER_API_KEY_ENV["nebius"] == "NEBIUS_API_KEY"
        assert cfg.PROVIDER_API_KEY_ENV["fireworks"] == "FIREWORKS_API_KEY"
        assert cfg.PROVIDER_API_KEY_ENV["openrouter"] == "OPENROUTER_API_KEY"

    def test_every_embedding_provider_has_an_endpoint(self):
        cfg = Config()
        assert set(cfg._PROVIDER_URL) == set(get_args(EmbdProvider))
        for url in cfg._PROVIDER_URL.values():
            assert url.startswith(("http://", "https://"))

    def test_model_api_errors_are_retried(self):
        # pydantic-ai raises connection errors as ModelAPIError, the parent of
        # ModelHTTPError.
        assert ModelAPIError in Config._RETRYABLE_EXCEPTIONS
        assert issubclass(ModelHTTPError, Config._RETRYABLE_EXCEPTIONS)

    def test_result_columns_shape(self):
        cfg = Config()
        assert cfg.RESULT_COLUMNS[0] == "run_id"
        assert "cytescore_similarity" in cfg.RESULT_COLUMNS
        assert "similarity_method" in cfg.RESULT_COLUMNS
        assert len(cfg.RESULT_COLUMNS) == len(set(cfg.RESULT_COLUMNS))

    def test_openrouter_routing_default(self):
        cfg = Config()
        routing = cfg.OPENROUTER_DEEPINFRA_ROUTING
        assert routing["provider"]["order"] == ["deepinfra"]
