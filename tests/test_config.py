"""Tests for cyteonto.config.Config (schema v3 defaults)."""

from cyteonto.config import Config


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

    def test_provider_urls_present_for_known_providers(self):
        cfg = Config()
        for provider in ("nebius", "deepinfra", "openrouter", "openai", "google", "ollama", "together"):
            assert provider in cfg._PROVIDER_URL
            assert cfg._PROVIDER_URL[provider].startswith(("http://", "https://"))

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
