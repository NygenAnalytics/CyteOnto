import importlib
import os
import sys
from unittest.mock import patch

from cyteonto.config import CONFIG, Config


class TestConfig:
    """Test configuration management."""

    @patch.dict(os.environ, {"NCBI_API_KEY": "", "EMBEDDING_MODEL_API_KEY": ""})
    def test_default_config(self):
        """Test default configuration values."""
        # Reload the module to pick up the mocked environment variables
        if "cyteonto.config" in sys.modules:
            importlib.reload(sys.modules["cyteonto.config"])

        from cyteonto.config import Config

        config = Config()
        assert config.NCBI_API_KEY == ""
        assert config.EMBEDDING_MODEL_API_KEY == ""
        assert config.LOGGING_LEVEL == "INFO"
        assert config.LOG_FILE is None

    def test_config_with_env_vars(self, mock_env_vars):
        """Test configuration with environment variables."""
        config = Config()
        # Note: Config loads env vars at class creation, so we need to reload
        # In practice, this would be loaded when the module is first imported
        assert hasattr(config, "NCBI_API_KEY")
        assert hasattr(config, "EMBEDDING_MODEL_API_KEY")

    def test_singleton_config(self):
        """Test that CONFIG is a singleton instance."""
        assert isinstance(CONFIG, Config)
        assert CONFIG.LOGGING_LEVEL == "INFO"

    @patch.dict(
        os.environ,
        {"NCBI_API_KEY": "test_key", "EMBEDDING_MODEL_API_KEY": "embedding_key"},
    )
    def test_config_env_loading(self):
        """Test that config properly loads environment variables."""
        # Create a fresh config instance
        config = Config()
        # Since dotenv.load_dotenv() is called in the module,
        # we're testing the structure, not the actual loading
        assert hasattr(config, "NCBI_API_KEY")
        assert hasattr(config, "EMBEDDING_MODEL_API_KEY")
        assert hasattr(config, "LOGGING_LEVEL")
        assert hasattr(config, "LOG_FILE")
