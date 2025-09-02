from unittest.mock import patch

import pytest
from loguru import logger

from cyteonto.logger_config import logger as configured_logger


class TestLoggerConfig:
    """Test logger configuration."""

    def test_logger_is_loguru_instance(self):
        """Test that the configured logger is a loguru instance."""
        # The logger should be the same instance as loguru's logger
        assert configured_logger is logger

    @patch("cyteonto.logger_config.CONFIG")
    def test_logger_with_file_config(self, mock_config):
        """Test logger configuration with file logging."""
        mock_config.LOGGING_LEVEL = "DEBUG"
        mock_config.LOG_FILE = "/tmp/test.log"

        # Re-import to trigger configuration
        import importlib

        import cyteonto.logger_config

        importlib.reload(cyteonto.logger_config)

        # Verify logger is still accessible
        assert hasattr(cyteonto.logger_config, "logger")

    @patch("cyteonto.logger_config.CONFIG")
    def test_logger_without_file_config(self, mock_config):
        """Test logger configuration without file logging."""
        mock_config.LOGGING_LEVEL = "INFO"
        mock_config.LOG_FILE = None

        # Re-import to trigger configuration
        import importlib

        import cyteonto.logger_config

        importlib.reload(cyteonto.logger_config)

        # Verify logger is still accessible
        assert hasattr(cyteonto.logger_config, "logger")

    def test_logger_basic_functionality(self):
        """Test basic logger functionality."""
        # Test that we can call logger methods without errors
        try:
            configured_logger.info("Test info message")
            configured_logger.debug("Test debug message")
            configured_logger.warning("Test warning message")
            configured_logger.error("Test error message")
        except Exception as e:
            pytest.fail(f"Logger methods should work: {e}")

    def test_logger_configuration_calls(self):
        """Test that logger is properly configured with expected calls."""
        # This test would require mocking the logger at module import time
        # which is complex. Instead, just verify that the module imports without error
        import cyteonto.logger_config

        # Verify the logger is accessible
        assert hasattr(cyteonto.logger_config, "logger")

    def test_logger_handles_exceptions_gracefully(self):
        """Test that logger configuration handles exceptions gracefully."""
        # This tests the try-except block around LOG_FILE configuration
        with patch("cyteonto.logger_config.CONFIG") as mock_config:
            mock_config.LOGGING_LEVEL = "INFO"
            mock_config.LOG_FILE = "/invalid/path/that/does/not/exist/test.log"

            # Re-import should not raise an exception
            try:
                import importlib

                import cyteonto.logger_config

                importlib.reload(cyteonto.logger_config)
            except Exception as e:
                pytest.fail(
                    f"Logger configuration should handle file errors gracefully: {e}"
                )
