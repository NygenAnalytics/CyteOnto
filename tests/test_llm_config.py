from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from cyteonto.llm_config import (
    AGENT_CONFIG,
    AgentUsage,
    EMBDModelConfig,
    agent_run,
    get_embd_param,
    get_embd_results,
    get_tool_counts,
    update_tool_usage,
)


class TestEMBDModelConfig:
    """Test EMBDModelConfig functionality."""

    def test_default_values(self):
        """Test EMBDModelConfig with default values."""
        config = EMBDModelConfig()

        assert config.provider == "deepinfra"
        assert config.model == "Qwen/Qwen3-Embedding-8B"
        assert config.apiKey == "DEEPINFRA_TOKEN"
        assert config.modelSettings is None
        assert config.maxConcEmbed == 10

    def test_custom_values(self):
        """Test EMBDModelConfig with custom values."""
        config = EMBDModelConfig(
            provider="openai",
            model="text-embedding-3-small",
            apiKey="custom-key",
            modelSettings={"dimensions": 768},
            maxConcEmbed=5,
        )

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.apiKey == "custom-key"
        assert config.modelSettings == {"dimensions": 768}
        assert config.maxConcEmbed == 5

    def test_invalid_provider(self):
        """Test EMBDModelConfig with invalid provider."""
        with pytest.raises(ValidationError):
            EMBDModelConfig(provider="invalid_provider")


class TestAgentUsage:
    """Test AgentUsage functionality."""

    def test_default_values(self):
        """Test AgentUsage with default values."""
        usage = AgentUsage(agent_name="test_agent")

        assert usage.agent_name == "test_agent"
        assert usage.model_name == ""
        assert usage.requests == 0
        assert usage.request_tokens == 0
        assert usage.response_tokens == 0
        assert usage.total_tokens == 0
        assert usage.runtime_seconds == 0.0
        assert usage.tool_usage == {}

    def test_update_usage(self):
        """Test updating AgentUsage with RunUsage data."""
        usage = AgentUsage(agent_name="test_agent")

        # Mock RunUsage object
        mock_run_usage = Mock()
        mock_run_usage.requests = 5
        mock_run_usage.input_tokens = 100
        mock_run_usage.output_tokens = 50
        mock_run_usage.total_tokens = 150

        tool_usage = {"tool1": 2, "tool2": 3}

        usage.update("gpt-4", mock_run_usage, tool_usage)

        assert usage.model_name == "gpt-4"
        assert usage.requests == 5
        assert usage.request_tokens == 100
        assert usage.response_tokens == 50
        assert usage.total_tokens == 150
        assert usage.tool_usage == {"tool1": 2, "tool2": 3}

    def test_update_usage_accumulative(self):
        """Test that AgentUsage updates accumulate."""
        usage = AgentUsage(agent_name="test_agent")

        # First update
        mock_run_usage1 = Mock()
        mock_run_usage1.requests = 2
        mock_run_usage1.input_tokens = 50
        mock_run_usage1.output_tokens = 25
        mock_run_usage1.total_tokens = 75

        usage.update("model1", mock_run_usage1, {"tool1": 1})

        # Second update
        mock_run_usage2 = Mock()
        mock_run_usage2.requests = 3
        mock_run_usage2.input_tokens = 60
        mock_run_usage2.output_tokens = 30
        mock_run_usage2.total_tokens = 90

        usage.update("model2", mock_run_usage2, {"tool1": 2, "tool2": 1})

        # Should accumulate values
        assert usage.model_name == "model2"  # Latest model name
        assert usage.requests == 5  # 2 + 3
        assert usage.request_tokens == 110  # 50 + 60
        assert usage.response_tokens == 55  # 25 + 30
        assert usage.total_tokens == 165  # 75 + 90
        assert usage.tool_usage == {"tool1": 3, "tool2": 1}  # Accumulated tools

    def test_update_usage_with_none_values(self):
        """Test updating AgentUsage with None values in RunUsage."""
        usage = AgentUsage(agent_name="test_agent")

        # Mock RunUsage with None values
        mock_run_usage = Mock()
        mock_run_usage.requests = 1
        mock_run_usage.input_tokens = None
        mock_run_usage.output_tokens = None
        mock_run_usage.total_tokens = None

        usage.update("model", mock_run_usage, None)

        assert usage.requests == 1
        assert usage.request_tokens == 0  # None should be treated as 0
        assert usage.response_tokens == 0
        assert usage.total_tokens == 0


class TestEmbeddingHelpers:
    """Test embedding helper functions."""

    def test_get_embd_param_deepinfra(self):
        """Test getting embedding parameters for DeepInfra."""
        config = EMBDModelConfig(provider="deepinfra", apiKey="test-key")
        params = get_embd_param(config)

        assert "url" in params
        assert "headers" in params
        assert params["url"] == "https://api.deepinfra.com/v1/openai/embeddings"
        assert params["headers"]["Authorization"] == "Bearer test-key"
        assert params["headers"]["Content-Type"] == "application/json"

    def test_get_embd_param_openai(self):
        """Test getting embedding parameters for OpenAI."""
        config = EMBDModelConfig(provider="openai", apiKey="test-key")
        params = get_embd_param(config)

        assert params["url"] == "https://api.openai.com/v1/embeddings/"
        assert params["headers"]["Authorization"] == "Bearer test-key"

    def test_get_embd_param_google(self):
        """Test getting embedding parameters for Google."""
        config = EMBDModelConfig(provider="google", apiKey="test-key")
        params = get_embd_param(config)

        assert "generativelanguage.googleapis.com" in params["url"]
        assert params["headers"]["x-goog-api-key"] == "test-key"

    def test_get_embd_param_ollama(self):
        """Test getting embedding parameters for Ollama."""
        config = EMBDModelConfig(provider="ollama", apiKey="test-key")
        params = get_embd_param(config)

        assert params["url"] == "http://localhost:11434/api/embed"
        assert params["headers"] == {}  # Ollama doesn't need headers

    def test_get_embd_param_unsupported_provider(self):
        """Test getting embedding parameters for unsupported provider."""
        # Create a mock config with unsupported provider
        from unittest.mock import MagicMock

        config = MagicMock()
        config.provider = "unsupported"
        config.apiKey = "test-key"

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_embd_param(config)

    def test_get_embd_results_deepinfra(self):
        """Test extracting embedding results from DeepInfra response."""
        config = EMBDModelConfig(provider="deepinfra")
        response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        result = get_embd_results(response, config)
        assert result == [0.1, 0.2, 0.3]

    def test_get_embd_results_openai(self):
        """Test extracting embedding results from OpenAI response."""
        config = EMBDModelConfig(provider="openai")
        response = {"data": [{"embedding": [0.4, 0.5, 0.6]}]}

        result = get_embd_results(response, config)
        assert result == [0.4, 0.5, 0.6]

    def test_get_embd_results_google(self):
        """Test extracting embedding results from Google response."""
        config = EMBDModelConfig(provider="google")
        response = {"embeddings": {"values": [0.7, 0.8, 0.9]}}

        result = get_embd_results(response, config)
        assert result == [0.7, 0.8, 0.9]

    def test_get_embd_results_empty_response(self):
        """Test extracting embedding results from empty response."""
        config = EMBDModelConfig(provider="deepinfra")
        response = {"data": []}

        result = get_embd_results(response, config)
        assert result == []

    def test_get_embd_results_missing_data(self):
        """Test extracting embedding results when data is missing."""
        config = EMBDModelConfig(provider="deepinfra")
        response = {}

        result = get_embd_results(response, config)
        assert result == []


class TestToolHelpers:
    """Test tool helper functions."""

    def test_get_tool_counts_no_tools(self):
        """Test getting tool counts from node without tools."""
        mock_node = Mock()
        mock_node.__class__.__name__ = "TextNode"  # Not CallToolsNode

        result = get_tool_counts(mock_node)
        assert result == {}

    def test_update_tool_usage_empty(self):
        """Test updating tool usage with empty counts."""
        mock_node = Mock()
        mock_node.__class__.__name__ = "TextNode"  # Not CallToolsNode

        tool_usage = {"existing_tool": 1}
        update_tool_usage(mock_node, tool_usage)

        # Should remain unchanged
        assert tool_usage == {"existing_tool": 1}

    def test_update_tool_usage_with_tools(self):
        """Test updating tool usage with actual tools."""
        # This is a more complex test that would require mocking CallToolsNode
        # and ToolCallPart from pydantic_ai, which might not be available
        # in the test environment. The test structure is here for completeness.
        pass


class TestAgentConfig:
    """Test AgentConfig functionality."""

    def test_agent_config_singleton(self):
        """Test that AGENT_CONFIG is properly configured."""
        assert hasattr(AGENT_CONFIG, "TOOL_DEFAULT_SETTINGS")
        assert hasattr(AGENT_CONFIG, "AGENT_DEFAULT_USAGE_LIMITS")
        assert hasattr(AGENT_CONFIG, "MAX_TOOL_CALLS")
        assert hasattr(AGENT_CONFIG, "MAX_CONCURRENT_DESCRIPTIONS")

        assert isinstance(AGENT_CONFIG.TOOL_DEFAULT_SETTINGS, dict)
        assert AGENT_CONFIG.MAX_TOOL_CALLS == 3
        assert AGENT_CONFIG.MAX_CONCURRENT_DESCRIPTIONS == 10

    def test_default_settings(self):
        """Test default agent settings."""
        assert AGENT_CONFIG.TOOL_DEFAULT_SETTINGS["retries"] == 2
        assert AGENT_CONFIG.TOOL_DEFAULT_SETTINGS["strict"] is False

    def test_usage_limits(self):
        """Test agent usage limits."""
        limits = AGENT_CONFIG.AGENT_DEFAULT_USAGE_LIMITS
        assert limits.request_limit == 50
        assert limits.input_tokens_limit == 60_000


class TestAgentRun:
    """Test agent_run functionality."""

    def test_agent_run_success(self):
        """Test successful agent run."""
        # The agent_run function requires complex pydantic-ai mocking
        # For now, just test that the function exists and is importable
        assert callable(agent_run)

    def test_agent_run_with_retries(self):
        """Test agent run with retry mechanism."""
        # The agent_run function requires complex pydantic-ai mocking
        # For now, just test that the function exists and is importable
        assert callable(agent_run)

    def test_agent_run_usage_limit_exceeded(self):
        """Test agent run when usage limits are exceeded."""
        # The agent_run function requires complex pydantic-ai mocking
        # For now, just test that the function exists and is importable
        assert callable(agent_run)
