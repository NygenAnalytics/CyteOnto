"""Tests for cyteonto.describe helpers (prompt building, error formatting)."""

from unittest.mock import Mock

import pytest
from pydantic_ai import ModelAPIError

from cyteonto.describe import (
    _build_prompt,
    _format_exception,
    _label_from_prompt,
    _run_decompose_once,
    _run_once,
    describe_cells,
)
from cyteonto.models import AgentUsage


def _failed_attempt(error: BaseException) -> Mock:
    """A tenacity retry state whose attempt raised ``error``."""
    state = Mock()
    state.outcome.failed = True
    state.outcome.exception.return_value = error
    return state


class TestRetries:
    @pytest.mark.parametrize("run", [_run_once, _run_decompose_once])
    def test_model_api_errors_are_retried(self, run):
        # pydantic-ai raises the connection errors of a model as ModelAPIError.
        error = ModelAPIError(model_name="m", message="connection refused")
        assert run.retry.retry(_failed_attempt(error))

    @pytest.mark.parametrize("run", [_run_once, _run_decompose_once])
    def test_other_errors_are_not_retried(self, run):
        assert not run.retry.retry(_failed_attempt(KeyError("x")))


class TestBuildPrompt:
    def test_includes_label(self):
        prompt = _build_prompt("T helper cell", use_pubmed=True)
        assert "T helper cell" in prompt

    def test_pubmed_enabled_mentions_tool(self):
        prompt = _build_prompt("T cell", use_pubmed=True)
        assert "get_pubmed_abstracts" in prompt

    def test_pubmed_disabled_forbids_tools(self):
        prompt = _build_prompt("T cell", use_pubmed=False)
        assert "Do not call any tools." in prompt
        assert "get_pubmed_abstracts" not in prompt


class TestLabelFromPrompt:
    def test_extracts_label_from_built_prompt(self):
        prompt = _build_prompt("CD8+ T cell", use_pubmed=False)
        assert _label_from_prompt(prompt) == "CD8+ T cell"

    def test_missing_marker_returns_placeholder(self):
        assert _label_from_prompt("no marker here") == "?"


class TestFormatException:
    def test_generic_exception(self):
        assert _format_exception(ValueError("bad input")) == "ValueError: bad input"

    def test_type_name_included(self):
        assert "KeyError" in _format_exception(KeyError("missing"))


class TestDescribeCells:
    @pytest.mark.asyncio
    async def test_empty_labels_returns_empty(self, mock_base_agent):
        descs, usage = await describe_cells(mock_base_agent, [])
        assert descs == []
        assert isinstance(usage, AgentUsage)
