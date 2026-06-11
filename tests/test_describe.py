"""Tests for cyteonto.describe helpers (prompt building, error formatting)."""

import pytest

from cyteonto.describe import (
    _build_prompt,
    _format_exception,
    _label_from_prompt,
    describe_cells,
)
from cyteonto.models import AgentUsage


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
