"""Tests for cyteonto.cyteonto pure units (no live agents or network)."""

from pathlib import Path

import numpy as np

from cyteonto.cyteonto import CyteOnto, _api_key_for_provider


class TestApiKeyForProvider:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "secret-key")
        assert _api_key_for_provider("nebius") == "secret-key"

    def test_falls_back_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        assert _api_key_for_provider("nebius", fallback="fb") == "fb"

    def test_unknown_provider_uses_fallback(self):
        assert _api_key_for_provider("unknown-provider", fallback="fb") == "fb"

    def test_returns_none_without_fallback(self, monkeypatch):
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        assert _api_key_for_provider("nebius") is None


class TestMethodFor:
    def test_no_matches(self):
        assert CyteOnto._method_for(None, None, 0.0) == "no_matches"

    def test_partial_match(self):
        assert CyteOnto._method_for("CL:1", None, 0.0) == "partial_match"
        assert CyteOnto._method_for(None, "CL:1", 0.0) == "partial_match"

    def test_cytescore_for_two_cl_ids(self):
        assert CyteOnto._method_for("CL:0000001", "CL:0000002", 0.5) == "cytescore"

    def test_string_similarity_for_non_cl(self):
        assert CyteOnto._method_for("X:1", "Y:2", 0.5) == "string_similarity"


class TestMatch:
    def _instance(self):
        # Bypass __init__ to test the pure matching logic in isolation.
        inst = object.__new__(CyteOnto)
        inst._ontology_embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        inst._ontology_ids = ["CL:0000001", "CL:0000002"]
        return inst

    def test_exact_match(self):
        inst = self._instance()
        out = inst._match(np.array([[1.0, 0.0]], dtype=np.float32))
        assert out[0][0] == "CL:0000001"
        assert out[0][1] > 0.99

    def test_below_threshold_returns_none(self):
        inst = self._instance()
        out = inst._match(
            np.array([[0.7, 0.7]], dtype=np.float32), min_similarity=0.99
        )
        assert out[0][0] is None

    def test_one_dimensional_query_reshaped(self):
        inst = self._instance()
        out = inst._match(np.array([0.0, 1.0], dtype=np.float32))
        assert len(out) == 1
        assert out[0][0] == "CL:0000002"


class TestCountFiles:
    def test_counts_only_files(self, temp_dir: Path):
        (temp_dir / "a.txt").write_text("x")
        nested = temp_dir / "sub"
        nested.mkdir()
        (nested / "b.txt").write_text("y")
        assert CyteOnto._count_files(temp_dir) == 2
