"""Tests for cyteonto.cyteonto pure units (no live agents or network)."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from cyteonto import storage
from cyteonto.cyteonto import CyteOnto, _api_key_for_provider, _is_empty
from cyteonto.models import AgentUsage, CellDescription


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
        inst._ontology_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        inst._ontology_ids = ["CL:0000001", "CL:0000002"]
        return inst

    def test_exact_match(self):
        inst = self._instance()
        out = inst._match(np.array([[1.0, 0.0]], dtype=np.float32))
        assert out[0][0] == "CL:0000001"
        assert out[0][1] > 0.99

    def test_below_threshold_returns_none(self):
        inst = self._instance()
        out = inst._match(np.array([[0.7, 0.7]], dtype=np.float32), min_similarity=0.99)
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


class TestIsEmpty:
    def test_empty_string(self):
        assert _is_empty("") is True

    def test_whitespace_only(self):
        assert _is_empty("   ") is True
        assert _is_empty("\t\n") is True

    def test_non_empty(self):
        assert _is_empty("T cell") is False
        assert _is_empty(" NK cell ") is False


class TestEmbedUserLabelsSkipsEmpty:
    @pytest.mark.asyncio
    async def test_empty_labels_not_described_and_zero_vectors(self, monkeypatch):
        inst = object.__new__(CyteOnto)
        inst.paths = Mock()
        inst.paths.user_embeddings.return_value = Path("/tmp/emb.npz")
        inst.paths.user_descriptions.return_value = Path("/tmp/desc.json")
        inst.llm_key = Mock()
        inst.embd_key = Mock()
        inst.reasoning = False
        inst.usage = AgentUsage(agentName="CyteOnto")

        described = CellDescription(
            initialLabel="T cell",
            descriptiveName="CD4+ helper T lymphocyte",
            function="Coordinates immune responses",
            diseaseRelevance="Autoimmune disease",
            developmentalStage="Mature",
        )

        describe_mock = AsyncMock(
            return_value=([described], AgentUsage(agentName="CellDescriptionAgent"))
        )
        inst._describe_labels = describe_mock
        inst._embed_with_failover = AsyncMock(
            return_value=np.array([[1.0, 2.0]], dtype=np.float32)
        )

        monkeypatch.setattr(storage, "save_descriptions", lambda *a, **k: None)
        monkeypatch.setattr(storage, "save_user_embeddings", lambda *a, **k: None)

        result = await inst._embed_user_labels(
            labels=["T cell", "", "   "],
            run_id="run-test",
            kind="author",
            identifier="author",
            use_cache=False,
        )

        # Only the single non-empty label is sent for description generation.
        describe_mock.assert_awaited_once_with(["T cell"])
        # Only the non-empty description sentence is embedded.
        embed_arg = inst._embed_with_failover.await_args.args[0]
        assert embed_arg == [described.to_sentence()]

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[0], np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(result[1], np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(result[2], np.zeros(2, dtype=np.float32))


class TestCompareEmptyHandling:
    @pytest.mark.asyncio
    async def test_empty_positions_get_blank_id_zero_score_empty_method(self):
        inst = object.__new__(CyteOnto)
        inst._embed_user_labels = AsyncMock(
            return_value=np.zeros((2, 2), dtype=np.float32)
        )
        inst._match = Mock(return_value=[("CL:0000001", 0.95), ("CL:0000002", 0.80)])
        sim = Mock()
        sim.similarity.return_value = 0.9
        inst._ensure_similarity = Mock(return_value=sim)

        df = await inst.compare(
            author_labels=["T cell", ""],
            algorithms={"algo0": ["B cell", ""], "algo1": ["", "NK cell"]},
            run_id="run-test",
        )

        def row(algo: str, idx: int) -> dict:
            return df[(df.algorithm == algo) & (df.pair_index == idx)].iloc[0].to_dict()

        # Both labels present -> normal cytescore path.
        r = row("algo0", 0)
        assert r["author_ontology_id"] == "CL:0000001"
        assert r["algorithm_ontology_id"] == "CL:0000001"
        assert r["cytescore_similarity"] == 0.9
        assert r["similarity_method"] == "cytescore"

        # Both labels empty -> blank ids, zero scores, empty method.
        r = row("algo0", 1)
        assert r["author_ontology_id"] == ""
        assert r["algorithm_ontology_id"] == ""
        assert r["author_embedding_similarity"] == 0.0
        assert r["algorithm_embedding_similarity"] == 0.0
        assert r["cytescore_similarity"] == 0.0
        assert r["similarity_method"] == "empty"

        # Only algorithm label empty -> author side kept, algorithm blanked.
        r = row("algo1", 0)
        assert r["author_ontology_id"] == "CL:0000001"
        assert r["author_embedding_similarity"] == 0.95
        assert r["algorithm_ontology_id"] == ""
        assert r["algorithm_embedding_similarity"] == 0.0
        assert r["cytescore_similarity"] == 0.0
        assert r["similarity_method"] == "empty"

        # Only author label empty -> algorithm side kept, author blanked.
        r = row("algo1", 1)
        assert r["author_ontology_id"] == ""
        assert r["algorithm_ontology_id"] == "CL:0000002"
        assert r["algorithm_embedding_similarity"] == 0.80
        assert r["cytescore_similarity"] == 0.0
        assert r["similarity_method"] == "empty"
