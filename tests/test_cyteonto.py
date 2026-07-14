"""Tests for cyteonto.cyteonto pure units (no live agents or network)."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from cyteonto import storage
from cyteonto.cyteonto import (
    CyteOnto,
    _api_key_for_provider,
    _hungarian_match_mean,
    _is_empty,
    _unique_parts,
)
from cyteonto.describe import (
    _normalize_decomposition,
    decompose_label,
    decompose_labels,
)
from cyteonto.models import AgentUsage, CellDescription, LabelDecomposition
from cyteonto.ontology import OntologyMapping


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


def _mock_mapping_for_compare(inst: CyteOnto) -> None:
    name_by_id = {
        "CL:0000001": ["T cell"],
        "CL:0000002": ["NK cell"],
        "CL:0000003": ["B cell"],
    }
    inst.mapping = Mock()
    inst.mapping.labels_for_id = lambda oid: name_by_id.get(oid, [])


class TestCompareEmptyHandling:
    @pytest.mark.asyncio
    async def test_empty_positions_get_blank_id_zero_score_empty_method(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        inst._resolve_label_parts = AsyncMock(
            side_effect=lambda labels, run_id, use_cache: {
                lbl: [lbl] for lbl in labels if lbl.strip()
            }
        )
        inst._embed_user_labels = AsyncMock(
            return_value=np.zeros((2, 2), dtype=np.float32)
        )
        inst._match = Mock(
            side_effect=[
                [("CL:0000001", 0.95)],
                [("CL:0000001", 0.95)],
                [("CL:0000002", 0.80)],
            ]
        )
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
        assert r["author_ontology_name"] == "T cell"
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
        assert r["author_ontology_name"] == "T cell"
        assert r["author_embedding_similarity"] == 0.95
        assert r["algorithm_ontology_id"] == ""
        assert r["algorithm_embedding_similarity"] == 0.0
        assert r["cytescore_similarity"] == 0.0
        assert r["similarity_method"] == "empty"

        # Only author label empty -> algorithm side kept, author blanked.
        r = row("algo1", 1)
        assert r["author_ontology_id"] == ""
        assert r["algorithm_ontology_id"] == "CL:0000002"
        assert r["algorithm_ontology_name"] == "NK cell"
        assert r["algorithm_embedding_similarity"] == 0.80
        assert r["cytescore_similarity"] == 0.0
        assert r["similarity_method"] == "empty"


class TestUniqueParts:
    def test_deduplicates_parts_across_labels(self):
        parts_map = {
            "A/B": ["A", "B"],
            "B": ["B"],
        }
        assert _unique_parts(["A/B", "B"], parts_map) == ["A", "B"]


class TestNormalizeDecomposition:
    def test_single_label_fallback_on_none(self):
        out = _normalize_decomposition("T cell", None)
        assert out == LabelDecomposition.single("T cell")

    def test_non_compound_forces_single_part(self):
        out = _normalize_decomposition(
            "T cell",
            LabelDecomposition(
                initialLabel="T cell", isCompound=False, parts=["T", "cell"]
            ),
        )
        assert out.parts == ["T cell"]
        assert out.isCompound is False

    def test_compound_requires_two_parts(self):
        out = _normalize_decomposition(
            "A/B",
            LabelDecomposition(initialLabel="A/B", isCompound=True, parts=["A"]),
        )
        assert out.parts == ["A/B"]

    def test_compound_keeps_cleaned_parts(self):
        out = _normalize_decomposition(
            "A/B",
            LabelDecomposition(initialLabel="A/B", isCompound=True, parts=["A", " B "]),
        )
        assert out.isCompound is True
        assert out.parts == ["A", "B"]


class TestDecomposeLabels:
    @pytest.mark.asyncio
    async def test_compound_input(self, mock_base_agent):
        compound = LabelDecomposition(
            initialLabel="AT2 cell–plasma cell doublet",
            isCompound=True,
            parts=["AT2 cell", "plasma cell"],
        )
        mock_agent = Mock()
        mock_agent.name = "LabelDecompositionAgent"
        mock_agent.model.model_name = "test-model"
        with (
            patch(
                "cyteonto.describe._build_decompose_agent",
                return_value=mock_agent,
            ),
            patch(
                "cyteonto.describe._run_decompose_once",
                new=AsyncMock(
                    return_value=(compound, {}, 1, 10, 5, 15),
                ),
            ),
        ):
            dec, usage = await decompose_label(
                mock_base_agent, "AT2 cell–plasma cell doublet"
            )
        assert dec.isCompound is True
        assert dec.parts == ["AT2 cell", "plasma cell"]
        assert usage.requests == 1

    @pytest.mark.asyncio
    async def test_simple_input(self, mock_base_agent):
        single = LabelDecomposition(
            initialLabel="T cell", isCompound=False, parts=["T cell"]
        )
        mock_agent = Mock()
        mock_agent.name = "LabelDecompositionAgent"
        mock_agent.model.model_name = "test-model"
        with (
            patch(
                "cyteonto.describe._build_decompose_agent",
                return_value=mock_agent,
            ),
            patch(
                "cyteonto.describe._run_decompose_once",
                new=AsyncMock(return_value=(single, {}, 1, 8, 4, 12)),
            ),
        ):
            dec, _ = await decompose_label(mock_base_agent, "T cell")
        assert dec.isCompound is False
        assert dec.parts == ["T cell"]

    @pytest.mark.asyncio
    async def test_semicolon_synonym_stays_single(self, mock_base_agent):
        single = LabelDecomposition(
            initialLabel="T cell;T lymphocyte",
            isCompound=False,
            parts=["T cell;T lymphocyte"],
        )
        mock_agent = Mock()
        mock_agent.name = "LabelDecompositionAgent"
        mock_agent.model.model_name = "test-model"
        with (
            patch(
                "cyteonto.describe._build_decompose_agent",
                return_value=mock_agent,
            ),
            patch(
                "cyteonto.describe._run_decompose_once",
                new=AsyncMock(return_value=(single, {}, 1, 8, 4, 12)),
            ),
        ):
            dec, _ = await decompose_label(mock_base_agent, "T cell;T lymphocyte")
        assert dec.isCompound is False
        assert dec.parts == ["T cell;T lymphocyte"]

    @pytest.mark.asyncio
    async def test_failure_falls_back_to_single_label(self, mock_base_agent):
        mock_agent = Mock()
        mock_agent.name = "LabelDecompositionAgent"
        mock_agent.model.model_name = "test-model"
        with (
            patch(
                "cyteonto.describe._build_decompose_agent",
                return_value=mock_agent,
            ),
            patch(
                "cyteonto.describe._run_decompose_once",
                new=AsyncMock(side_effect=RuntimeError("provider down")),
            ),
        ):
            dec, _ = await decompose_label(mock_base_agent, "T cell")
        assert dec == LabelDecomposition.single("T cell")

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self, mock_base_agent):
        async def fake_decompose(agent, label, **kwargs):
            if "doublet" in label:
                return (
                    LabelDecomposition(
                        initialLabel=label,
                        isCompound=True,
                        parts=["AT2 cell", "plasma cell"],
                    ),
                    AgentUsage(agentName="LabelDecompositionAgent"),
                )
            return (
                LabelDecomposition.single(label),
                AgentUsage(agentName="LabelDecompositionAgent"),
            )

        with patch("cyteonto.describe.decompose_label", side_effect=fake_decompose):
            results, _ = await decompose_labels(
                mock_base_agent,
                ["T cell", "AT2 cell–plasma cell doublet"],
            )
        assert results[0].parts == ["T cell"]
        assert results[1].parts == ["AT2 cell", "plasma cell"]


class TestHungarianMatchMean:
    def test_same_compound_2x2(self):
        scores = np.array([[0.98, 0.06], [0.05, 0.97]], dtype=np.float64)
        final, assignments = _hungarian_match_mean(scores)
        assert len(assignments) == 2
        assert final == pytest.approx(0.975, abs=0.001)

    def test_partial_overlap_3x2_with_coverage(self):
        scores = np.array([[0.98, 0.05], [0.06, 0.08], [0.05, 0.07]], dtype=np.float64)
        final, _ = _hungarian_match_mean(scores)
        assert final == pytest.approx(0.35, abs=0.01)

    def test_no_overlap_2x2(self):
        scores = np.array([[0.06, 0.05], [0.08, 0.07]], dtype=np.float64)
        final, _ = _hungarian_match_mean(scores)
        assert final == pytest.approx(0.065, abs=0.001)

    def test_unequal_1x2_applies_coverage(self):
        scores = np.array([[0.05, 0.92]], dtype=np.float64)
        final, assignments = _hungarian_match_mean(scores)
        assert assignments == [(0, 1)]
        assert final == pytest.approx(0.46, abs=0.001)

    def test_empty_matrix(self):
        final, assignments = _hungarian_match_mean(np.zeros((0, 2)))
        assert final == 0.0
        assert assignments == []


class TestCompareCompoundLabels:
    @pytest.mark.asyncio
    async def test_hungarian_2x1_with_coverage(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {"A1/A2": ["A1", "A2"]},
                {"G1": ["G1"]},
            ]
        )
        inst._embed_user_labels = AsyncMock(
            return_value=np.zeros((3, 2), dtype=np.float32)
        )
        inst._match = Mock(
            return_value=[
                ("CL:0000001", 0.9),
                ("CL:0000002", 0.8),
                ("CL:0000003", 0.7),
            ]
        )
        sim = Mock()
        sim.similarity.side_effect = [0.6, 0.2]
        inst._ensure_similarity = Mock(return_value=sim)

        df = await inst.compare(
            author_labels=["A1/A2"],
            algorithms={"algo0": ["G1"]},
            run_id="run-test",
        )

        row = df.iloc[0]
        assert row["author_label"] == "A1/A2"
        assert row["algorithm_label"] == "G1"
        assert row["cytescore_similarity"] == 0.3
        assert row["similarity_method"] == "cytescore_compound"
        assert row["author_ontology_id"] == "CL:0000001"
        assert row["algorithm_ontology_id"] == "CL:0000001"
        assert row["author_ontology_name"] == "T cell"
        assert sim.similarity.call_count == 2

    @pytest.mark.asyncio
    async def test_hungarian_2x2_same_compound(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {"A/B": ["A", "B"]},
                {"A/B": ["A", "B"]},
            ]
        )
        inst._embed_user_labels = AsyncMock(
            return_value=np.zeros((2, 2), dtype=np.float32)
        )
        inst._match = Mock(
            side_effect=[
                [("CL:0000001", 0.9), ("CL:0000002", 0.8)],
                [("CL:0000001", 0.85), ("CL:0000002", 0.75)],
            ]
        )
        sim = Mock()
        sim.similarity.side_effect = [0.98, 0.06, 0.05, 0.97]
        inst._ensure_similarity = Mock(return_value=sim)

        df = await inst.compare(
            author_labels=["A/B"],
            algorithms={"algo0": ["A/B"]},
            run_id="run-test",
        )

        row = df.iloc[0]
        assert row["cytescore_similarity"] == pytest.approx(0.975, abs=0.001)
        assert row["similarity_method"] == "cytescore_compound"
        assert row["author_ontology_id"] == "CL:0000001;CL:0000002"
        assert row["algorithm_ontology_id"] == "CL:0000001;CL:0000002"
        assert row["pair_index"] == 0

    @pytest.mark.asyncio
    async def test_compound_pair_index_not_shadowed(self):
        """Repeated author strings must keep distinct pair_index values."""
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        doublet = "AT2 cell-plasma cell doublet"
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {
                    doublet: ["AT2 cell", "plasma cell"],
                    "T cell": ["T cell"],
                },
                {
                    "Plasma cell": ["Plasma cell"],
                    "AT2 cell / Plasma cell": ["AT2 cell", "Plasma cell"],
                    "T cell": ["T cell"],
                },
            ]
        )
        inst._embed_user_labels = AsyncMock(
            return_value=np.zeros((3, 2), dtype=np.float32)
        )
        inst._match = Mock(
            return_value=[
                ("CL:0000001", 0.9),
                ("CL:0000002", 0.8),
                ("CL:0000003", 0.7),
            ]
        )
        sim = Mock()
        sim.similarity.return_value = 0.5
        inst._ensure_similarity = Mock(return_value=sim)

        df = await inst.compare(
            author_labels=[doublet, doublet, "T cell"],
            algorithms={
                "scenarios": ["Plasma cell", "AT2 cell / Plasma cell", "T cell"]
            },
            run_id="run-test",
        )

        assert df["pair_index"].tolist() == [0, 1, 2]
        assert df["algorithm_label"].tolist() == [
            "Plasma cell",
            "AT2 cell / Plasma cell",
            "T cell",
        ]

    @pytest.mark.asyncio
    async def test_non_compound_pair_unchanged(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {"T cell": ["T cell"]},
                {"B cell": ["B cell"]},
            ]
        )
        inst._embed_user_labels = AsyncMock(
            return_value=np.zeros((1, 2), dtype=np.float32)
        )
        inst._match = Mock(return_value=[("CL:0000001", 0.95)])
        sim = Mock()
        sim.similarity.return_value = 0.9
        inst._ensure_similarity = Mock(return_value=sim)

        df = await inst.compare(
            author_labels=["T cell"],
            algorithms={"algo0": ["B cell"]},
            run_id="run-test",
        )

        row = df.iloc[0]
        assert row["cytescore_similarity"] == 0.9
        assert row["similarity_method"] == "cytescore"
        assert row["author_ontology_name"] == "T cell"
        assert row["algorithm_ontology_name"] == "T cell"


class TestOntologyNames:
    def test_names_for_multiple_ids(self, sample_ontology_csv_file):
        inst = object.__new__(CyteOnto)
        inst.mapping = OntologyMapping(sample_ontology_csv_file)
        inst.mapping.load()
        inst._similarity = Mock()
        inst._similarity._find_class = Mock(return_value=None)

        joined = ";".join(["CL:0000001", "CL:0000002"])
        assert inst._ontology_names_for_ids(joined) == "T cell;B cell"

    def test_unknown_id_returns_empty_segment(self):
        inst = object.__new__(CyteOnto)
        inst.mapping = Mock()
        inst.mapping.labels_for_id = Mock(return_value=[])
        similarity = Mock()
        similarity._find_class = Mock(return_value=None)
        inst._ensure_similarity = Mock(return_value=similarity)

        assert inst._ontology_names_for_ids("CL:9999999") == ""
