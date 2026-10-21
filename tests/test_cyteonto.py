"""Tests for cyteonto.cyteonto pure units (no live agents or network)."""

import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from pydantic_ai.exceptions import UsageLimitExceeded

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
from cyteonto.models import (
    AgentUsage,
    CellDescription,
    EmbdConfig,
    LabelDecomposition,
    LlmConfig,
)
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
    async def test_empty_labels_not_described_and_zero_vectors(
        self, monkeypatch, tmp_path
    ):
        inst = object.__new__(CyteOnto)
        inst.paths = Mock()
        inst.paths.user_embeddings.return_value = tmp_path / "emb.npz"
        inst.paths.user_descriptions.return_value = tmp_path / "desc.json"
        inst.llm_key = Mock()
        inst.embd_key = Mock()
        inst._embd_keys = [inst.embd_key]
        inst._embedding_switched = False
        inst._embedding_primary_used = False
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
    # The embedding model that compare() checks for a failover.
    inst.embd_key = Mock()
    inst._embedding_primary_used = False


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
        assert out.parts == ["a", "b"]

    def test_compound_lowercases_parts(self):
        out = _normalize_decomposition(
            "a/b",
            LabelDecomposition(
                initialLabel="a/b",
                isCompound=True,
                parts=["AT2 cell", "Plasma cell"],
            ),
        )
        assert out.parts == ["at2 cell", "plasma cell"]


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
        assert dec.parts == ["at2 cell", "plasma cell"]
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
    async def test_failure_returns_no_parts(self, mock_base_agent):
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
        assert dec == LabelDecomposition.failed("T cell")
        assert dec.parts == []

    @pytest.mark.asyncio
    async def test_a_label_over_the_usage_limits_stays_whole(self, mock_base_agent):
        mock_agent = Mock()
        mock_agent.name = "LabelDecompositionAgent"
        with (
            patch(
                "cyteonto.describe._build_decompose_agent",
                return_value=mock_agent,
            ),
            patch(
                "cyteonto.describe._run_decompose_once",
                new=AsyncMock(side_effect=UsageLimitExceeded("request_limit of 50")),
            ),
        ):
            dec, _ = await decompose_label(mock_base_agent, "T cell")
        # Another try would exceed the limits again, so the split is final.
        assert dec == LabelDecomposition.single("T cell")

    @pytest.mark.asyncio
    async def test_a_crashed_task_leaves_its_label_without_parts(self, mock_base_agent):
        async def fake_decompose(agent, label, **kwargs):
            if label == "bad":
                raise RuntimeError("crash")
            return LabelDecomposition.single(label), AgentUsage()

        with patch("cyteonto.describe.decompose_label", new=fake_decompose):
            results, _ = await decompose_labels(mock_base_agent, ["good", "bad"])

        assert results == [
            LabelDecomposition.single("good"),
            LabelDecomposition.failed("bad"),
        ]

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
                {"a1/a2": ["a1", "a2"]},
                {"g1": ["g1"]},
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
            compound_scoring="hungarian_mean",
        )

        row = df.iloc[0]
        assert row["author_label"] == "a1/a2"
        assert row["algorithm_label"] == "g1"
        assert row["cytescore_similarity"] == 0.3
        assert row["similarity_method"] == "cytescore_compound"
        assert row["author_ontology_id"] == "CL:0000001;CL:0000002"
        assert row["algorithm_ontology_id"] == "CL:0000001"
        assert row["author_ontology_name"] == "T cell;NK cell"
        assert row["author_embedding_similarity"] == "0.9;0.8"
        assert row["algorithm_embedding_similarity"] == 0.9
        assert sim.similarity.call_count == 2

    @pytest.mark.asyncio
    async def test_max_2x1_default(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {"a1/a2": ["a1", "a2"]},
                {"g1": ["g1"]},
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
        assert row["cytescore_similarity"] == 0.6
        assert row["similarity_method"] == "cytescore_compound"

    @pytest.mark.asyncio
    async def test_invalid_compound_scoring_raises(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)

        with pytest.raises(ValueError, match="compound_scoring"):
            await inst.compare(
                author_labels=["T cell"],
                algorithms={"algo0": ["B cell"]},
                run_id="run-test",
                compound_scoring="mean",  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_hungarian_2x2_same_compound(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {"a/b": ["a", "b"]},
                {"a/b": ["a", "b"]},
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
            compound_scoring="hungarian_mean",
        )

        row = df.iloc[0]
        assert row["cytescore_similarity"] == pytest.approx(0.975, abs=0.001)
        assert row["similarity_method"] == "cytescore_compound"
        assert row["author_ontology_id"] == "CL:0000001;CL:0000002"
        assert row["algorithm_ontology_id"] == "CL:0000001;CL:0000002"
        assert row["author_embedding_similarity"] == "0.9;0.8"
        assert row["algorithm_embedding_similarity"] == "0.85;0.75"
        assert row["pair_index"] == 0

    @pytest.mark.asyncio
    async def test_compound_pair_index_not_shadowed(self):
        """Repeated author strings must keep distinct pair_index values."""
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        doublet = "at2 cell-plasma cell doublet"
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {
                    doublet: ["at2 cell", "plasma cell"],
                    "t cell": ["t cell"],
                },
                {
                    "plasma cell": ["plasma cell"],
                    "at2 cell / plasma cell": ["at2 cell", "plasma cell"],
                    "t cell": ["t cell"],
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
            "plasma cell",
            "at2 cell / plasma cell",
            "t cell",
        ]

    @pytest.mark.asyncio
    async def test_non_compound_pair_unchanged(self):
        inst = object.__new__(CyteOnto)
        _mock_mapping_for_compare(inst)
        inst._resolve_label_parts = AsyncMock(
            side_effect=[
                {"t cell": ["t cell"]},
                {"b cell": ["b cell"]},
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


def _instance(
    tmp_path: Path,
    fallback_embedding: EmbdConfig | None = None,
    embedding: EmbdConfig | None = None,
) -> CyteOnto:
    """A CyteOnto with a mock agent on a two-term ontology in ``tmp_path``."""
    cell_ontology = tmp_path / "data" / "cell_ontology"
    cell_ontology.mkdir(parents=True, exist_ok=True)
    (cell_ontology / "cell_to_cell_ontology.csv").write_text(
        "ontology_id,label\nCL:0000001,T cell\nCL:0000002,B cell\n"
    )
    (cell_ontology / "cell_to_cell_ontology_enriched.csv").write_text(
        "ontology_id,label,label_normalized\n"
        "CL:0000001,T cell,t cell\nCL:0000002,B cell,b cell\n"
    )
    (cell_ontology / "cl.owl").write_text("")
    agent = Mock()
    agent.model.model_name = "llm-model"
    return CyteOnto(
        agent=agent,
        embedding=embedding
        or EmbdConfig(provider="nebius", model="primary-model", apiKey="k"),
        llm=LlmConfig(provider="nebius", model="llm-model"),
        data_dir=tmp_path / "data",
        fallback_embedding=fallback_embedding,
    )


def _description(label: str) -> CellDescription:
    return CellDescription(
        initialLabel=label,
        descriptiveName=label,
        function="Function.",
        diseaseRelevance="Not established",
        developmentalStage="Not established",
    )


def _describe_terms(inst: CyteOnto) -> None:
    descriptions = {
        "CL:0000001": _description("t cell"),
        "CL:0000002": _description("b cell"),
    }
    storage.save_descriptions(
        inst.paths.ontology_descriptions(inst.llm_key), descriptions, inst.llm_key
    )


def _save_ontology_embeddings(inst: CyteOnto, embedding: EmbdConfig) -> Path:
    key = embedding.to_artifact_key()
    path = inst.paths.ontology_embeddings(inst.llm_key, key)
    storage.save_ontology_embeddings(
        path,
        np.eye(2, 3, dtype=np.float32),
        ["CL:0000001", "CL:0000002"],
        inst.llm_key,
        key,
    )
    return path


# Its description mentions "b cell" and its label does not, so the two embed apart.
_PLASMABLAST = CellDescription(
    initialLabel="plasmablast",
    descriptiveName="antibody-secreting b cell",
    function="Secretes antibodies.",
    diseaseRelevance="Not established",
    developmentalStage="Not established",
)


def _embedder(calls: list[tuple[str, int]], fails):
    """An embed_texts stand-in that returns None when ``fails(provider, texts)``.

    Texts that mention "b cell" get the second unit vector and all others the
    first, so that labels match the term whose description shares their wording.
    """

    async def fake_embed_texts(texts, cfg):
        calls.append((cfg.provider, len(texts)))
        if fails(cfg.provider, texts):
            return None
        return np.array(
            [[0.0, 1.0, 0.0] if "b cell" in t else [1.0, 0.0, 0.0] for t in texts],
            dtype=np.float32,
        )

    return fake_embed_texts


_FALLBACK = EmbdConfig(
    provider="openrouter", model="fallback-model", apiKey="k", modelSettings={}
)


class TestEmbeddingFailover:
    @pytest.mark.asyncio
    async def test_switches_model_after_embedding_the_missing_ontology(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: provider == "nebius")
        with patch("cyteonto.cyteonto.embed_texts", fake):
            first = await inst._embed_with_failover(["a label"])
            second = await inst._embed_with_failover(["another label"])

        assert first is not None and second is not None
        # The fallback model embeds the labels and then the two ontology
        # descriptions; afterwards the primary is not tried again.
        assert calls == [
            ("nebius", 1),
            ("openrouter", 1),
            ("openrouter", 2),
            ("openrouter", 1),
        ]
        assert inst.embedding is inst.fallback_embedding
        assert inst.embd_key == _FALLBACK.to_artifact_key()
        loaded = storage.load_ontology_embeddings(
            inst.paths.ontology_embeddings(inst.llm_key, inst.embd_key)
        )
        assert loaded is not None
        assert loaded[1] == ["CL:0000001", "CL:0000002"]

    @pytest.mark.asyncio
    async def test_after_the_switch_a_failing_fallback_is_tried_once(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        calls: list[tuple[str, int]] = []
        down = {"nebius"}
        fake = _embedder(calls, lambda provider, texts: provider in down)
        with patch("cyteonto.cyteonto.embed_texts", fake):
            await inst._embed_with_failover(["a label"])
            down.add("openrouter")
            calls.clear()
            assert await inst._embed_with_failover(["another label"]) is None

        assert calls == [("openrouter", 1)]

    @pytest.mark.asyncio
    async def test_a_failed_ontology_embedding_keeps_the_primary(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        primary, primary_key = inst.embedding, inst.embd_key
        calls: list[tuple[str, int]] = []
        down = {"nebius"}
        # The fallback embeds the label but fails on the ontology descriptions.
        fake = _embedder(
            calls, lambda provider, texts: provider in down or len(texts) > 1
        )
        with patch("cyteonto.cyteonto.embed_texts", fake):
            with pytest.raises(RuntimeError, match="ontology descriptions"):
                await inst._embed_with_failover(["a label"])
            assert inst.embedding is primary
            assert inst.embd_key == primary_key
            assert inst.model_pair_usage.embeddingTier == "primary"
            # Once the primary recovers, the next call uses it.
            down.clear()
            calls.clear()
            assert await inst._embed_with_failover(["a label"]) is not None

        assert calls == [("nebius", 1)]

    @pytest.mark.asyncio
    async def test_failover_while_embedding_the_ontology_embeds_it_once(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        primary_key = inst.embd_key
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: provider == "nebius")
        with patch("cyteonto.cyteonto.embed_texts", fake):
            await inst._prepare_ontology(force_regenerate=False)

        # One term checks the fallback, then all terms are embedded with it.
        assert calls == [("nebius", 2), ("openrouter", 1), ("openrouter", 2)]
        fallback_path = inst.paths.ontology_embeddings(
            inst.llm_key, _FALLBACK.to_artifact_key()
        )
        assert fallback_path.exists()
        assert not inst.paths.ontology_embeddings(inst.llm_key, primary_key).exists()

    @pytest.mark.asyncio
    async def test_failover_while_embedding_the_ontology_reuses_the_fallback_file(
        self, tmp_path
    ):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        _save_ontology_embeddings(inst, _FALLBACK)
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: provider == "nebius")
        with patch("cyteonto.cyteonto.embed_texts", fake):
            await inst._prepare_ontology(force_regenerate=False)

        assert calls == [("nebius", 2), ("openrouter", 1)]
        assert inst.embd_key == _FALLBACK.to_artifact_key()

    @pytest.mark.asyncio
    async def test_a_failing_fallback_is_not_switched_to_for_its_ontology_file(
        self, tmp_path
    ):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        _save_ontology_embeddings(inst, _FALLBACK)
        primary_key = inst.embd_key
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: True)
        with patch("cyteonto.cyteonto.embed_texts", fake):
            with pytest.raises(RuntimeError, match="ontology embeddings"):
                await inst._prepare_ontology(force_regenerate=False)

        assert calls == [("nebius", 2), ("openrouter", 1)]
        assert inst.embd_key == primary_key

    @pytest.mark.asyncio
    async def test_failover_without_ontology_descriptions_raises(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: provider == "nebius")
        with patch("cyteonto.cyteonto.embed_texts", fake):
            with pytest.raises(FileNotFoundError, match="from_config"):
                await inst._embed_with_failover(["a label"])

        assert inst.embedding.provider == "nebius"

    @pytest.mark.asyncio
    async def test_a_failed_save_of_the_ontology_embeddings_keeps_the_primary(
        self, tmp_path, monkeypatch
    ):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        primary_key = inst.embd_key
        monkeypatch.setattr(storage, "save_ontology_embeddings", lambda *a, **k: False)
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: provider == "nebius")
        with patch("cyteonto.cyteonto.embed_texts", fake):
            with pytest.raises(RuntimeError, match="Failed to save"):
                await inst._embed_with_failover(["a label"])

        assert inst.embd_key == primary_key

    @pytest.mark.asyncio
    async def test_an_unreadable_fallback_file_is_made_again(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        path = inst.paths.ontology_embeddings(inst.llm_key, _FALLBACK.to_artifact_key())
        path.write_bytes(b"truncated")
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: provider == "nebius")
        with patch("cyteonto.cyteonto.embed_texts", fake):
            await inst._embed_with_failover(["a label"])

        assert calls == [("nebius", 1), ("openrouter", 1), ("openrouter", 2)]
        assert storage.load_ontology_embeddings(path) is not None

    @pytest.mark.asyncio
    async def test_a_fallback_equal_to_the_primary_is_not_tried(self, tmp_path):
        same = EmbdConfig(provider="nebius", model="primary-model", apiKey="k")
        inst = _instance(tmp_path, fallback_embedding=same)
        calls: list[tuple[str, int]] = []
        fake = _embedder(calls, lambda provider, texts: True)
        with patch("cyteonto.cyteonto.embed_texts", fake):
            assert await inst._embed_with_failover(["a label"]) is None

        assert calls == [("nebius", 1)]


def _describe_mock() -> AsyncMock:
    return AsyncMock(
        side_effect=lambda **kw: (
            [_description(label) for label in kw["labels"]],
            AgentUsage(),
        )
    )


class TestEmbeddingTier:
    @pytest.mark.asyncio
    async def test_cached_embeddings_of_the_primary_count_as_primary(self, tmp_path):
        calls: list[tuple[str, int]] = []
        down: set[str] = set()
        fake = _embedder(calls, lambda provider, texts: provider in down)
        with (
            patch("cyteonto.cyteonto.describe_cells", _describe_mock()),
            patch("cyteonto.cyteonto.embed_texts", fake),
        ):
            first = _instance(tmp_path, fallback_embedding=_FALLBACK)
            _describe_terms(first)
            await first._embed_user_labels(["t cell"], "run", "author", "author", True)
            # A new instance matches the author labels with the cached embeddings
            # of the primary model, then fails over for the algorithm's.
            inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
            await inst._embed_user_labels(["t cell"], "run", "author", "author", True)
            down.add("nebius")
            await inst._embed_user_labels(["b cell"], "run", "algorithm", "algo", True)

        assert inst.model_pair_usage.embeddingTier == "mixed"

    @pytest.mark.asyncio
    async def test_embedding_the_ontology_does_not_count(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        calls: list[tuple[str, int]] = []
        # The primary embeds the two terms, then fails for the label.
        fake = _embedder(
            calls, lambda provider, texts: provider == "nebius" and len(texts) == 1
        )
        with (
            patch("cyteonto.cyteonto.describe_cells", _describe_mock()),
            patch("cyteonto.cyteonto.embed_texts", fake),
        ):
            await inst._prepare_ontology(force_regenerate=False)
            await inst._embed_user_labels(["t cell"], "run", "author", "author", True)

        assert inst.model_pair_usage.embeddingTier == "fallback"
        assert inst.model_pair_usage.embedding == _FALLBACK


class TestOntologyEmbeddingFiles:
    @pytest.mark.asyncio
    async def test_new_term_descriptions_make_the_old_embeddings_stale(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        storage.save_descriptions(
            inst.paths.ontology_descriptions(inst.llm_key),
            {"CL:0000001": _description("t cell")},
            inst.llm_key,
        )
        for cfg in (inst.embedding, _FALLBACK):
            _save_ontology_embeddings(inst, cfg)
        calls: list[tuple[str, int]] = []
        down = {"nebius", "openrouter"}
        fake = _embedder(calls, lambda provider, texts: provider in down)
        with (
            patch("cyteonto.cyteonto.describe_cells", _describe_mock()),
            patch("cyteonto.cyteonto.embed_texts", fake),
        ):
            with pytest.raises(RuntimeError):
                await inst._prepare_ontology(force_regenerate=False)
            # Both files were made before the second term had a description. They
            # can still be loaded until new ones replace them, but are not reused.
            inst._load_ontology_embeddings()
            down.clear()
            calls.clear()
            await inst._prepare_ontology(force_regenerate=False)
            assert calls == [("nebius", 2)]
            # At a failover, the fallback model's file is made again too.
            down.add("nebius")
            calls.clear()
            await inst._embed_with_failover(["a label"])

        assert calls == [("nebius", 1), ("openrouter", 1), ("openrouter", 2)]

    @pytest.mark.asyncio
    async def test_files_made_after_the_descriptions_are_reused(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        for cfg in (inst.embedding, _FALLBACK):
            _save_ontology_embeddings(inst, cfg)
        calls: list[tuple[str, int]] = []
        down: set[str] = set()
        fake = _embedder(calls, lambda provider, texts: provider in down)
        with patch("cyteonto.cyteonto.embed_texts", fake):
            await inst._prepare_ontology(force_regenerate=False)
            down.add("nebius")
            await inst._embed_with_failover(["a label"])

        # Like the files that setup downloads, no term is embedded again.
        assert calls == [("nebius", 1), ("openrouter", 1)]

    @pytest.mark.asyncio
    async def test_another_models_file_made_before_new_descriptions_is_made_again(
        self, tmp_path
    ):
        inst = _instance(tmp_path)
        other = _instance(tmp_path, embedding=_FALLBACK)
        storage.save_descriptions(
            inst.paths.ontology_descriptions(inst.llm_key),
            {"CL:0000001": _description("t cell")},
            inst.llm_key,
        )
        _save_ontology_embeddings(inst, _FALLBACK)
        calls: list[tuple[str, int]] = []
        with (
            patch("cyteonto.cyteonto.describe_cells", _describe_mock()),
            patch("cyteonto.cyteonto.embed_texts", _embedder(calls, lambda *a: False)),
        ):
            await inst._prepare_ontology(force_regenerate=False)
            await other._prepare_ontology(force_regenerate=False)

        # The other model's file predates the second term's description.
        assert calls == [("nebius", 2), ("openrouter", 2)]

    @pytest.mark.asyncio
    async def test_an_unreadable_file_is_made_again(self, tmp_path):
        inst = _instance(tmp_path)
        _describe_terms(inst)
        path = inst.paths.ontology_embeddings(inst.llm_key, inst.embd_key)
        path.write_bytes(b"truncated")
        calls: list[tuple[str, int]] = []
        with patch("cyteonto.cyteonto.embed_texts", _embedder(calls, lambda *a: False)):
            await inst._prepare_ontology(force_regenerate=False)

        assert calls == [("nebius", 2)]
        assert storage.load_ontology_embeddings(path) is not None

    def test_a_failed_save_keeps_the_old_file(self, tmp_path, monkeypatch):
        inst = _instance(tmp_path)
        path = _save_ontology_embeddings(inst, inst.embedding)

        def broken_savez(file, **arrays):
            # Writes part of the file, as on a full disk, then fails.
            with open(file, "wb") if isinstance(file, Path) else nullcontext(file) as f:
                f.write(b"partial")
            raise OSError("No space left on device")

        monkeypatch.setattr(storage.np, "savez_compressed", broken_savez)
        saved = storage.save_ontology_embeddings(
            path,
            np.zeros((2, 3), dtype=np.float32),
            ["CL:0000001", "CL:0000002"],
            inst.llm_key,
            inst.embd_key,
        )

        assert not saved
        loaded = storage.load_ontology_embeddings(path)
        assert loaded is not None
        np.testing.assert_array_equal(loaded[0], np.eye(2, 3))
        assert [p.name for p in path.parent.iterdir()] == [path.name]

    @pytest.mark.asyncio
    async def test_force_regenerate_removes_the_fallback_embeddings(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        fallback_path = _save_ontology_embeddings(inst, _FALLBACK)
        # No term gets a description this time.
        describe = AsyncMock(
            side_effect=lambda **kw: (
                [CellDescription.blank(label=label) for label in kw["labels"]],
                AgentUsage(),
            )
        )
        calls: list[tuple[str, int]] = []
        with (
            patch("cyteonto.cyteonto.describe_cells", describe),
            patch("cyteonto.cyteonto.embed_texts", _embedder(calls, lambda *a: False)),
        ):
            await inst._prepare_ontology(force_regenerate=True)

        assert not fallback_path.exists()


class _FakeSimilarity:
    """Scores every pair 0.75 with the fallback model's embeddings, else 0.25."""

    def __init__(self, owl_path, embeddings_path):
        self.score = 0.75 if "fallback-model" in embeddings_path.name else 0.25

    def similarity(self, *args, **kwargs):
        return self.score


def _primary_fails_from(n: int):
    """Fail the primary model's embedding requests from its n-th on."""
    primary_calls: list[list[str]] = []

    def fails(provider: str, texts: list[str]) -> bool:
        if provider != "nebius":
            return False
        primary_calls.append(texts)
        return len(primary_calls) >= n

    return fails


async def _compare(inst: CyteOnto, calls, fails, **kwargs):
    decompose = AsyncMock(
        side_effect=lambda **kw: (
            [LabelDecomposition.single(label) for label in kw["labels"]],
            AgentUsage(),
        )
    )
    with (
        patch("cyteonto.cyteonto.describe_cells", _describe_mock()),
        patch("cyteonto.cyteonto.decompose_labels", decompose),
        patch("cyteonto.cyteonto.embed_texts", _embedder(calls, fails)),
        patch("cyteonto.cyteonto.OntologySimilarity", _FakeSimilarity),
    ):
        return await inst.compare(**kwargs)


class TestCompareAfterFailover:
    @pytest.mark.asyncio
    async def test_a_failover_during_compare_scores_with_the_fallback_model(
        self, tmp_path
    ):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        _save_ontology_embeddings(inst, inst.embedding)
        calls: list[tuple[str, int]] = []
        # The primary embeds the author labels, then fails for the algorithm's.
        df = await _compare(
            inst,
            calls,
            _primary_fails_from(2),
            author_labels=["T cell"],
            algorithms={"algo": ["T cell"]},
            run_id="run",
        )

        row = df.iloc[0]
        assert row["algorithm_ontology_id"] == "CL:0000001"
        # The pair is scored with the similarity of the fallback model's embeddings.
        assert row["cytescore_similarity"] == 0.75
        # The author labels are embedded again with the fallback model.
        assert calls == [
            ("nebius", 1),
            ("nebius", 1),
            ("openrouter", 1),
            ("openrouter", 2),
            ("openrouter", 1),
        ]
        assert inst.model_pair_usage.embeddingTier == "fallback"

    @pytest.mark.asyncio
    async def test_the_sides_before_a_failover_are_embedded_again(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        _save_ontology_embeddings(inst, inst.embedding)
        calls: list[tuple[str, int]] = []
        # The primary embeds the author labels and a1, then fails for a2.
        df = await _compare(
            inst,
            calls,
            _primary_fails_from(3),
            author_labels=["T cell"],
            algorithms={"a1": ["T cell"], "a2": ["B cell"]},
            run_id="run",
        )

        # Both algorithms are scored with one model, so they can be compared.
        assert df["cytescore_similarity"].tolist() == [0.75, 0.75]
        assert df["algorithm_ontology_id"].tolist() == ["CL:0000001", "CL:0000002"]
        assert calls == [
            ("nebius", 1),
            ("nebius", 1),
            ("nebius", 1),
            ("openrouter", 1),
            ("openrouter", 2),
            ("openrouter", 1),
            ("openrouter", 1),
        ]
        for kind, identifier in (("author", "author"), ("algorithm", "a1")):
            path = inst.paths.user_embeddings(
                "run", kind, identifier, inst.llm_key, _FALLBACK.to_artifact_key()
            )
            assert path.exists()

    @pytest.mark.asyncio
    async def test_a_failover_on_the_first_side_embeds_each_side_once(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        _save_ontology_embeddings(inst, inst.embedding)
        calls: list[tuple[str, int]] = []
        df = await _compare(
            inst,
            calls,
            _primary_fails_from(1),
            author_labels=["T cell"],
            algorithms={"algo": ["B cell"]},
            run_id="run",
        )

        assert df["cytescore_similarity"].tolist() == [0.75]
        assert calls == [
            ("nebius", 1),
            ("openrouter", 1),
            ("openrouter", 2),
            ("openrouter", 1),
        ]
        assert inst.model_pair_usage.embeddingTier == "fallback"

    @pytest.mark.asyncio
    async def test_an_earlier_call_with_the_primary_makes_the_tier_mixed(
        self, tmp_path
    ):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        _describe_terms(inst)
        _save_ontology_embeddings(inst, inst.embedding)
        calls: list[tuple[str, int]] = []
        # The first call embeds both sides with the primary; the second fails over
        # on its algorithm.
        fails = _primary_fails_from(4)
        args = {"author_labels": ["T cell"], "algorithms": {"algo": ["B cell"]}}
        await _compare(inst, calls, fails, run_id="r1", **args)
        assert inst.model_pair_usage.embeddingTier == "primary"
        await _compare(inst, calls, fails, run_id="r2", **args)

        assert inst.model_pair_usage.embeddingTier == "mixed"


class TestDecompositionCache:
    @pytest.mark.asyncio
    async def test_use_cache_false_keeps_the_other_labels_of_the_run(self, tmp_path):
        inst = _instance(tmp_path)
        decompose = AsyncMock(
            side_effect=[
                (
                    [LabelDecomposition.single("a"), LabelDecomposition.single("b")],
                    AgentUsage(),
                ),
                ([LabelDecomposition.single("b")], AgentUsage()),
            ]
        )
        with patch("cyteonto.cyteonto.decompose_labels", decompose):
            await inst._resolve_label_parts(["a", "b"], "run", use_cache=True)
            # A new compare() call.
            inst._call_decompositions = {}
            parts = await inst._resolve_label_parts(["b"], "run", use_cache=False)

        assert parts == {"b": ["b"]}
        assert decompose.await_args_list[1].kwargs["labels"] == ["b"]
        (cache,) = (inst.paths.user_dir / "decompositions" / "run").glob("*.json")
        assert json.loads(cache.read_text())["decompositions"] == {
            "a": ["a"],
            "b": ["b"],
        }

    @pytest.mark.asyncio
    async def test_failed_decompositions_are_retried_by_the_next_call(self, tmp_path):
        inst = _instance(tmp_path)
        decompose = AsyncMock(
            return_value=([LabelDecomposition.failed("x")], AgentUsage())
        )
        with patch("cyteonto.cyteonto.decompose_labels", decompose):
            first = await inst._resolve_label_parts(["x"], "run", use_cache=True)
            # compare() forgets the failures of the previous call.
            inst._call_decompositions = {}
            second = await inst._resolve_label_parts(["x"], "run", use_cache=True)

        assert first == second == {"x": ["x"]}
        assert decompose.await_count == 2
        assert not (inst.paths.user_dir / "decompositions" / "run").exists()

    @pytest.mark.asyncio
    async def test_a_split_over_the_usage_limits_is_not_retried(self, tmp_path):
        inst = _instance(tmp_path)
        agent = Mock()
        agent.name = "LabelDecompositionAgent"
        run_once = AsyncMock(side_effect=UsageLimitExceeded("request_limit of 50"))
        with (
            patch("cyteonto.describe._build_decompose_agent", return_value=agent),
            patch("cyteonto.describe._run_decompose_once", new=run_once),
        ):
            first = await inst._resolve_label_parts(["x"], "run", use_cache=True)
            # A new compare() call.
            inst._call_decompositions = {}
            second = await inst._resolve_label_parts(["x"], "run", use_cache=True)

        assert first == second == {"x": ["x"]}
        assert run_once.await_count == 1
        (cache,) = (inst.paths.user_dir / "decompositions" / "run").glob("*.json")
        assert json.loads(cache.read_text())["decompositions"] == {"x": ["x"]}

    @pytest.mark.asyncio
    async def test_a_failed_decomposition_is_not_retried_within_a_call(self, tmp_path):
        inst = _instance(tmp_path)
        decompose = AsyncMock(
            side_effect=[
                ([LabelDecomposition.failed("x")], AgentUsage()),
                ([LabelDecomposition.single("y")], AgentUsage()),
            ]
        )
        with patch("cyteonto.cyteonto.decompose_labels", decompose):
            await inst._resolve_label_parts(["x"], "run", use_cache=True)
            parts = await inst._resolve_label_parts(["x", "y"], "run", use_cache=True)

        # Every side of a call splits "x" the same way, so that equal labels match.
        assert parts == {"x": ["x"], "y": ["y"]}
        assert decompose.await_args_list[1].kwargs["labels"] == ["y"]

    @pytest.mark.asyncio
    async def test_without_the_cache_every_side_splits_a_label_the_same_way(
        self, tmp_path
    ):
        inst = _instance(tmp_path)
        split = LabelDecomposition(initialLabel="x", isCompound=True, parts=["a", "b"])
        decompose = AsyncMock(
            side_effect=[
                ([split], AgentUsage()),
                ([LabelDecomposition.single("y")], AgentUsage()),
            ]
        )
        with patch("cyteonto.cyteonto.decompose_labels", decompose):
            author = await inst._resolve_label_parts(["x"], "run", use_cache=False)
            algorithm = await inst._resolve_label_parts(
                ["x", "y"], "run", use_cache=False
            )

        assert author == {"x": ["a", "b"]}
        assert algorithm == {"x": ["a", "b"], "y": ["y"]}
        assert decompose.await_args_list[1].kwargs["labels"] == ["y"]

    @pytest.mark.asyncio
    async def test_without_the_cache_a_failed_split_keeps_the_cached_one(
        self, tmp_path
    ):
        inst = _instance(tmp_path)
        split = LabelDecomposition(initialLabel="x", isCompound=True, parts=["a", "b"])
        decompose = AsyncMock(
            side_effect=[
                ([split], AgentUsage()),
                ([LabelDecomposition.failed("x")], AgentUsage()),
            ]
        )
        with patch("cyteonto.cyteonto.decompose_labels", decompose):
            await inst._resolve_label_parts(["x"], "run", use_cache=True)
            # A new compare() call.
            inst._call_decompositions = {}
            parts = await inst._resolve_label_parts(["x"], "run", use_cache=False)

        assert parts == {"x": ["a", "b"]}
        assert decompose.await_count == 2


class TestDescriptionCache:
    @pytest.mark.asyncio
    async def test_use_cache_false_keeps_the_other_descriptions(self, tmp_path):
        inst = _instance(tmp_path)
        describe = AsyncMock(
            side_effect=lambda **kw: (
                [_description(label) for label in kw["labels"]],
                AgentUsage(),
            )
        )
        calls: list[tuple[str, int]] = []
        with (
            patch("cyteonto.cyteonto.describe_cells", describe),
            patch(
                "cyteonto.cyteonto.embed_texts",
                _embedder(calls, lambda provider, texts: False),
            ),
        ):
            await inst._embed_user_labels(
                ["a", "b"], "run", "author", "author", use_cache=True
            )
            await inst._embed_user_labels(
                ["a"], "run", "author", "author", use_cache=False
            )

        assert describe.await_args_list[1].kwargs["labels"] == ["a"]
        saved = storage.load_descriptions(
            inst.paths.user_descriptions("run", "author", "author", inst.llm_key)
        )
        assert saved is not None and set(saved) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_a_failed_new_description_keeps_the_cached_one(self, tmp_path):
        inst = _instance(tmp_path)
        results = [[_PLASMABLAST], [CellDescription.blank(label="plasmablast")]]
        describe = AsyncMock(side_effect=lambda **kw: (results.pop(0), AgentUsage()))
        calls: list[tuple[str, int]] = []
        args = (["plasmablast"], "run", "author", "author")
        with (
            patch("cyteonto.cyteonto.describe_cells", describe),
            patch("cyteonto.cyteonto.embed_texts", _embedder(calls, lambda *a: False)),
        ):
            first = await inst._embed_user_labels(*args, use_cache=True)
            again = await inst._embed_user_labels(*args, use_cache=False)
            cached = await inst._embed_user_labels(*args, use_cache=True)

        # The label is embedded with its cached description, which the file keeps,
        # and not with its text.
        np.testing.assert_array_equal(first, [[0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(again, first)
        np.testing.assert_array_equal(cached, first)
        assert calls == [("nebius", 1), ("nebius", 1)]

    @pytest.mark.asyncio
    async def test_new_descriptions_make_the_old_embeddings_stale(self, tmp_path):
        inst = _instance(tmp_path, fallback_embedding=_FALLBACK)
        results = [[CellDescription.blank(label="plasmablast")], [_PLASMABLAST]]
        describe = AsyncMock(side_effect=lambda **kw: (results.pop(0), AgentUsage()))
        calls: list[tuple[str, int]] = []
        down: set[str] = set()
        fake = _embedder(calls, lambda provider, texts: provider in down)
        args = (["plasmablast"], "run", "author", "author")
        path = inst.paths.user_embeddings(
            "run", "author", "author", inst.llm_key, inst.embd_key
        )
        with (
            patch("cyteonto.cyteonto.describe_cells", describe),
            patch("cyteonto.cyteonto.embed_texts", fake),
        ):
            # Without a description, the label is embedded with its text.
            stale = await inst._embed_user_labels(*args, use_cache=True)
            down.update({"nebius", "openrouter"})
            with pytest.raises(RuntimeError, match="Failed to embed labels"):
                await inst._embed_user_labels(*args, use_cache=True)
            # The file stays until a new one replaces it, but is not reused.
            assert path.exists()
            down.clear()
            fresh = await inst._embed_user_labels(*args, use_cache=True)

        np.testing.assert_array_equal(stale, [[1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(fresh, [[0.0, 1.0, 0.0]])

    @pytest.mark.asyncio
    async def test_another_models_file_made_before_new_descriptions_is_not_reused(
        self, tmp_path
    ):
        inst = _instance(tmp_path)
        other = _instance(tmp_path, embedding=_FALLBACK)
        results = [[_description("plasmablast")], [_PLASMABLAST]]
        describe = AsyncMock(side_effect=lambda **kw: (results.pop(0), AgentUsage()))
        calls: list[tuple[str, int]] = []
        args = (["plasmablast"], "run", "author", "author")
        with (
            patch("cyteonto.cyteonto.describe_cells", describe),
            patch("cyteonto.cyteonto.embed_texts", _embedder(calls, lambda *a: False)),
        ):
            await inst._embed_user_labels(*args, use_cache=True)
            old = await other._embed_user_labels(*args, use_cache=True)
            # The first instance describes the label again, which the files share.
            await inst._embed_user_labels(*args, use_cache=False)
            new = await other._embed_user_labels(*args, use_cache=True)

        np.testing.assert_array_equal(old, [[1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(new, [[0.0, 1.0, 0.0]])


class TestMatchWithoutLabels:
    def test_no_embeddings_match_nothing(self):
        inst = object.__new__(CyteOnto)
        inst._load_ontology_embeddings = Mock(side_effect=AssertionError("unused"))
        assert inst._match(np.zeros((0, 3), dtype=np.float32)) == []


class TestCompareAnndata:
    @pytest.mark.asyncio
    async def test_reads_the_columns_and_passes_the_options(self):
        import anndata as ad  # type: ignore
        import pandas as pd  # type: ignore

        inst = object.__new__(CyteOnto)
        inst.compare = AsyncMock(return_value="results")
        adata = ad.AnnData(
            obs=pd.DataFrame({"author": ["T cell"], "pred": ["B cell"]}, index=["c1"])
        )

        result = await inst.compare_anndata(
            author_labels=["T cell"],
            anndata_objects=[adata],
            target_columns=["pred"],
            author_column="author",
            run_id="run",
            min_match_similarity=0.5,
        )

        assert result == "results"
        kwargs = inst.compare.await_args.kwargs
        assert kwargs["author_labels"] == ["t cell"]
        assert kwargs["algorithms"] == [("pred", ["b cell"])]
        assert kwargs["min_match_similarity"] == 0.5

    @pytest.mark.asyncio
    async def test_several_objects_name_each_algorithm_by_its_object(self):
        import anndata as ad  # type: ignore
        import pandas as pd  # type: ignore

        inst = object.__new__(CyteOnto)
        inst.compare = AsyncMock(return_value="results")
        objects = [
            ad.AnnData(obs=pd.DataFrame({"pred": ["T cell"]}, index=["c1"])),
            ad.AnnData(
                obs=pd.DataFrame(
                    {"author": ["T cell"], "pred": ["B cell"]}, index=["c1"]
                )
            ),
            ad.AnnData(
                obs=pd.DataFrame(
                    {"author": ["T cell"], "pred": ["NK cell"]}, index=["c1"]
                )
            ),
        ]

        await inst.compare_anndata(
            author_labels=["T cell"],
            anndata_objects=objects,
            target_columns=["pred"],
            author_column="author",
            run_id="run",
        )

        # The first object has no author column and is skipped.
        assert inst.compare.await_args.kwargs["algorithms"] == [
            ("pred_1", ["b cell"]),
            ("pred_2", ["nk cell"]),
        ]

    @pytest.mark.asyncio
    async def test_several_objects_are_compared_with_the_author_labels(self, tmp_path):
        import anndata as ad  # type: ignore
        import pandas as pd  # type: ignore

        inst = _instance(tmp_path)
        _describe_terms(inst)
        _save_ontology_embeddings(inst, inst.embedding)
        index = ["c1", "c2"]
        objects = [
            ad.AnnData(
                obs=pd.DataFrame(
                    {"author": ["T cell", "B cell"], "pred": ["T cell", "B cell"]},
                    index=index,
                )
            ),
            ad.AnnData(
                obs=pd.DataFrame(
                    {"author": ["T cell", "B cell"], "pred": ["B cell", "B cell"]},
                    index=index,
                )
            ),
        ]
        calls: list[tuple[str, int]] = []
        decompose = AsyncMock(
            side_effect=lambda **kw: (
                [LabelDecomposition.single(label) for label in kw["labels"]],
                AgentUsage(),
            )
        )
        with (
            patch("cyteonto.cyteonto.describe_cells", _describe_mock()),
            patch("cyteonto.cyteonto.decompose_labels", decompose),
            patch("cyteonto.cyteonto.embed_texts", _embedder(calls, lambda *a: False)),
            patch("cyteonto.cyteonto.OntologySimilarity", _FakeSimilarity),
        ):
            df = await inst.compare_anndata(
                author_labels=["T cell", "B cell"],
                anndata_objects=objects,
                target_columns=["pred"],
                author_column="author",
                run_id="run",
            )

        assert df["algorithm"].tolist() == ["pred_0", "pred_0", "pred_1", "pred_1"]
        assert df["algorithm_ontology_id"].tolist() == [
            "CL:0000001",
            "CL:0000002",
            "CL:0000002",
            "CL:0000002",
        ]

    @pytest.mark.asyncio
    async def test_an_object_with_other_rows_fails_before_any_model_call(
        self, tmp_path
    ):
        import anndata as ad  # type: ignore
        import pandas as pd  # type: ignore

        inst = _instance(tmp_path)
        objects = [
            ad.AnnData(
                obs=pd.DataFrame(
                    {"author": ["T cell"], "pred": ["T cell"]}, index=["c1"]
                )
            ),
            ad.AnnData(
                obs=pd.DataFrame(
                    {"author": ["T cell", "B cell"], "pred": ["T cell", "B cell"]},
                    index=["c1", "c2"],
                )
            ),
        ]
        decompose = AsyncMock()
        calls: list[tuple[str, int]] = []
        with (
            patch("cyteonto.cyteonto.decompose_labels", decompose),
            patch("cyteonto.cyteonto.embed_texts", _embedder(calls, lambda *a: False)),
        ):
            with pytest.raises(ValueError, match="Label length mismatch for 'pred_1'"):
                await inst.compare_anndata(
                    author_labels=["T cell"],
                    anndata_objects=objects,
                    target_columns=["pred"],
                    author_column="author",
                    run_id="run",
                )

        assert decompose.await_count == 0
        assert calls == []
