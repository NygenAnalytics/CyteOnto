"""Tests for cyteonto.ontology (OntologyMapping and OntologySimilarity)."""

import numpy as np
import pandas as pd  # type: ignore
import pytest

from cyteonto.ontology import OntologyMapping, OntologySimilarity


class TestOntologyMapping:
    def test_load_success(self, sample_ontology_csv_file):
        mapping = OntologyMapping(sample_ontology_csv_file)
        assert mapping.load() is True
        assert len(mapping.df) == 3

    def test_load_missing_file(self, temp_dir):
        mapping = OntologyMapping(temp_dir / "missing.csv")
        assert mapping.load() is False

    def test_label_to_id_and_back(self, sample_ontology_csv_file):
        mapping = OntologyMapping(sample_ontology_csv_file)
        mapping.load()
        assert mapping.label_to_id("T cell") == "CL:0000001"
        assert mapping.label_to_id("not a cell") is None
        assert mapping.labels_for_id("CL:0000002") == ["B cell"]
        assert mapping.labels_for_id("CL:9999999") == []

    def test_ids_and_joined_labels_groups_duplicates(self, temp_dir):
        data = {
            "ontology_id": ["CL:0000001", "CL:0000001", "CL:0000002"],
            "label": ["T cell", "T lymphocyte", "B cell"],
        }
        csv_path = temp_dir / "dup.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)

        mapping = OntologyMapping(csv_path)
        mapping.load()
        ids, joined = mapping.ids_and_joined_labels()
        joined_by_id = dict(zip(ids, joined))

        assert set(ids) == {"CL:0000001", "CL:0000002"}
        assert joined_by_id["CL:0000001"] == "T cell;T lymphocyte"
        assert joined_by_id["CL:0000002"] == "B cell"


class TestOntologySimilarityHelpers:
    def test_cosine_orthogonal(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert OntologySimilarity._cosine(v1, v2) == 0.0

    def test_cosine_identical(self):
        v = np.array([0.5, 0.5])
        assert OntologySimilarity._cosine(v, v) == pytest.approx(1.0)

    def test_cosine_zero_vector(self):
        assert OntologySimilarity._cosine(np.zeros(2), np.array([1.0, 1.0])) == 0.0

    def test_gaussian_hill_at_center(self):
        assert OntologySimilarity._gaussian_hill(1.0, center=1.0) == 1.0

    def test_gaussian_hill_custom_params(self):
        # center=0, width=1 -> exp(0) == 1.0 at x==0
        assert OntologySimilarity._gaussian_hill(0.0, center=0.0, width=1.0) == 1.0

    def test_simple_normalizes_separators(self):
        assert OntologySimilarity._simple("T_cell", "T-cell") == 1.0


class TestOntologySimilarityMetrics:
    def _make(self):
        sim = OntologySimilarity(owl_path=None)
        sim.embedding_map = {
            "CL:0000001": np.array([1.0, 0.0]),
            "CL:0000002": np.array([0.0, 1.0]),
            "CL:0000003": np.array([0.707, 0.707]),
        }
        return sim

    def test_identical_ids_short_circuit(self):
        sim = self._make()
        assert sim.similarity("CL:0000001", "CL:0000001") == 1.0

    def test_non_string_inputs(self):
        sim = self._make()
        assert sim.similarity(None, "CL:0000001") == 0.0  # type: ignore[arg-type]

    def test_cosine_direct(self):
        sim = self._make()
        assert sim.similarity("CL:0000001", "CL:0000002", metric="cosine_direct") == 0.0

    def test_cosine_kernel_default_params_low_for_orthogonal(self):
        sim = self._make()
        score = sim.similarity("CL:0000001", "CL:0000002", metric="cosine_kernel")
        assert 0.0 <= score < 0.01

    def test_cosine_kernel_custom_params(self):
        sim = self._make()
        score = sim.similarity(
            "CL:0000001",
            "CL:0000002",
            metric="cosine_kernel",
            metric_params={"center": 0.0, "width": 1.0},
        )
        assert score == 1.0

    def test_cosine_kernel_width_increases_partial_match(self):
        sim = self._make()
        default = sim.similarity("CL:0000001", "CL:0000003", metric="cosine_kernel")
        wider = sim.similarity(
            "CL:0000001",
            "CL:0000003",
            metric="cosine_kernel",
            metric_params={"width": 1.0},
        )
        assert wider > default

    def test_cosine_missing_embedding_returns_zero(self):
        sim = self._make()
        assert sim.similarity("CL:0000001", "CL:9999999", metric="cosine_direct") == 0.0

    def test_simple_metric(self):
        sim = self._make()
        score = sim.similarity("T helper cell", "T cell", metric="simple")
        assert 0.0 < score < 1.0

    def test_unknown_metric_falls_back_to_simple(self):
        sim = self._make()
        # No ontology loaded, so non-embedding metrics fall back to simple.
        score = sim.similarity("T cell", "B cell", metric="path")
        assert 0.0 <= score <= 1.0
