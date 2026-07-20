"""Tests for cyteonto.setup."""

import sys

import pandas as pd  # type: ignore

from cyteonto.paths import PathConfig
from cyteonto.setup import (
    BACKUP_EMBD_KEY,
    BACKUP_LLM_KEY,
    ONTOLOGY_ENRICHED_CSV_URL,
    PRIMARY_LLM_KEY,
    main,
)


def test_setup_builds_deduped_enriched_csv_when_not_on_cdn(temp_dir, monkeypatch):
    paths = PathConfig(str(temp_dir))
    csv_path = paths.ontology_csv
    enriched_path = paths.ontology_enriched_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    paths.ontology_owl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ontology_id": ["CL:1", "CL:1", "CL:2"],
            "label": ["Plasma cell", "PLASMA CELL", "NK cell"],
        }
    ).to_csv(csv_path, index=False)

    def fake_download(url: str, destination, *, force: bool) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination == csv_path:
            return False
        if url == ONTOLOGY_ENRICHED_CSV_URL:
            raise OSError("not on CDN yet")
        destination.touch()
        return True

    monkeypatch.setattr("cyteonto.setup._download", fake_download)
    monkeypatch.setattr(sys, "argv", ["setup.py", "--data-dir", str(temp_dir)])

    assert main() == 0

    original = pd.read_csv(csv_path)
    assert original["label"].tolist() == ["Plasma cell", "PLASMA CELL", "NK cell"]

    enriched = pd.read_csv(enriched_path)
    assert len(enriched) == 2
    assert enriched["label"].tolist() == ["Plasma cell", "NK cell"]
    assert enriched["label_normalized"].tolist() == ["plasma cell", "nk cell"]


def test_setup_continues_when_backup_downloads_fail(temp_dir, monkeypatch):
    paths = PathConfig(str(temp_dir))
    csv_path = paths.ontology_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    paths.ontology_owl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ontology_id": ["CL:1"], "label": ["T cell"]}).to_csv(
        csv_path, index=False
    )
    backup_desc = paths.ontology_descriptions(BACKUP_LLM_KEY)
    backup_emb = paths.ontology_embeddings(BACKUP_LLM_KEY, BACKUP_EMBD_KEY)
    primary_desc = paths.ontology_descriptions(PRIMARY_LLM_KEY)

    def fake_download(url: str, destination, *, force: bool) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination in (backup_desc, backup_emb):
            raise OSError("backup unavailable")
        if destination == csv_path:
            return False
        if url == ONTOLOGY_ENRICHED_CSV_URL:
            raise OSError("not on CDN yet")
        destination.touch()
        return True

    monkeypatch.setattr("cyteonto.setup._download", fake_download)
    monkeypatch.setattr(sys, "argv", ["setup.py", "--data-dir", str(temp_dir)])

    assert main() == 0
    assert primary_desc.exists()
    assert not backup_desc.exists()
    assert not backup_emb.exists()
