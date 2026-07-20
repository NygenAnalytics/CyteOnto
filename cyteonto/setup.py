"""Download shipped and precomputed assets into the cyteonto data tree.

Run:
    uv run python cyteonto/setup.py
    uv run python cyteonto/setup.py --force
"""

import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore
import requests
from tqdm.auto import tqdm  # type: ignore

from .config import Config  # noqa: E402
from .logger import logger  # noqa: E402
from .models import EmbdConfig, LlmConfig, ModelArtifactKey  # noqa: E402
from .paths import PathConfig, artifact_key_segment  # noqa: E402

# Nygen public R2 bucket for CyteOnto v2
BASE_URL = "https://pub-d8bf3af01ebe421abded39c4cb33d88a.r2.dev/cyteonto_v2"

_cfg = Config()

PRIMARY_LLM = LlmConfig(
    provider=_cfg.PRIMARY_LLM_PROVIDER, model=_cfg.PRIMARY_LLM_MODEL
)
PRIMARY_EMBEDDING = EmbdConfig(
    provider=_cfg.PRIMARY_EMBEDDING_PROVIDER,  # type: ignore[arg-type]
    model=_cfg.PRIMARY_EMBEDDING_MODEL,
    modelSettings={},
)

BACKUP_LLM = LlmConfig(
    provider=_cfg.FALLBACK_LLM_PROVIDER, model=_cfg.FALLBACK_LLM_MODEL
)
BACKUP_EMBEDDING = EmbdConfig(
    provider=_cfg.FALLBACK_EMBEDDING_PROVIDER,  # type: ignore[arg-type]
    model=_cfg.FALLBACK_EMBEDDING_MODEL,
)

PRIMARY_LLM_KEY = PRIMARY_LLM.to_artifact_key()
PRIMARY_EMBD_KEY = PRIMARY_EMBEDDING.to_artifact_key()
BACKUP_LLM_KEY = BACKUP_LLM.to_artifact_key()
BACKUP_EMBD_KEY = BACKUP_EMBEDDING.to_artifact_key()

ONTOLOGY_CSV_URL: str = f"{BASE_URL}/cell_ontology/cell_to_cell_ontology.csv"
ONTOLOGY_ENRICHED_CSV_URL: str = (
    f"{BASE_URL}/cell_ontology/cell_to_cell_ontology_enriched.csv"
)
ONTOLOGY_OWL_URL: str = f"{BASE_URL}/cell_ontology/cl.owl"


def _descriptions_url(llm_key: ModelArtifactKey) -> str:
    return f"{BASE_URL}/descriptions/descriptions_{artifact_key_segment(llm_key)}.json"


def _embeddings_url(llm_key: ModelArtifactKey, embd_key: ModelArtifactKey) -> str:
    return (
        f"{BASE_URL}/embeddings/embeddings_{artifact_key_segment(llm_key)}_"
        f"{artifact_key_segment(embd_key)}.npz"
    )


def _primary_ontology_artifact_targets(
    paths: PathConfig,
) -> list[tuple[str, Path]]:
    return [
        (
            _descriptions_url(PRIMARY_LLM_KEY),
            paths.ontology_descriptions(PRIMARY_LLM_KEY),
        ),
        (
            _embeddings_url(PRIMARY_LLM_KEY, PRIMARY_EMBD_KEY),
            paths.ontology_embeddings(PRIMARY_LLM_KEY, PRIMARY_EMBD_KEY),
        ),
    ]


def _backup_ontology_artifact_targets(paths: PathConfig) -> list[tuple[str, Path]]:
    return [
        (
            _descriptions_url(BACKUP_LLM_KEY),
            paths.ontology_descriptions(BACKUP_LLM_KEY),
        ),
        (
            _embeddings_url(BACKUP_LLM_KEY, BACKUP_EMBD_KEY),
            paths.ontology_embeddings(BACKUP_LLM_KEY, BACKUP_EMBD_KEY),
        ),
    ]


def _download(url: str, destination: Path, *, force: bool) -> bool:
    """
    Stream ``url`` into ``destination``. Returns True on write, False on skip.
    """
    if destination.exists() and not force:
        logger.info(f"Already present, skipping: {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            tmp_path.open("wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))

    tmp_path.replace(destination)
    logger.info(f"Wrote {destination}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist on disk.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root data directory (defaults to cyteonto/data).",
    )
    args = parser.parse_args()

    paths = PathConfig(data_dir=args.data_dir)

    required_targets = [
        (ONTOLOGY_CSV_URL, paths.ontology_csv),
        (ONTOLOGY_OWL_URL, paths.ontology_owl),
        *_primary_ontology_artifact_targets(paths),
    ]
    optional_targets = _backup_ontology_artifact_targets(paths)

    required_failures: list[str] = []
    for url, dest in required_targets:
        try:
            _download(url, dest, force=args.force)
        except Exception as exc:
            logger.error(f"Failed to download {dest.name}: {exc}")
            required_failures.append(dest.name)

    if required_failures:
        logger.error(
            f"{len(required_failures)} required download(s) failed: {required_failures}"
        )
        return 1

    optional_failures: list[str] = []
    for url, dest in optional_targets:
        try:
            _download(url, dest, force=args.force)
        except Exception as exc:
            logger.warning(f"Optional backup download failed for {dest.name}: {exc}")
            optional_failures.append(dest.name)

    if optional_failures:
        logger.warning(
            f"{len(optional_failures)} optional backup download(s) failed: "
            f"{optional_failures}; continuing with primary model pair only"
        )

    enriched_path = paths.ontology_enriched_csv
    try:
        _download(ONTOLOGY_ENRICHED_CSV_URL, enriched_path, force=args.force)
    except Exception as exc:
        logger.warning(
            f"Enriched ontology CSV not available from CDN ({exc}); "
            "will build locally from the shipped original if needed"
        )

    csv_path = paths.ontology_csv
    if not enriched_path.exists() and csv_path.exists():
        logger.info("Building enriched ontology CSV locally from shipped original")
        df = pd.read_csv(csv_path)
        df["label_normalized"] = df["label"].astype(str).str.lower()
        dup_mask = df.duplicated(subset=["ontology_id", "label_normalized"], keep=False)
        if dup_mask.any():
            for (oid, norm), grp in df[dup_mask].groupby(
                ["ontology_id", "label_normalized"]
            ):
                originals = grp["label"].astype(str).tolist()
                logger.warning(
                    f"Normalized label collision for {oid} {norm!r}: "
                    f"original labels {originals}; keeping {originals[0]!r}"
                )
        before = len(df)
        df = df.drop_duplicates(
            subset=["ontology_id", "label_normalized"], keep="first"
        )
        dropped = before - len(df)
        if dropped:
            logger.info(
                f"Dropped {dropped} duplicate ontology rows "
                "(same ontology_id and label_normalized)"
            )
        enriched_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(enriched_path, index=False)
        logger.info(f"Wrote enriched ontology CSV: {enriched_path}")

    logger.info(f"Setup complete. Data tree is ready under {paths.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
