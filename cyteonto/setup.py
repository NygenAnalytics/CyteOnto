"""Download shipped and precomputed assets into the cyteonto data tree.

Run:
    uv run python cyteonto/setup.py
    uv run python cyteonto/setup.py --force
"""

import argparse
import sys
from pathlib import Path

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
ONTOLOGY_OWL_URL: str = f"{BASE_URL}/cell_ontology/cl.owl"


def _descriptions_url(llm_key: ModelArtifactKey) -> str:
    return f"{BASE_URL}/descriptions/descriptions_{artifact_key_segment(llm_key)}.json"


def _embeddings_url(llm_key: ModelArtifactKey, embd_key: ModelArtifactKey) -> str:
    return (
        f"{BASE_URL}/embeddings/embeddings_{artifact_key_segment(llm_key)}_"
        f"{artifact_key_segment(embd_key)}.npz"
    )


def _ontology_artifact_targets(paths: PathConfig) -> list[tuple[str, Path]]:
    """Primary and backup description + embedding files on the CDN."""
    pairs: list[tuple[ModelArtifactKey, ModelArtifactKey]] = [
        (PRIMARY_LLM_KEY, PRIMARY_EMBD_KEY),
        (BACKUP_LLM_KEY, BACKUP_EMBD_KEY),
    ]
    targets: list[tuple[str, Path]] = []
    for llm_key, embd_key in pairs:
        targets.append(
            (_descriptions_url(llm_key), paths.ontology_descriptions(llm_key))
        )
        targets.append(
            (
                _embeddings_url(llm_key, embd_key),
                paths.ontology_embeddings(llm_key, embd_key),
            )
        )
    return targets


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

    targets = [
        (ONTOLOGY_CSV_URL, paths.ontology_csv),
        (ONTOLOGY_OWL_URL, paths.ontology_owl),
        *_ontology_artifact_targets(paths),
    ]

    failures: list[str] = []
    for url, dest in targets:
        try:
            _download(url, dest, force=args.force)
        except Exception as exc:
            logger.error(f"Failed to download {dest.name}: {exc}")
            failures.append(dest.name)

    if failures:
        logger.error(f"{len(failures)} download(s) failed: {failures}")
        return 1

    logger.info(f"Setup complete. Data tree is ready under {paths.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
