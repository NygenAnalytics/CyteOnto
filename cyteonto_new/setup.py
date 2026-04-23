"""Download shipped and precomputed assets into the cyteonto_new data tree.

Run:
    uv run python cyteonto_new/setup.py
    uv run python cyteonto_new/setup.py --force
"""

import argparse
import sys
from pathlib import Path

import requests
from tqdm.auto import tqdm  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyteonto_new.logger import logger  # noqa: E402
from cyteonto_new.paths import PathConfig  # noqa: E402

BASE_URL = "https://pub-d8bf3af01ebe421abded39c4cb33d88a.r2.dev/cyteonto_v2/"

DEFAULT_TEXT_MODEL: str = "moonshotai/Kimi-K2.5"
DEFAULT_EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"

# Paste the public Cloudflare URLs below. Each value must point to the raw file.
ONTOLOGY_CSV_URL: str = f"{BASE_URL}/cell_ontology/cell_to_cell_ontology.csv"
ONTOLOGY_OWL_URL: str = f"{BASE_URL}/cell_ontology/cl.owl"
ONTOLOGY_DESCRIPTIONS_URL: str = (
    f"{BASE_URL}/descriptions/descriptions_moonshotai-Kimi-K2.5.json"
)
ONTOLOGY_EMBEDDINGS_URL: str = (
    f"{BASE_URL}/embeddings/embeddings_moonshotai-Kimi-K2.5_qwen-qwen3-embedding-8b.npz"
)


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
        help="Root data directory (defaults to cyteonto_new/data).",
    )
    args = parser.parse_args()

    paths = PathConfig(data_dir=args.data_dir)

    targets = [
        (ONTOLOGY_CSV_URL, paths.ontology_csv),
        (ONTOLOGY_OWL_URL, paths.ontology_owl),
        (
            ONTOLOGY_DESCRIPTIONS_URL,
            paths.ontology_descriptions(DEFAULT_TEXT_MODEL),
        ),
        (
            ONTOLOGY_EMBEDDINGS_URL,
            paths.ontology_embeddings(DEFAULT_TEXT_MODEL, DEFAULT_EMBEDDING_MODEL),
        ),
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
