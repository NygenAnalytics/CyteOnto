"""On-disk storage for embeddings (NPZ) and descriptions (JSON).

Two NPZ layouts are supported:

* Ontology layout (legacy-compatible): keys ``embeddings``, ``ontology_ids``, ``metadata``.
* User layout: keys ``embeddings``, ``labels``, ``metadata``.

Keeping ``labels`` inline with the user NPZ lets us detect when a user re-runs
``compare`` with a different label set without any sidecar metadata file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .logger import logger
from .models import CellDescription


def _default_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


def save_ontology_embeddings(
    path: Path,
    embeddings: np.ndarray,
    ontology_ids: list[str],
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = _default_meta(extra_metadata)
        meta.update(
            {
                "num_embeddings": int(len(embeddings)),
                "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            }
        )
        np.savez_compressed(
            path,
            embeddings=embeddings,
            ontology_ids=np.array(ontology_ids, dtype=object),
            metadata=np.array(meta, dtype=object),
        )
        logger.info(f"Saved {len(embeddings)} ontology embeddings to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save ontology embeddings to {path}: {e}")
        return False


def load_ontology_embeddings(
    path: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        ontology_ids = data["ontology_ids"].tolist()
        metadata = data["metadata"].item()
        logger.info(f"Loaded {len(embeddings)} ontology embeddings from {path}")
        return embeddings, ontology_ids, metadata
    except Exception as e:
        logger.error(f"Failed to load ontology embeddings from {path}: {e}")
        return None


def save_user_embeddings(
    path: Path,
    embeddings: np.ndarray,
    labels: list[str],
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = _default_meta(extra_metadata)
        meta.update(
            {
                "num_embeddings": int(len(embeddings)),
                "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            }
        )
        np.savez_compressed(
            path,
            embeddings=embeddings,
            labels=np.array(labels, dtype=object),
            metadata=np.array(meta, dtype=object),
        )
        logger.info(f"Saved {len(embeddings)} user embeddings to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save user embeddings to {path}: {e}")
        return False


def load_user_embeddings(
    path: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        if "labels" not in data.files:
            logger.debug(f"User NPZ at {path} has no 'labels' key; treating as stale")
            return None
        embeddings = data["embeddings"]
        labels = data["labels"].tolist()
        metadata = data["metadata"].item() if "metadata" in data.files else {}
        logger.info(f"Loaded {len(embeddings)} user embeddings from {path}")
        return embeddings, labels, metadata
    except Exception as e:
        logger.error(f"Failed to load user embeddings from {path}: {e}")
        return None


def save_descriptions(
    path: Path,
    descriptions: dict[str, CellDescription],
) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.model_dump() for k, v in descriptions.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(descriptions)} descriptions to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save descriptions to {path}: {e}")
        return False


def load_descriptions(path: Path) -> dict[str, CellDescription] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        descriptions = {k: CellDescription.model_validate(v) for k, v in raw.items()}
        logger.info(f"Loaded {len(descriptions)} descriptions from {path}")
        return descriptions
    except Exception as e:
        logger.error(f"Failed to load descriptions from {path}: {e}")
        return None
