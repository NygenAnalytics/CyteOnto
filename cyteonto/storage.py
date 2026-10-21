"""On-disk storage for embeddings (NPZ) and descriptions (JSON), schema v3."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .logger import logger
from .models import (
    CellDescription,
    DescriptionFileEnvelope,
    ModelArtifactKey,
)


def _artifact_key_dict(key: ModelArtifactKey) -> dict[str, str]:
    return {
        "provider": key.provider,
        "model": key.model,
        "company": key.company,
        "modelName": key.modelName,
    }


def _default_meta(
    llm_key: ModelArtifactKey,
    embd_key: ModelArtifactKey,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "schemaVersion": "3.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llm": _artifact_key_dict(llm_key),
        "embedding": _artifact_key_dict(embd_key),
    }
    if extra:
        meta.update(extra)
    return meta


def _savez_replacing(path: Path, **arrays: Any) -> None:
    """Write an NPZ beside ``path`` and then move it over ``path``.

    Readers of ``path`` see the old file or the new one, never a partial write.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def made_after_descriptions(metadata: dict[str, Any], desc_path: Path) -> bool:
    """Whether an embeddings file was made after its descriptions last changed.

    A file made before is stale, whichever embedding model made it, and is made
    again instead of reused. Without a descriptions file, nothing makes it stale.
    """
    try:
        with open(desc_path, "r", encoding="utf-8") as f:
            updated = json.load(f)["updatedAt"]
    except FileNotFoundError:
        return True
    except Exception:
        return False
    try:
        made = datetime.fromisoformat(metadata["timestamp"])
        return made >= datetime.fromisoformat(updated)
    except (KeyError, TypeError, ValueError):
        return False


def load_embeddings_metadata(path: Path) -> dict[str, Any] | None:
    """Return the metadata of an embeddings file without loading its arrays."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return data["metadata"].item()
    except Exception as e:
        logger.error(f"Failed to read the metadata of {path}: {e}")
        return None


def save_ontology_embeddings(
    path: Path,
    embeddings: np.ndarray,
    ontology_ids: list[str],
    llm_key: ModelArtifactKey,
    embd_key: ModelArtifactKey,
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = _default_meta(llm_key, embd_key, extra_metadata)
        meta.update(
            {
                "num_embeddings": int(len(embeddings)),
                "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            }
        )
        _savez_replacing(
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
    llm_key: ModelArtifactKey,
    embd_key: ModelArtifactKey,
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = _default_meta(llm_key, embd_key, extra_metadata)
        meta.update(
            {
                "num_embeddings": int(len(embeddings)),
                "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
            }
        )
        _savez_replacing(
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
    artifact_key: ModelArtifactKey,
) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = DescriptionFileEnvelope(
            artifactKey=artifact_key,
            updatedAt=datetime.now(timezone.utc).isoformat(),
            descriptions=descriptions,
        )
        payload = envelope.model_dump(mode="json")
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
        if not isinstance(raw, dict) or "schemaVersion" not in raw:
            logger.error(
                f"Descriptions at {path} are not schema v3 (missing envelope). "
                "Regenerate or remove the legacy file."
            )
            return None
        if raw.get("schemaVersion") != "3.0":
            logger.error(
                f"Descriptions at {path} have unsupported schemaVersion "
                f"{raw.get('schemaVersion')!r}; expected {'3.0'!r}."
            )
            return None
        envelope = DescriptionFileEnvelope.model_validate(raw)
        logger.info(f"Loaded {len(envelope.descriptions)} descriptions from {path}")
        return envelope.descriptions
    except Exception as e:
        logger.error(f"Failed to load descriptions from {path}: {e}")
        return None
