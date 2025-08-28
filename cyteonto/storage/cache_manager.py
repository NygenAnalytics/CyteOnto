# cyteonto/storage/cache_manager.py

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..logger_config import logger
from .vector_store import VectorStore


class CacheMetadata:
    """Metadata for cached files."""

    def __init__(
        self,
        labels_hash: str,
        model_config: Dict[str, str],
        creation_time: datetime,
        file_size: int,
        version: str = "1.0",
    ):
        self.labels_hash = labels_hash
        self.model_config = model_config
        self.creation_time = creation_time
        self.file_size = file_size
        self.version = version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels_hash": self.labels_hash,
            "model_config": self.model_config,
            "creation_time": self.creation_time.isoformat(),
            "file_size": self.file_size,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        return cls(
            labels_hash=data["labels_hash"],
            model_config=data["model_config"],
            creation_time=datetime.fromisoformat(data["creation_time"]),
            file_size=data["file_size"],
            version=data.get("version", "1.0"),
        )


class CacheManager:
    """Manages caching for embeddings and descriptions."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def compute_labels_hash(self, labels: List[str]) -> str:
        """Compute hash of labels for cache validation."""
        labels_str = "|".join(sorted(labels))
        return hashlib.sha256(labels_str.encode()).hexdigest()[:16]

    def get_cache_metadata_path(self, cache_file_path: Path) -> Path:
        """Get metadata file path for a cache file."""
        return cache_file_path.with_suffix(cache_file_path.suffix + ".meta")

    def save_cache_metadata(
        self, cache_file_path: Path, labels: List[str], model_config: Dict[str, str]
    ) -> bool:
        """Save metadata for a cached file."""
        try:
            metadata = CacheMetadata(
                labels_hash=self.compute_labels_hash(labels),
                model_config=model_config,
                creation_time=datetime.now(),
                file_size=cache_file_path.stat().st_size,
            )

            metadata_path = self.get_cache_metadata_path(cache_file_path)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            logger.debug(f"Saved cache metadata to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            return False

    def load_cache_metadata(self, cache_file_path: Path) -> Optional[CacheMetadata]:
        """Load metadata for a cached file."""
        try:
            metadata_path = self.get_cache_metadata_path(cache_file_path)
            if not metadata_path.exists():
                return None

            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return CacheMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            return None

    def validate_cache(
        self, cache_file_path: Path, labels: List[str], model_config: Dict[str, str]
    ) -> bool:
        """Validate if cached file is still valid for given labels and config."""
        if not cache_file_path.exists():
            return False

        metadata = self.load_cache_metadata(cache_file_path)
        if metadata is None:
            logger.warning(f"No metadata found for cache file {cache_file_path}")
            return False

        # Check labels hash
        current_hash = self.compute_labels_hash(labels)
        if metadata.labels_hash != current_hash:
            logger.debug("Cache invalid: labels hash mismatch")
            return False

        # Check model configuration
        if metadata.model_config != model_config:
            logger.debug("Cache invalid: model configuration changed")
            return False

        # Check file size consistency
        try:
            current_size = cache_file_path.stat().st_size
            if current_size != metadata.file_size:
                logger.warning("Cache file size changed, may be corrupted")
                return False
        except Exception:
            return False

        logger.debug(f"Cache valid for {cache_file_path}")
        return True

    def load_cached_embeddings(
        self, cache_file_path: Path, labels: List[str], model_config: Dict[str, str]
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Load cached embeddings with validation."""
        if not self.validate_cache(cache_file_path, labels, model_config):
            return None

        try:
            result = self.vector_store.load_embeddings(cache_file_path)
            if result is None:
                return None

            embeddings, ontology_ids, metadata = result
            logger.info(
                f"Loaded {len(embeddings)} cached embeddings from {cache_file_path}"
            )
            return embeddings, labels  # Return original labels for consistency
        except Exception as e:
            logger.error(f"Failed to load cached embeddings: {e}")
            return None

    def save_embeddings_with_metadata(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        cache_file_path: Path,
        model_config: Dict[str, str],
    ) -> bool:
        """Save embeddings with metadata."""
        try:
            # Save embeddings
            success = self.vector_store.save_embeddings(
                embeddings=embeddings,
                ontology_ids=[f"term:{idx}" for idx in range(len(embeddings))],
                filepath=cache_file_path,
            )

            if not success:
                return False

            # Save metadata
            return self.save_cache_metadata(cache_file_path, labels, model_config)
        except Exception as e:
            logger.error(f"Failed to save embeddings with metadata: {e}")
            return False

    def cleanup_invalid_caches(self, cache_directory: Path) -> int:
        """Remove cache files that are corrupted or have missing metadata."""
        cleaned_count = 0

        if not cache_directory.exists():
            return 0

        try:
            for cache_file in cache_directory.glob("*.npz"):
                metadata_file = self.get_cache_metadata_path(cache_file)

                # Check if metadata exists
                if not metadata_file.exists():
                    logger.info(f"Removing cache file without metadata: {cache_file}")
                    cache_file.unlink()
                    cleaned_count += 1
                    continue

                # Try to load metadata
                metadata = self.load_cache_metadata(cache_file)
                if metadata is None:
                    logger.info(
                        f"Removing cache file with invalid metadata: {cache_file}"
                    )
                    cache_file.unlink()
                    metadata_file.unlink()
                    cleaned_count += 1
                    continue

                # Check file integrity
                try:
                    result = self.vector_store.load_embeddings(cache_file)
                    if result is None:
                        logger.info(f"Removing corrupted cache file: {cache_file}")
                        cache_file.unlink()
                        metadata_file.unlink()
                        cleaned_count += 1
                except Exception:
                    logger.info(f"Removing unreadable cache file: {cache_file}")
                    cache_file.unlink()
                    if metadata_file.exists():
                        metadata_file.unlink()
                    cleaned_count += 1

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} invalid cache files")

        return cleaned_count

    def get_cache_stats(self, cache_directory: Path) -> Dict[str, Any]:
        """Get statistics about cached files."""
        stats: Dict[str, Any] = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "valid_files": 0,
            "invalid_files": 0,
            "oldest_cache": None,
            "newest_cache": None,
        }

        if not cache_directory.exists():
            return stats

        try:
            cache_times = []

            for cache_file in cache_directory.glob("*.npz"):
                stats["total_files"] += 1
                stats["total_size_mb"] += cache_file.stat().st_size / (1024 * 1024)

                metadata = self.load_cache_metadata(cache_file)
                if metadata is not None:
                    stats["valid_files"] += 1
                    cache_times.append(metadata.creation_time)
                else:
                    stats["invalid_files"] += 1

            if cache_times:
                stats["oldest_cache"] = min(cache_times).isoformat()
                stats["newest_cache"] = max(cache_times).isoformat()

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")

        return stats
