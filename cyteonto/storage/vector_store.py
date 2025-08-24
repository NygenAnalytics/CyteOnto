# cyteonto/storage/vector_store.py

import json
from pathlib import Path

import numpy as np

from ..logger_config import logger
from ..model import CellDescription


class VectorStore:
    """Manages storage and retrieval of embeddings in NPZ format."""

    def __init__(self):
        """Initialize vector store."""
        pass

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        ontology_ids: list[str],
        filepath: Path,
    ) -> bool:
        """
        Save embeddings to NPZ file with metadata.

        Args:
            embeddings: NumPy array of embeddings
            ontology_ids: list of corresponding ontology IDs
            filepath: Path to save NPZ file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save embeddings and metadata
            np.savez_compressed(
                filepath,
                embeddings=embeddings,
                ontology_ids=np.array(ontology_ids, dtype=object),
                metadata=np.array(
                    {
                        "num_embeddings": len(embeddings),
                        "embedding_dim": embeddings.shape[1]
                        if len(embeddings) > 0
                        else 0,
                        "version": "1.0",
                    },
                    dtype=object,
                ),
            )

            logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False

    def load_embeddings(
        self, filepath: Path
    ) -> tuple[np.ndarray, list[str], dict] | None:
        """
        Load embeddings from NPZ file.

        Args:
            filepath: Path to NPZ file

        Returns:
            Tuple of (embeddings, ontology_ids, metadata) or None if failed
        """
        try:
            if not filepath.exists():
                logger.warning(f"Embedding file not found: {filepath}")
                return None

            data = np.load(filepath, allow_pickle=True)

            embeddings = data["embeddings"]
            ontology_ids = data["ontology_ids"].tolist()
            metadata = data["metadata"].item()

            logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
            return embeddings, ontology_ids, metadata

        except Exception as e:
            logger.error(f"Failed to load embeddings from {filepath}: {e}")
            return None

    def save_descriptions(
        self, descriptions: dict[str, CellDescription], filepath: Path
    ) -> bool:
        """
        Save cell descriptions to JSON file.

        Args:
            descriptions: Dictionary mapping ontology IDs to descriptions
            filepath: Path to save JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(descriptions, f, indent=4, ensure_ascii=False)

            logger.info(f"Saved {len(descriptions)} descriptions to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save descriptions: {e}")
            return False

    def load_descriptions(self, filepath: Path) -> dict[str, CellDescription] | None:
        """
        Load cell descriptions from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary of descriptions or None if failed
        """
        try:
            if not filepath.exists():
                logger.warning(f"Descriptions file not found: {filepath}")
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                descriptions = json.load(f)

            logger.info(f"Loaded {len(descriptions)} descriptions from {filepath}")
            return descriptions

        except Exception as e:
            logger.error(f"Failed to load descriptions from {filepath}: {e}")
            return None

    def check_embedding_file_exists(self, filepath: Path) -> bool:
        """Check if embedding file exists and is valid."""
        if not filepath.exists():
            return False

        try:
            # Try to load and validate basic structure
            data = np.load(filepath, allow_pickle=True)
            required_keys = ["embeddings", "labels", "ontology_ids"]
            return all(key in data for key in required_keys)
        except Exception as e:
            logger.error(f"Invalid embedding file {filepath}: {e}")
            return False
