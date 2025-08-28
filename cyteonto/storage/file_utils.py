# cyteonto/storage/file_utils.py

from pathlib import Path
from typing import Optional

from ..logger_config import logger
from ..path_config import PathConfig


class FileManager:
    """Utility class for managing file presence and paths."""

    def __init__(
        self, base_data_path: Optional[str] = None, user_data_path: Optional[str] = None
    ):
        """Initialize file manager with path configuration."""
        self.path_config = PathConfig(base_data_path, user_data_path)

    def get_embedding_file_path(self, text_model: str, embedding_model: str) -> Path:
        """
        Generate embedding file path for ontology embeddings.

        Args:
            text_model: Name of text generation model
            embedding_model: Name of embedding model

        Returns:
            Path to embedding file
        """
        return self.path_config.get_ontology_embedding_path(text_model, embedding_model)

    def get_descriptions_file_path(self, text_model: str) -> Path:
        """
        Generate descriptions file path for ontology terms.

        Args:
            text_model: Name of text generation model

        Returns:
            Path to descriptions JSON file
        """
        return self.path_config.get_ontology_descriptions_path(text_model)

    def get_user_embeddings_path(
        self,
        identifier: str,
        text_model: str,
        embedding_model: str,
        category: str = "general",
        study_name: str | None = None,
    ) -> Path:
        """
        Generate user embeddings file path with algorithm-specific naming.

        Args:
            identifier: Algorithm name or identifier
            text_model: Text generation model name
            embedding_model: Embedding model name
            category: Category subfolder ('algorithms', 'author', 'general')
            study_name: Study name for organizing files (optional)
        """
        return self.path_config.get_user_embeddings_path(
            identifier, text_model, embedding_model, category, study_name
        )

    def get_user_descriptions_path(
        self,
        identifier: str,
        text_model: str,
        category: str = "general",
        study_name: str | None = None,
    ) -> Path:
        """
        Generate user descriptions file path with algorithm-specific naming.

        Args:
            identifier: Algorithm name or identifier
            text_model: Text generation model name
            category: Category subfolder ('algorithms', 'author', 'general')
            study_name: Study name for organizing files (optional)
        """
        return self.path_config.get_user_descriptions_path(
            identifier, text_model, category, study_name
        )

    def check_file_exists(self, filepath: Path) -> bool:
        """Check if file exists."""
        return filepath.exists() and filepath.is_file()

    def ensure_directory_exists(self, filepath: Path) -> None:
        """Ensure directory exists for given filepath."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

    def get_ontology_mapping_path(self) -> Path:
        """Get path to cell ontology mapping CSV."""
        return self.path_config.get_ontology_mapping_path()

    def get_ontology_owl_path(self) -> Path:
        """Get path to cell ontology OWL file."""
        return self.path_config.get_ontology_owl_path()

    def validate_data_files(self) -> dict[str, bool]:
        """
        Validate presence of core data files.

        Returns:
            Dictionary with validation results
        """
        validation_results = self.path_config.validate_core_files()

        for file_type, exists in validation_results.items():
            if not exists:
                logger.warning(f"Missing {file_type} file")
            else:
                logger.info(f"Found {file_type} file")

        return validation_results
