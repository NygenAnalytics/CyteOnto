# cyteonto/storage/file_utils.py

from pathlib import Path

from ..logger_config import logger


class FileManager:
    """Utility class for managing file presence and paths."""

    def __init__(self, base_data_path: str | None = None):
        """Initialize file manager with base data path."""
        if base_data_path is None:
            # Default to data directory relative to this file
            self.base_data_path = Path(__file__).parent.parent / "data"
        else:
            self.base_data_path = Path(base_data_path)

    def get_embedding_file_path(self, text_model: str, embedding_model: str) -> Path:
        """
        Generate embedding file path following naming convention:
        CO_<text-gen-model-name>_<embd-model-name>.npz

        Args:
            text_model: Name of text generation model
            embedding_model: Name of embedding model

        Returns:
            Path to embedding file
        """
        # Clean model names for filename
        text_clean = text_model.replace("/", "-").replace(":", "-")
        embd_clean = embedding_model.replace("/", "-").replace(":", "-")
        filename = f"CO_{text_clean}_{embd_clean}.npz"
        return self.base_data_path / "embedding" / "cell_ontology" / filename

    def get_descriptions_file_path(self, text_model: str) -> Path:
        """
        Generate descriptions file path for ontology terms.

        Args:
            text_model: Name of text generation model

        Returns:
            Path to descriptions JSON file
        """
        text_clean = text_model.replace("/", "-").replace(":", "-")
        filename = f"descriptions_{text_clean}.json"
        return self.base_data_path / "embedding" / "descriptions" / filename

    def check_file_exists(self, filepath: Path) -> bool:
        """Check if file exists."""
        return filepath.exists() and filepath.is_file()

    def ensure_directory_exists(self, filepath: Path) -> None:
        """Ensure directory exists for given filepath."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

    def get_ontology_mapping_path(self) -> Path:
        """Get path to cell ontology mapping CSV."""
        return self.base_data_path / "cell_ontology" / "cell_to_cell_ontology.csv"

    def get_ontology_owl_path(self) -> Path:
        """Get path to cell ontology OWL file."""
        return self.base_data_path / "cell_ontology" / "cl.owl"

    def validate_data_files(self) -> dict[str, bool]:
        """
        Validate presence of core data files.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "ontology_mapping": self.check_file_exists(
                self.get_ontology_mapping_path()
            ),
            "ontology_owl": self.check_file_exists(self.get_ontology_owl_path()),
        }

        for file_type, exists in validation_results.items():
            if not exists:
                logger.warning(f"Missing {file_type} file")
            else:
                logger.info(f"Found {file_type} file")

        return validation_results
