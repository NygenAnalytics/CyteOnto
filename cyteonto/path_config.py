# cyteonto/path_config.py

from pathlib import Path


class PathConfig:
    """Centralized path configuration for all CyteOnto file operations."""

    def __init__(
        self, base_data_path: str | None = None, user_data_path: str | None = None
    ):
        """
        Initialize path configuration.

        Args:
            base_data_path: Base path for core data files (ontology files)
            user_data_path: Base path for user-generated files (embeddings, descriptions)
        """
        if base_data_path is None:
            # Default to data directory relative to this file
            self.base_data_path = Path(__file__).parent / "data"
        else:
            self.base_data_path = Path(base_data_path)

        if user_data_path is None:
            # Default user files to subdirectory of base_data_path
            self.user_data_path = self.base_data_path / "user_files"
        else:
            self.user_data_path = Path(user_data_path)

        # Ensure directories exist
        self.base_data_path.mkdir(parents=True, exist_ok=True)
        self.user_data_path.mkdir(parents=True, exist_ok=True)

    def get_ontology_embedding_path(
        self, text_model: str, embedding_model: str
    ) -> Path:
        """Get path for ontology embeddings following naming convention."""
        text_clean = self._clean_model_name(text_model)
        embd_clean = self._clean_model_name(embedding_model)
        filename = f"embeddings_{text_clean}_{embd_clean}.npz"
        path = self.base_data_path / "embedding" / "cell_ontology" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_ontology_descriptions_path(self, text_model: str) -> Path:
        """Get path for ontology descriptions."""
        text_clean = self._clean_model_name(text_model)
        filename = f"descriptions_{text_clean}.json"
        path = self.base_data_path / "embedding" / "descriptions" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_user_embeddings_path(
        self,
        identifier: str,
        text_model: str,
        embedding_model: str,
        category: str = "general",
        study_name: str | None = None,
    ) -> Path:
        """
        Get path for user-generated embeddings with algorithm-specific naming.

        Args:
            identifier: Algorithm name or identifier (e.g., 'CellTypist', 'author', 'user_labels')
            text_model: Text generation model name
            embedding_model: Embedding model name
            category: Category subfolder ('algorithms', 'author', 'general')
            study_name: Study name for organizing files (optional)
        """
        text_clean = self._clean_model_name(text_model)
        embd_clean = self._clean_model_name(embedding_model)
        identifier_clean = self._clean_identifier(identifier)
        filename = f"{identifier_clean}_embeddings_{text_clean}_{embd_clean}.npz"

        if study_name:
            study_clean = self._clean_identifier(study_name)
            path = (
                self.user_data_path / "embeddings" / study_clean / category / filename
            )
        else:
            path = self.user_data_path / "embeddings" / category / filename

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_user_descriptions_path(
        self,
        identifier: str,
        text_model: str,
        category: str = "general",
        study_name: str | None = None,
    ) -> Path:
        """
        Get path for user-generated descriptions with algorithm-specific naming.

        Args:
            identifier: Algorithm name or identifier
            text_model: Text generation model name
            category: Category subfolder ('algorithms', 'author', 'general')
            study_name: Study name for organizing files (optional)
        """
        text_clean = self._clean_model_name(text_model)
        identifier_clean = self._clean_identifier(identifier)
        filename = f"{identifier_clean}_descriptions_{text_clean}.json"

        if study_name:
            study_clean = self._clean_identifier(study_name)
            path = (
                self.user_data_path / "descriptions" / study_clean / category / filename
            )
        else:
            path = self.user_data_path / "descriptions" / category / filename

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_ontology_mapping_path(self) -> Path:
        """Get path to cell ontology mapping CSV."""
        return self.base_data_path / "cell_ontology" / "cell_to_cell_ontology.csv"

    def get_ontology_owl_path(self) -> Path:
        """Get path to cell ontology OWL file."""
        return self.base_data_path / "cell_ontology" / "cl.owl"

    def validate_core_files(self) -> dict[str, bool]:
        """Validate presence of core data files."""
        return {
            "ontology_mapping": self.get_ontology_mapping_path().exists(),
            "ontology_owl": self.get_ontology_owl_path().exists(),
        }

    def _clean_model_name(self, model_name: str) -> str:
        """Clean model names for safe filename usage."""
        return model_name.replace("/", "-").replace(":", "-").replace(" ", "_")

    def _clean_identifier(self, identifier: str) -> str:
        """Clean identifiers for safe filename usage."""
        return (
            identifier.replace("/", "-")
            .replace(":", "-")
            .replace(" ", "_")
            .replace(".", "_")
        )
