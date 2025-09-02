# cyteonto/setup.py

from pathlib import Path
from typing import Tuple, cast

from pydantic_ai import Agent

from .config import CONFIG
from .llm_config import EMBDModelConfig, EMBDProvider
from .logger_config import logger
from .ontology import OntologyExtractor
from .pipeline import DescriptionGenerator, EmbeddingGenerator
from .storage import FileManager, VectorStore


class CyteOntoSetup:
    """Handles setup phase for CyteOnto including file checking and generation."""

    def __init__(
        self,
        base_data_path: str,
        base_agent: Agent,
        embedding_model: str,
        embedding_provider: str,
    ) -> None:
        """

        Args:
            base_data_path: Base path for data files
            base_agent: Text generation model name (defaults to config)
            embedding_model: Embedding model name (defaults to config)
            embedding_provider: Embedding provider (defaults to config)
        """
        self.file_manager = FileManager(base_data_path)
        self.vector_store = VectorStore()

        # Model configuration
        self.base_agent = base_agent
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider

        # Initialize components
        self.ontology_extractor = OntologyExtractor(
            self.file_manager.get_ontology_mapping_path()
        )
        self.description_generator = DescriptionGenerator(self.vector_store)
        self.embedding_generator = EmbeddingGenerator(self.vector_store)

        logger.info(
            f"Setup initialized with base_agent='{self.base_agent.model.model_name}', embedding_model='{self.embedding_model}'"  # type: ignore
        )

    def validate_core_files(self) -> bool:
        """

        Returns:
            True if all core files exist, False otherwise
        """
        logger.info("Validating core data files...")
        validation_results = self.file_manager.validate_data_files()

        missing_files = [
            file_type for file_type, exists in validation_results.items() if not exists
        ]

        if missing_files:
            logger.error(f"Missing core files: {missing_files}")
            return False

        logger.info("All core files validated successfully")
        return True

    def check_embedding_file_exists(
        self, custom_path: Path | None = None
    ) -> Tuple[bool, Path]:
        """
        Check if embedding file exists.

        Args:
            custom_path: Custom path to embedding file

        Returns:
            Tuple of (exists, filepath)
        """
        if custom_path:
            filepath = custom_path
        else:
            filepath = self.file_manager.get_embedding_file_path(
                self.base_agent.model.model_name,  # type: ignore
                self.embedding_model,
            )

        exists = self.vector_store.check_embedding_file_exists(filepath)
        logger.info(
            f"Embedding file check: {filepath} - {'EXISTS' if exists else 'MISSING'}"
        )

        return exists, filepath

    def check_descriptions_file_exists(self) -> Tuple[bool, Path]:
        """
        Check if descriptions file exists.

        Returns:
            Tuple of (exists, filepath)
        """
        filepath = self.file_manager.get_descriptions_file_path(
            self.base_agent.model.model_name  # type: ignore
        )
        descriptions = self.vector_store.load_descriptions(filepath)
        exists = descriptions is not None and len(descriptions) > 0

        logger.info(
            f"Descriptions file check: {filepath} - {'EXISTS' if exists else 'MISSING'}"
        )
        return exists, filepath

    async def setup_embeddings(
        self,
        base_agent: Agent,
        generate_embeddings: bool = True,
        custom_embedding_path: Path | None = None,
        force_regenerate: bool = False,
    ) -> bool:
        """
        Setup embeddings with file checking and generation.

        Args:
            base_agent: Agent for generating descriptions
            generate_embeddings: Whether to generate embeddings if missing
            custom_embedding_path: Custom path for embedding file
            force_regenerate: Force regeneration even if files exist

        Returns:
            True if embeddings are ready, False otherwise
        """
        logger.info("Starting embedding setup process...")

        # Validate core files first
        if not self.validate_core_files():
            return False

        # Check embedding file
        embeddings_exist, embeddings_path = self.check_embedding_file_exists(
            custom_embedding_path
        )
        descriptions_exist, descriptions_path = self.check_descriptions_file_exists()

        if embeddings_exist and not force_regenerate:
            logger.info(
                "Embeddings already exist and force_regenerate=False, setup complete"
            )
            return True

        if not generate_embeddings:
            logger.warning("Embeddings missing but generate_embeddings=False")
            return False

        logger.info(
            "Embeddings missing or force_regenerate=True, starting generation process..."
        )

        # Load ontology mappings
        success = self.ontology_extractor.load_mapping()
        if not success:
            logger.error("Failed to load ontology mappings")
            return False

        # Get ontology terms and build mappings
        ontology_ids, ontology_terms = (
            self.ontology_extractor.get_ontology_terms_for_description_generation()
        )
        ontology_to_labels, label_to_ontology = self.ontology_extractor.build_mappings()

        logger.info(f"Found {len(ontology_terms)} ontology terms for processing")

        # Step 1: Generate descriptions if needed
        if not descriptions_exist or force_regenerate:
            logger.info("Generating descriptions for ontology terms...")
            descriptions = (
                await self.description_generator.generate_descriptions_for_terms(
                    base_agent=base_agent,
                    terms=ontology_terms,
                    descriptions_file_path=descriptions_path,
                    ontology_ids=ontology_ids,
                )
            )
        else:
            logger.info("Loading existing descriptions...")
            descriptions = self.vector_store.load_descriptions(descriptions_path)  # type: ignore
            if descriptions is None:
                descriptions = {}  # type: ignore

        if not descriptions:
            logger.error("Failed to generate or load descriptions")
            return False

        # Step 2: Generate embeddings
        logger.info("Generating embeddings for descriptions...")

        embd_model_config = EMBDModelConfig(
            provider=cast(EMBDProvider, self.embedding_provider),
            model=self.embedding_model,
            apiKey=CONFIG.EMBEDDING_MODEL_API_KEY,
        )

        embedding_result = await self.embedding_generator.generate_ontology_embeddings(
            descriptions=descriptions,
            embd_model_config=embd_model_config,
            embeddings_file_path=embeddings_path,
        )

        if embedding_result is None:
            logger.error("Failed to generate embeddings")
            return False

        embeddings, ontology_ids = embedding_result
        logger.info(f"Successfully generated and saved {len(embeddings)} embeddings")

        return True

    def get_setup_info(self) -> dict:
        """
        Get information about current setup state.

        Returns:
            Dictionary with setup information
        """
        embeddings_exist, embeddings_path = self.check_embedding_file_exists()
        descriptions_exist, descriptions_path = self.check_descriptions_file_exists()
        core_files = self.file_manager.validate_data_files()

        return {
            "base_agent": self.base_agent.model.model_name,  # type: ignore
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "embeddings_path": str(embeddings_path),
            "embeddings_exist": embeddings_exist,
            "descriptions_path": str(descriptions_path),
            "descriptions_exist": descriptions_exist,
            "core_files": core_files,
            "ready": embeddings_exist and all(core_files.values()),
        }
