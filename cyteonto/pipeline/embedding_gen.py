# cyteonto/pipeline/embedding_gen.py

from pathlib import Path

import numpy as np
from pydantic_ai import Agent

from ..llm_config import EMBDModelConfig
from ..logger_config import logger
from ..model import CellDescription, to_sentence
from ..models.embeddings import generate_embeddings
from ..storage import CacheManager, VectorStore
from .description_gen import DescriptionGenerator


class EmbeddingGenerator:
    """Generates embeddings for cell descriptions."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize embedding generator.

        Args:
            vector_store: VectorStore instance for saving embeddings
        """
        self.vector_store = vector_store
        self.cache_manager = CacheManager(vector_store)

    async def generate_ontology_embeddings(
        self,
        descriptions: dict[str, CellDescription],
        embd_model_config: EMBDModelConfig,
        embeddings_file_path: Path,
    ) -> tuple[np.ndarray, list[str]] | None:
        """
        Generate embeddings for ontology term descriptions.

        Args:
            descriptions: Dictionary mapping ontology_id to description text
            embd_model_config: Embedding model configuration
            embeddings_file_path: Path to save embeddings

        Returns:
            Tuple of (embeddings, ontology_ids) or None if failed
        """
        logger.info(
            f"Starting embedding generation for {len(descriptions)} descriptions"
        )

        # Prepare data for embedding
        ontology_ids = list(descriptions.keys())
        texts_to_embed = [
            to_sentence(CellDescription.model_validate(descriptions[ontology_id]))
            for ontology_id in ontology_ids
        ]

        # Generate embeddings
        # No need to batch, semaphore will handle!
        all_embeddings: np.ndarray | None = await generate_embeddings(
            texts_to_embed, embd_model_config
        )
        if all_embeddings is None:
            logger.error("Failed to generate embeddings")
            return None
        logger.info(f"Generated {len(all_embeddings)} total embeddings")
        try:
            # Save embeddings to file
            success = self.vector_store.save_embeddings(
                embeddings=all_embeddings,
                ontology_ids=ontology_ids,
                filepath=embeddings_file_path,
            )

            if not success:
                logger.error("Failed to save embeddings to file")
                return None

            return all_embeddings, ontology_ids

        except Exception as e:
            logger.error(f"Error concatenating embeddings: {e}")
            return None

    async def generate_single_embedding(
        self, text: str, embd_model_config: EMBDModelConfig
    ) -> np.ndarray | None:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            embd_model_config: Embedding model configuration

        Returns:
            Embedding array or None if failed
        """
        try:
            embeddings = await generate_embeddings([text], embd_model_config)
            if embeddings is not None and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.error("Failed to generate single embedding")
                return None
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return None

    async def generate_user_embeddings(
        self,
        user_labels: list[str],
        embd_model_config: EMBDModelConfig,
        base_agent: Agent | None = None,
        descriptions: list[str] | None = None,
        identifier: str = "user_labels",
        embeddings_file_path: Path | None = None,
        descriptions_file_path: Path | None = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, list[str]] | None:
        """
        Generate embeddings for user-provided cell labels.
        Can optionally generate descriptions first using LLM.

        Args:
            user_labels: List of user cell type labels
            embd_model_config: Embedding model configuration
            base_agent: Optional agent for generating descriptions
            descriptions: Pre-computed descriptions to use (optional)
            identifier: Identifier for this set of labels (e.g., algorithm name)
            embeddings_file_path: Custom path for embeddings file
            descriptions_file_path: Custom path for descriptions file
            use_cache: Whether to use cached files if available

        Returns:
            Tuple of (embeddings, labels) or None if failed
        """
        logger.info(
            f"Generating embeddings for {len(user_labels)} user labels with identifier: {identifier}"
        )

        # Determine file paths - need FileManager access for this
        if embeddings_file_path is None or descriptions_file_path is None:
            # We need access to FileManager to generate proper paths
            # This will be handled by the calling code (CyteOnto class)
            logger.warning("File paths not provided, using current working directory")
            if embeddings_file_path is None:
                embeddings_file_path = Path(f"{identifier}_embeddings.npz")
            if descriptions_file_path is None:
                descriptions_file_path = Path(f"{identifier}_descriptions.json")

        # Check for cached files first if use_cache is enabled
        if use_cache:
            model_config = {
                "embedding_model": embd_model_config.model,
                "embedding_provider": embd_model_config.provider,
                "text_model": base_agent.model.model_name if base_agent else "none",  # type: ignore
            }

            cached_result = self.cache_manager.load_cached_embeddings(
                embeddings_file_path, user_labels, model_config
            )
            if cached_result is not None:
                cached_embeddings, cached_labels = cached_result
                logger.info(
                    f"Using validated cached embeddings from {embeddings_file_path}"
                )
                return cached_embeddings, cached_labels

        texts_to_embed = user_labels.copy()

        # If descriptions are provided, use them
        if descriptions is not None:
            logger.info(
                f"Using provided {len(descriptions)} descriptions for embedding"
            )
            texts_to_embed = descriptions

        elif base_agent:
            desc_gen = DescriptionGenerator(self.vector_store)

            results = await desc_gen.generate_descriptions_for_terms(
                base_agent, user_labels, descriptions_file_path
            )
            if results:
                texts_to_embed = [to_sentence(x) for x in results.values()]
            else:
                logger.error("Failed to generate descriptions")
                return None

        # Generate embeddings
        try:
            embeddings = await generate_embeddings(texts_to_embed, embd_model_config)
            if embeddings is not None:
                # Save embeddings with metadata using CacheManager
                model_config = {
                    "embedding_model": embd_model_config.model,
                    "embedding_provider": embd_model_config.provider,
                    "text_model": base_agent.model.model_name if base_agent else "none",  # type: ignore
                }

                success = self.cache_manager.save_embeddings_with_metadata(
                    embeddings=embeddings,
                    labels=user_labels,
                    cache_file_path=embeddings_file_path,
                    model_config=model_config,
                )

                if success:
                    logger.info(
                        f"Generated and saved {len(embeddings)} embeddings with metadata to {embeddings_file_path}"
                    )
                else:
                    logger.error("Failed to save embeddings with metadata")

                return embeddings, texts_to_embed
            else:
                logger.error("Failed to generate user embeddings")
                return None
        except Exception as e:
            logger.error(f"Error generating user embeddings: {e}")
            return None
