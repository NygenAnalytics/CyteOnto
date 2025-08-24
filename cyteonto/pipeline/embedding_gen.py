# cyteonto/pipeline/embedding_gen.py

from pathlib import Path

import numpy as np
from pydantic_ai import Agent

from ..llm_config import EMBDModelConfig
from ..logger_config import logger
from ..model import CellDescription
from ..models.embeddings import generate_embeddings
from ..storage import VectorStore
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
        texts_to_embed = [x.to_sentence() for x in descriptions.values()]
        ontology_ids = list(descriptions.keys())

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
        filename_descriptions: str = "author_descriptions.json",
        filename_embeddings: str = "author_embeddings.npz",
    ) -> tuple[np.ndarray, list[str]] | None:
        """
        Generate embeddings for user-provided cell labels.
        Can optionally generate descriptions first using LLM.

        Args:
            user_labels: List of user cell type labels
            embd_model_config: Embedding model configuration
            base_agent: Optional agent for generating descriptions

        Returns:
            Tuple of (embeddings, labels) or None if failed
        """
        logger.info(f"Generating embeddings for {len(user_labels)} user labels")

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
                base_agent, user_labels, Path(filename_descriptions)
            )
            if results:
                texts_to_embed = [x.to_sentence() for x in results.values()]
            else:
                logger.error("Failed to generate descriptions")
                return None

        # Generate embeddings
        try:
            embeddings = await generate_embeddings(texts_to_embed, embd_model_config)
            if embeddings is not None:
                # save embeddings to file
                self.vector_store.save_embeddings(
                    embeddings=embeddings,
                    ontology_ids=[f"term:{idx}" for idx in range(len(texts_to_embed))],
                    filepath=Path(filename_embeddings),
                )
                logger.info(f"Generated embeddings for {len(embeddings)} user labels")
                return embeddings, texts_to_embed
            else:
                logger.error("Failed to generate user embeddings")
                return None
        except Exception as e:
            logger.error(f"Error generating user embeddings: {e}")
            return None
