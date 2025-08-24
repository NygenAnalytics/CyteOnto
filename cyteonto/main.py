# cyteonto/main.py

from pathlib import Path

import anndata as ad  # type:ignore
import numpy as np  # type:ignore
import pandas as pd  # type:ignore
from pydantic_ai import Agent

from .config import CONFIG
from .llm_config import EMBDModelConfig
from .logger_config import logger
from .matcher import CyteOntoMatcher
from .pipeline import EmbeddingGenerator
from .storage import FileManager, VectorStore


class CyteOnto:
    """
    Main CyteOnto class for batch pairwise cell type comparisons.
    """

    def __init__(
        self,
        base_agent: Agent,
        embedding_model: str,
        embedding_provider: str,
        base_data_path: str | None = None,
        embeddings_file_path: Path | None = None,
    ) -> None:
        """
        Initialize CyteOnto for batch processing.

        Args:
            base_data_path: Base path for data files
            base_agent: Agent for text generation
            embedding_model: Embedding model name
            embedding_provider: Embedding provider
            embeddings_file_path: Custom path to embeddings file
        """
        self.file_manager = FileManager(base_data_path)
        self.vector_store = VectorStore()

        self.author_descriptions: list[str] | None = None
        self.author_embeddings: list[np.ndarray] | None = None

        self.base_agent = base_agent
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider

        # Determine embeddings file path
        if embeddings_file_path:
            self.embeddings_file_path = embeddings_file_path
        else:
            self.embeddings_file_path = self.file_manager.get_embedding_file_path(
                self.base_agent.model.model_name,  # type: ignore
                self.embedding_model,
            )

        # Initialize matcher
        self.matcher = CyteOntoMatcher(
            embeddings_file_path=self.embeddings_file_path,
            base_data_path=base_data_path,
        )

        # Initialize embedding generator for user queries
        self.embedding_generator = EmbeddingGenerator(self.vector_store)

        # Cached components
        self._embd_model_config: EMBDModelConfig | None = None

        logger.info(
            f"CyteOnto initialized with models: text='{self.base_agent.model.model_name}', embedding='{self.embedding_model}'"  # type: ignore
        )

    def _get_embd_model_config(self) -> EMBDModelConfig:
        """Get embedding model configuration."""
        if self._embd_model_config is None:
            from typing import cast

            from .llm_config import EMBDProvider

            self._embd_model_config = EMBDModelConfig(
                provider=cast(EMBDProvider, self.embedding_provider),
                model=self.embedding_model,
                apiKey=CONFIG.EMBEDDING_MODEL_API_KEY,
            )
        return self._embd_model_config

    async def compare_single_pair(
        self,
        author_labels: list[str],
        algorithm_labels: list[str],
    ) -> list[float]:
        """
        Compare a single pair of author vs algorithm labels.

        Args:
            author_labels: Author (reference) cell type labels from cell ontology
            algorithm_labels: Algorithm (predicted) cell type labels from cell ontology

        Returns:
            List of ontology similarities
        """
        logger.info(
            f"Comparing {len(author_labels)} author vs {len(algorithm_labels)} algorithm labels"
        )

        # Generate embeddings for algorithm labels
        embd_config = self._get_embd_model_config()

        if self.author_descriptions is None:
            # try loading the file
            author_descriptions_dict = self.vector_store.load_descriptions(
                Path("author_descriptions.json")
            )

            if author_descriptions_dict:
                self.author_descriptions = [
                    x.to_sentence() for x in author_descriptions_dict.values()
                ]

        author_embeddings_result = (
            await self.embedding_generator.generate_user_embeddings(
                author_labels,
                embd_config,
                base_agent=self.base_agent,
                descriptions=self.author_descriptions,
                filename_descriptions="author_descriptions.json",
                filename_embeddings="author_embeddings.npz",
            )
        )

        if author_embeddings_result:
            self.author_embeddings, _ = author_embeddings_result  # type: ignore

        algorithm_embeddings_result = (
            await self.embedding_generator.generate_user_embeddings(
                algorithm_labels,
                embd_config,
                base_agent=self.base_agent,
                filename_descriptions="algorithm_descriptions.json",
                filename_embeddings="algorithm_embeddings.npz",
            )
        )
        if algorithm_embeddings_result:
            algorithm_embeddings, _ = algorithm_embeddings_result  # type: ignore

        author_matches = self.matcher.match_embeddings_to_ontology(
            self.author_embeddings,  # type: ignore
            min_similarity=0.1,
        )
        algorithm_matches = self.matcher.match_embeddings_to_ontology(
            algorithm_embeddings,  # type: ignore
            min_similarity=0.1,
        )

        # Get ontology IDs for matched terms
        author_ontology_ids = []
        for match in author_matches:
            if match:
                author_ontology_ids.append(match["ontology_id"])
            else:
                author_ontology_ids.append(None)

        algorithm_ontology_ids = []
        for match in algorithm_matches:
            if match:
                algorithm_ontology_ids.append(match["ontology_id"])
            else:
                algorithm_ontology_ids.append(None)

        ontology_similarities = self.matcher.compute_ontology_similarity(
            author_ontology_ids, algorithm_ontology_ids
        )

        logger.info(f"Computed similarities: ontology={len(ontology_similarities)}")
        return ontology_similarities

    async def compare_batch(
        self,
        author_labels: list[str],
        algo_comparison_data: list[tuple[str, list[str]]],
    ) -> pd.DataFrame:
        """
        Perform batch comparisons between multiple algorithm results.

        Args:
            author_labels: Author (reference) cell type labels from cell ontology
            algo_comparison_data: List of (algorithm_name, algorithm_labels) tuples

        Returns:
            DataFrame with comparison results
        """
        logger.info(
            f"Starting batch comparison for {len(algo_comparison_data)} algorithms"
        )

        results = []

        for algorithm_name, algorithm_labels in algo_comparison_data:
            logger.info(f"Processing algorithm: {algorithm_name}")

            ontology_sims = await self.compare_single_pair(
                author_labels, algorithm_labels
            )

            # Create result rows
            for i, (author_label, algo_label) in enumerate(
                zip(author_labels, algorithm_labels)
            ):
                ontology_sim = ontology_sims[i] if i < len(ontology_sims) else 0.0

                results.append(
                    {
                        "algorithm": algorithm_name,
                        "author_label": author_label,
                        "algorithm_label": algo_label,
                        "ontology_similarity": ontology_sim,
                        "pair_index": i,
                    }
                )

        results_df = pd.DataFrame(results)
        logger.info(f"Batch comparison completed: {len(results_df)} comparisons")

        return results_df

    async def compare_anndata_objects(
        self,
        author_labels: list[str],
        anndata_objects: list[ad.AnnData],
        target_columns: list[str],
        author_column: str,
        algorithm_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare cell type annotations across AnnData objects.

        Args:
            anndata_objects: List of AnnData objects
            target_columns: List of column names in obs for algorithm results
            author_column: Column name for author (reference) labels
            algorithm_names: Optional names for algorithms (defaults to target_columns)

        Returns:
            DataFrame with comparison results
        """
        logger.info(
            f"Comparing {len(target_columns)} algorithms across {len(anndata_objects)} AnnData objects"
        )

        if algorithm_names is None:
            algorithm_names = target_columns.copy()

        if len(algorithm_names) != len(target_columns):
            raise ValueError("algorithm_names length must match target_columns length")

        all_comparison_data = []

        # Extract comparison data from AnnData objects
        for adata in anndata_objects:
            if author_column not in adata.obs:
                logger.warning(
                    f"Author column '{author_column}' not found in AnnData object"
                )
                continue

            for target_col, algo_name in zip(target_columns, algorithm_names):
                if target_col not in adata.obs:
                    logger.warning(
                        f"Target column '{target_col}' not found in AnnData object"
                    )
                    continue

                algorithm_labels = adata.obs[target_col].astype(str).tolist()
                all_comparison_data.append((algo_name, algorithm_labels))

        # Perform batch comparison
        return await self.compare_batch(author_labels, all_comparison_data)
