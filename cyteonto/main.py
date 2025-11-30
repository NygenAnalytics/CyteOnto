# cyteonto/main.py

from pathlib import Path
from typing import Any

import anndata as ad  # type:ignore
import numpy as np  # type:ignore
import pandas as pd  # type:ignore
from pydantic_ai import Agent
from tqdm.auto import tqdm  # type:ignore

from .config import CONFIG
from .llm_config import EMBDModelConfig
from .logger_config import logger
from .matcher import CyteOntoMatcher
from .model import to_sentence
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
        user_data_path: str | None = None,
        embeddings_file_path: Path | None = None,
        enable_user_file_caching: bool = True,
    ) -> None:
        """
        Initialize CyteOnto for batch processing.

        Args:
            base_data_path: Base path for core data files (ontology files)
            user_data_path: Base path for user-generated files
            base_agent: Agent for text generation
            embedding_model: Embedding model name
            embedding_provider: Embedding provider
            embeddings_file_path: Custom path to ontology embeddings file
            enable_user_file_caching: Enable caching for user-generated files
        """
        self.file_manager = FileManager(base_data_path, user_data_path)
        self.vector_store = VectorStore()

        self.author_descriptions: list[str] | None = None
        self.author_embeddings: list[np.ndarray] | None = None

        self.base_agent = base_agent
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.enable_user_file_caching = enable_user_file_caching

        # Determine ontology embeddings file path
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
            base_agent=self.base_agent,
            embedding_model=self.embedding_model,
        )

        # Initialize embedding generator for user queries
        self.embedding_generator = EmbeddingGenerator(self.vector_store)

        # Cached components
        self._embd_model_config: EMBDModelConfig | None = None

        logger.info(
            f"CyteOnto initialized with models: text='{self.base_agent.model.model_name}', embedding='{self.embedding_model}'"  # type: ignore
        )

    @classmethod
    async def with_setup(
        cls,
        base_agent: Agent,
        embedding_model: str,
        embedding_provider: str,
        base_data_path: str | None = None,
        user_data_path: str | None = None,
        embeddings_file_path: Path | None = None,
        enable_user_file_caching: bool = True,
        force_regenerate: bool = False,
    ) -> "CyteOnto":
        """
        Create CyteOnto instance with automatic setup.

        Args:
            base_agent: Agent for text generation
            embedding_model: Embedding model name
            embedding_provider: Embedding provider
            base_data_path: Base path for core data files
            user_data_path: Base path for user-generated files
            embeddings_file_path: Custom path to ontology embeddings file
            enable_user_file_caching: Enable caching for user-generated files
            force_regenerate: Force regeneration of embeddings even if they exist

        Returns:
            Initialized CyteOnto instance with setup completed
        """
        from .setup import CyteOntoSetup

        # Initialize CyteOnto instance
        cyto = cls(
            base_agent=base_agent,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            base_data_path=base_data_path,
            user_data_path=user_data_path,
            embeddings_file_path=embeddings_file_path,
            enable_user_file_caching=enable_user_file_caching,
        )

        # Run setup
        setup = CyteOntoSetup(
            base_data_path=base_data_path
            or str(cyto.file_manager.path_config.base_data_path),
            base_agent=base_agent,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
        )

        success = await setup.setup_embeddings(
            base_agent=base_agent,
            generate_embeddings=True,
            custom_embedding_path=embeddings_file_path,
            force_regenerate=force_regenerate,
        )

        if not success:
            logger.error("Setup failed during CyteOnto.with_setup()")
            raise RuntimeError("Failed to complete ontology embeddings setup")

        logger.info("CyteOnto.with_setup() completed successfully")
        return cyto

    def cleanup_user_cache(self) -> int:
        """
        Clean up invalid user cache files.

        Returns:
            Number of files cleaned up
        """
        from .storage import CacheManager

        cache_manager = CacheManager(self.vector_store)

        # Clean up user file caches
        user_embeddings_dir = (
            self.file_manager.path_config.user_data_path / "embeddings"
        )
        cleaned = 0

        for category_dir in user_embeddings_dir.glob("*"):
            if category_dir.is_dir():
                cleaned += cache_manager.cleanup_invalid_caches(category_dir)

        return cleaned

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get statistics about user cache files.

        Returns:
            Dictionary with cache statistics
        """
        from .storage import CacheManager

        cache_manager = CacheManager(self.vector_store)
        user_embeddings_dir = (
            self.file_manager.path_config.user_data_path / "embeddings"
        )

        total_stats: dict[str, Any] = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "valid_files": 0,
            "invalid_files": 0,
            "categories": dict[str, Any](),
        }

        for category_dir in user_embeddings_dir.glob("*"):
            if category_dir.is_dir():
                category_stats = cache_manager.get_cache_stats(category_dir)
                total_stats["categories"][category_dir.name] = category_stats
                total_stats["total_files"] += category_stats["total_files"]
                total_stats["total_size_mb"] += category_stats["total_size_mb"]
                total_stats["valid_files"] += category_stats["valid_files"]
                total_stats["invalid_files"] += category_stats["invalid_files"]

        return total_stats

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
        algorithm_name: str = "algorithm",
        study_name: str | None = None,
        metric: str = "cosine_kernel",
        metric_params: dict | None = None,
    ) -> list[dict]:
        """
        Compare a single pair of author vs algorithm labels with detailed results.

        Args:
            author_labels: Author (reference) cell type labels from cell ontology
            algorithm_labels: Algorithm (predicted) cell type labels from cell ontology
            algorithm_name: Name of the algorithm (for file naming and caching)
            study_name: Name of the study for organizing files (optional)
            metric: Metric to use for similarity calculation (e.g., "cosine_kernel")
            metric_params: Additional parameters for the similarity metric

        Returns:
            List of detailed comparison dictionaries
        """
        study_info = f" (study: {study_name})" if study_name else ""
        logger.info(
            f"Detailed comparison: {len(author_labels)} author vs {len(algorithm_labels)} algorithm labels for '{algorithm_name}'{study_info}"
        )

        # Generate proper file paths
        embd_config = self._get_embd_model_config()
        text_model_name = self.base_agent.model.model_name  # type: ignore

        # Generate author embeddings with proper paths
        author_embeddings_path = self.file_manager.get_user_embeddings_path(
            "author", text_model_name, self.embedding_model, "author", study_name
        )
        author_descriptions_path = self.file_manager.get_user_descriptions_path(
            "author", text_model_name, "author", study_name
        )

        # Load cached descriptions if available
        if self.author_descriptions is None:
            author_descriptions_dict = self.vector_store.load_descriptions(
                author_descriptions_path
            )
            if author_descriptions_dict:
                self.author_descriptions = [
                    to_sentence(x) for x in author_descriptions_dict.values()
                ]

        author_embeddings_result = (
            await self.embedding_generator.generate_user_embeddings(
                author_labels,
                embd_config,
                base_agent=self.base_agent,
                descriptions=self.author_descriptions,
                identifier="author",
                embeddings_file_path=author_embeddings_path,
                descriptions_file_path=author_descriptions_path,
                use_cache=self.enable_user_file_caching,
            )
        )

        if author_embeddings_result:
            self.author_embeddings, _ = author_embeddings_result  # type: ignore

        # Generate algorithm embeddings with algorithm-specific paths
        algorithm_embeddings_path = self.file_manager.get_user_embeddings_path(
            algorithm_name,
            text_model_name,
            self.embedding_model,
            "algorithms",
            study_name,
        )
        algorithm_descriptions_path = self.file_manager.get_user_descriptions_path(
            algorithm_name, text_model_name, "algorithms", study_name
        )

        algorithm_embeddings_result = (
            await self.embedding_generator.generate_user_embeddings(
                algorithm_labels,
                embd_config,
                base_agent=self.base_agent,
                identifier=algorithm_name,
                embeddings_file_path=algorithm_embeddings_path,
                descriptions_file_path=algorithm_descriptions_path,
                use_cache=self.enable_user_file_caching,
            )
        )
        if algorithm_embeddings_result:
            algorithm_embeddings, _ = algorithm_embeddings_result  # type: ignore

        # Get detailed matches with similarity scores
        author_matches = self.matcher.match_embeddings_to_ontology(
            self.author_embeddings,  # type: ignore
            min_similarity=0.1,
        )
        algorithm_matches = self.matcher.match_embeddings_to_ontology(
            algorithm_embeddings,  # type: ignore
            min_similarity=0.1,
        )

        # Prepare detailed results
        detailed_results = []

        for i, (author_label, algorithm_label) in enumerate(
            zip(author_labels, algorithm_labels)
        ):
            # Get author match details
            author_match = author_matches[i] if i < len(author_matches) else None
            author_ontology_id = author_match["ontology_id"] if author_match else None
            author_embedding_similarity = (
                author_match["similarity"] if author_match else 0.0
            )

            # Get algorithm match details
            algorithm_match = (
                algorithm_matches[i] if i < len(algorithm_matches) else None
            )
            algorithm_ontology_id = (
                algorithm_match["ontology_id"] if algorithm_match else None
            )
            algorithm_embedding_similarity = (
                algorithm_match["similarity"] if algorithm_match else 0.0
            )

            # Compute ontology hierarchy similarity
            ontology_hierarchy_similarity = 0.0
            similarity_method = "no_matches"

            if author_ontology_id and algorithm_ontology_id:
                ontology_similarities = self.matcher.compute_ontology_similarity(
                    [author_ontology_id],
                    [algorithm_ontology_id],
                    [author_embedding_similarity],
                    [algorithm_embedding_similarity],
                    metric=metric,
                    metric_params=metric_params,
                )
                ontology_hierarchy_similarity = (
                    ontology_similarities[0] if ontology_similarities else 0.0
                )
                similarity_method = (
                    "ontology_hierarchy"
                    if author_ontology_id.startswith("CL:")
                    and algorithm_ontology_id.startswith("CL:")
                    else "string_similarity"
                )
            elif not author_ontology_id and not algorithm_ontology_id:
                similarity_method = "no_matches"
            else:
                similarity_method = "partial_match"

            detailed_results.append(
                {
                    "author_label": author_label,
                    "algorithm_label": algorithm_label,
                    "author_ontology_id": author_ontology_id,
                    "author_embedding_similarity": round(
                        author_embedding_similarity, 4
                    ),
                    "algorithm_ontology_id": algorithm_ontology_id,
                    "algorithm_embedding_similarity": round(
                        algorithm_embedding_similarity, 4
                    ),
                    "ontology_hierarchy_similarity": round(
                        ontology_hierarchy_similarity, 4
                    ),
                    "similarity_method": similarity_method,
                    "study_name": study_name,
                    "pair_index": i,
                }
            )

        logger.info(f"Generated detailed results for {len(detailed_results)} pairs")
        return detailed_results

    async def compare_batch(
        self,
        author_labels: list[str],
        algo_comparison_data: list[tuple[str, list[str]]],
        study_name: str | None = None,
        metric: str = "cosine_kernel",
        metric_params: dict | None = None,
    ) -> pd.DataFrame:
        """
        Perform detailed batch comparisons between multiple algorithm results.

        Args:
            author_labels: Author (reference) cell type labels from cell ontology
            algo_comparison_data: List of (algorithm_name, algorithm_labels) tuples
            study_name: Name of the study for organizing files (optional)
            metric: Metric to use for similarity calculation (e.g., "cosine_kernel")
            metric_params: Additional parameters for the similarity metric

        Returns:
            DataFrame with detailed comparison results including:
            - algorithm: Algorithm name
            - author_label: Original author label
            - algorithm_label: Original algorithm label
            - author_ontology_id: Matched ontology ID for author label
            - author_embedding_similarity: Similarity between author label and its ontology match
            - algorithm_ontology_id: Matched ontology ID for algorithm label
            - algorithm_embedding_similarity: Similarity between algorithm label and its ontology match
            - ontology_hierarchy_similarity: Final similarity between the two ontology terms
            - similarity_method: Method used for final similarity computation
            - study_name: Name of the study (if provided)
            - pair_index: Index of the pair
        """
        logger.info(
            f"Starting detailed batch comparison for {len(algo_comparison_data)} algorithms"
        )

        all_results = []

        for algorithm_name, algorithm_labels in tqdm(
            algo_comparison_data,
            total=len(algo_comparison_data),
            desc="Comparing Algorithms",
        ):
            logger.info(f"Processing algorithm: {algorithm_name}")

            detailed_results = await self.compare_single_pair(
                author_labels,
                algorithm_labels,
                algorithm_name,
                study_name,
                metric,
                metric_params,
            )

            # Add algorithm name to each result
            for result in detailed_results:
                result["algorithm"] = algorithm_name
                all_results.append(result)

        # Create DataFrame with proper column ordering
        results_df = pd.DataFrame(all_results)

        # Reorder columns for better readability
        column_order = [
            "study_name",
            "algorithm",
            "pair_index",
            "author_label",
            "algorithm_label",
            "author_ontology_id",
            "author_embedding_similarity",
            "algorithm_ontology_id",
            "algorithm_embedding_similarity",
            "ontology_hierarchy_similarity",
            "similarity_method",
        ]

        # Only include columns that exist (in case of future changes)
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]

        logger.info(
            f"Detailed batch comparison completed: {len(results_df)} comparisons"
        )
        logger.info(
            f"Similarity methods used: {results_df['similarity_method'].value_counts().to_dict()}"
        )

        return results_df

    async def compare_anndata_objects(
        self,
        author_labels: list[str],
        anndata_objects: list[ad.AnnData],
        target_columns: list[str],
        author_column: str,
        algorithm_names: list[str] | None = None,
        metric_params: dict | None = None,
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
        return await self.compare_batch(
            author_labels, all_comparison_data, metric_params=metric_params
        )
