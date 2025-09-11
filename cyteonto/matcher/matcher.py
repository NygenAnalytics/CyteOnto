# cyteonto/matcher/matcher.py

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from ..logger_config import logger
from ..ontology import OntologyExtractor, OntologySimilarity
from ..storage import FileManager, VectorStore


class CyteOntoMatcher:
    """
    Matcher class for CyteOnto providing cosine similarity and ontology hierarchy matching.
    """

    def __init__(
        self,
        embeddings_file_path: Path | None = None,
        base_data_path: str | None = None,
    ):
        """
        Initialize CyteOnto matcher.

        Args:
            embeddings_file_path: Path to embeddings NPZ file
            base_data_path: Base path for data files
        """
        self.file_manager = FileManager(base_data_path)
        self.vector_store = VectorStore()
        self.embeddings_file_path = embeddings_file_path

        # Cached data
        self._ontology_embeddings: np.ndarray | None = None
        self._ontology_ids: list[str] | None = None
        self._ontology_extractor: OntologyExtractor | None = None
        self._ontology_similarity: OntologySimilarity | None = None

        logger.info(f"Loading ontology embeddings from {self.embeddings_file_path}")
        self.embeddings_ready = self._load_ontology_embeddings()
        logger.info(f"Loaded success: {self.embeddings_ready}")

        logger.info("CyteOntoMatcher initialized")

    def _load_ontology_embeddings(self) -> bool:
        """
        Load ontology embeddings from file.

        Returns:
            True if successful, False otherwise
        """
        if self._ontology_embeddings is not None:
            return True  # Already loaded

        if self.embeddings_file_path is None:
            logger.error("No embeddings file path provided, maybe not generated")
            return False

        result = self.vector_store.load_embeddings(self.embeddings_file_path)
        if result is None:
            logger.error(
                "Failed to load ontology embeddings, maybe not generated or corrupted"
            )
            return False

        self._ontology_embeddings, self._ontology_ids, _ = result
        logger.info(f"Loaded {len(self._ontology_embeddings)} ontology embeddings")
        return True

    def _get_ontology_extractor(self) -> OntologyExtractor | None:
        """Get ontology extractor, creating if needed."""
        if self._ontology_extractor is None:
            mapping_path = self.file_manager.get_ontology_mapping_path()
            self._ontology_extractor = OntologyExtractor(mapping_path)
            if not self._ontology_extractor.load_mapping():
                logger.error("Failed to load ontology mapping")
                return None
        return self._ontology_extractor

    def _get_ontology_similarity(self) -> OntologySimilarity:
        """Get ontology similarity calculator, creating if needed."""
        if self._ontology_similarity is None:
            owl_path = self.file_manager.get_ontology_owl_path()
            self._ontology_similarity = OntologySimilarity(owl_path)
        return self._ontology_similarity

    def find_closest_ontology_terms(
        self, query_embeddings: np.ndarray, top_k: int = 5, min_similarity: float = 0.0
    ) -> list[list[dict]]:
        """
        Find closest ontology terms using cosine similarity.

        Args:
            query_embeddings: Array of query embeddings (n_queries, embedding_dim)
            top_k: Number of top matches to return per query
            min_similarity: Minimum similarity threshold

        Returns:
            list of lists, where each inner list contains top_k matches for a query.
            Each match is a dict with keys: label, ontology_id, similarity
        """
        if not self.embeddings_ready:
            logger.error(
                "Ontology embeddings not loaded, cannot find closest terms without embeddings"
            )
            return []

        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Compute cosine similarities
        similarities = cosine_similarity(query_embeddings, self._ontology_embeddings)

        results = []
        for i, query_similarities in enumerate(similarities):
            # Get top-k indices
            top_indices = np.argsort(query_similarities)[::-1][:top_k]

            query_results = []
            for idx in top_indices:
                similarity = float(query_similarities[idx])
                if similarity >= min_similarity:
                    query_results.append(
                        {
                            "label": self._ontology_ids[idx]
                            if self._ontology_ids
                            else None,
                            "ontology_id": self._ontology_ids[idx]
                            if self._ontology_ids
                            else None,
                            "similarity": similarity,
                        }
                    )

            results.append(query_results)
            logger.debug(
                f"Query {i}: found {len(query_results)} matches above threshold"
            )

        return results

    def compute_ontology_similarity(
        self,
        author_ontology_terms: list[str],
        user_ontology_terms: list[str],
        author_ontology_score: list[float] | None = None,
        user_ontology_score: list[float] | None = None,
        metric: str = "cosine_kernel",
    ) -> list[float]:
        """
        Compute ontology hierarchy-based similarity between author and user labels.

        Args:
            author_ontology_terms: list of author (reference) cell type ontology terms
            user_ontology_terms: list of user (predicted) cell type ontology terms

        Returns:
            list of similarity scores (one per pair)
        """
        extractor = self._get_ontology_extractor()
        if extractor is None:
            logger.error("Cannot compute ontology similarity without extractor")
            return [0.0] * len(author_ontology_terms)

        similarity_calc = self._get_ontology_similarity()

        # Build mappings if needed
        _, label_to_ontology = extractor.build_mappings()

        if metric == "cosine_ensemble" and (
            author_ontology_score is None or user_ontology_score is None
        ):
            logger.error(
                "Cosine ensemble metric requires author and user ontology scores"
            )
            return [0.0] * len(author_ontology_terms)
        else:
            # make arrays of 1.0 if scores not provided
            if author_ontology_score is None:
                author_ontology_score = [1.0] * len(author_ontology_terms)
            if user_ontology_score is None:
                user_ontology_score = [1.0] * len(user_ontology_terms)

        similarities = []
        for author_label, user_label, author_score, user_score in zip(
            author_ontology_terms,
            user_ontology_terms,
            author_ontology_score,
            user_ontology_score,
        ):
            # Get ontology IDs for labels
            author_ontology_id = label_to_ontology.get(author_label, author_label)
            user_ontology_id = label_to_ontology.get(user_label, user_label)

            # Compute ontology-based similarity
            similarity = similarity_calc.compute_ontology_similarity(
                author_ontology_id,
                user_ontology_id,
                author_score,
                user_score,
                metric=metric,
            )
            similarities.append(similarity)

        logger.debug(f"Computed ontology similarities for {len(similarities)} pairs")
        return similarities

    def match_embeddings_to_ontology(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 1,
        min_similarity: float = 0.1,
    ) -> list[dict | None]:
        """
        Match cell type ontology terms to ontology terms using embeddings.

        Args:
            query_embeddings: Embeddings for the labels
            top_k: Number of matches to consider (returns best)
            min_similarity: Minimum similarity threshold

        Returns:
            list of best ontology matches (one per label)
        """
        matches = self.find_closest_ontology_terms(
            query_embeddings, top_k=top_k, min_similarity=min_similarity
        )

        results: list[dict | None] = []
        for i, embedding in enumerate(query_embeddings):
            if i < len(matches) and len(matches[i]) > 0:
                best_match = matches[i][0]  # Take best match
                results.append(best_match)
            else:
                logger.warning(f"No ontology match found for embedding: {embedding}")
                results.append(None)

        return results
