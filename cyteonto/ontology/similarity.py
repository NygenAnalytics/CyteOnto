# cyteonto/ontology/similarity.py

from difflib import SequenceMatcher
from pathlib import Path

from owlready2 import get_ontology  # type: ignore

from ..logger_config import logger


class OntologySimilarity:
    """Computes ontology-based similarity between cell types."""

    def __init__(self, owl_file_path: Path | None = None):
        """
        Initialize ontology similarity calculator.

        Args:
            owl_file_path: Path to Cell Ontology OWL file
        """
        self.owl_file_path = owl_file_path
        self._ontology = None
        self._ontology_loaded = False

    def _load_ontology(self) -> bool:
        """
        Load Cell Ontology OWL file.

        Returns:
            True if successful, False otherwise
        """

        if self._ontology_loaded:
            return self._ontology is not None

        try:
            if self.owl_file_path and self.owl_file_path.exists():
                # Load local OWL file
                self._ontology = get_ontology(f"file://{self.owl_file_path.absolute()}")
                logger.info("Loading local Cell Ontology OWL file...")
            else:
                # Load from URL
                self._ontology = get_ontology("http://purl.obolibrary.org/obo/cl.owl")
                logger.info("Loading Cell Ontology from URL...")

            if self._ontology is not None:
                self._ontology.load()
            self._ontology_loaded = True
            logger.info("Cell Ontology loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Cell Ontology: {e}")
            self._ontology = None
            self._ontology_loaded = True
            return False

    def compute_simple_similarity(self, term1: str, term2: str) -> float:
        """
        Compute simple string-based similarity when OWL ontology is not available.

        Args:
            term1: First term
            term2: Second term

        Returns:
            Similarity score between 0 and 1
        """
        # Normalize terms
        term1_norm = term1.lower().replace("_", " ").replace("-", " ")
        term2_norm = term2.lower().replace("_", " ").replace("-", " ")

        # Compute similarity
        similarity = SequenceMatcher(None, term1_norm, term2_norm).ratio()

        # Boost similarity for exact matches
        if term1_norm == term2_norm:
            similarity = 1.0

        return similarity

    def _get_ancestors(self, cls):
        """Get ancestors of a class."""
        return set(cls.ancestors()) if cls else set()

    def compute_ontology_similarity(
        self, ontology_id1: str, ontology_id2: str
    ) -> float:
        """
        Compute similarity between two ontology terms using weighted ancestor intersection.
        Falls back to simple string similarity if OWL loading fails.

        Args:
            ontology_id1: First ontology term ID (e.g., "CL:0000000")
            ontology_id2: Second ontology term ID (e.g., "CL:0000001")

        Returns:
            Similarity score between 0 and 1
        """
        if not isinstance(ontology_id1, str) or not isinstance(ontology_id2, str):
            return 0.0

        # Quick check for identical terms
        if ontology_id1 == ontology_id2:
            return 1.0

        # Load ontology if not loaded
        if not self._ontology_loaded:
            self._load_ontology()

        if self._ontology is None:
            # Fallback to simple string similarity
            logger.warning(
                f"Ontology not loaded, falling back to simple string similarity for {ontology_id1} and {ontology_id2}"
            )
            return self.compute_simple_similarity(ontology_id1, ontology_id2)

        # check if ontology_id1 and ontology_id2 are ontology id format
        if not ontology_id1.startswith("CL:") or not ontology_id2.startswith("CL:"):
            logger.warning(
                f"Ontology IDs are not in CL: format, falling back to simple string similarity for {ontology_id1} and {ontology_id2}"
            )
            return self.compute_simple_similarity(ontology_id1, ontology_id2)

        try:
            # Convert CL:0000000 format to URI format for search
            class1 = self._ontology.search_one(iri="*" + ontology_id1.replace(":", "_"))
            class2 = self._ontology.search_one(iri="*" + ontology_id2.replace(":", "_"))

            if not class1 or not class2:
                # Fallback to simple string similarity
                logger.warning(
                    f"Ontology IDs are not found in the ontology, falling back to simple string similarity for {ontology_id1} and {ontology_id2}"
                )
                return self.compute_simple_similarity(ontology_id1, ontology_id2)

            def weighted_ancestors(cls):
                """Get weighted ancestors where weight is inverse of depth"""
                ancestors = self._get_ancestors(cls)
                return {a: 1.0 / max(1, len(self._get_ancestors(a))) for a in ancestors}

            wa1 = weighted_ancestors(class1)
            wa2 = weighted_ancestors(class2)

            all_ancestors = set(wa1.keys()) | set(wa2.keys())
            intersection = set(wa1.keys()) & set(wa2.keys())

            if not all_ancestors:
                return 0.0

            # Compute weighted Jaccard similarity
            weight_sum_intersection = sum(
                (wa1.get(a, 0) + wa2.get(a, 0)) / 2 for a in intersection
            )
            weight_sum_union = sum(
                (wa1.get(a, 0) + wa2.get(a, 0)) / 2 for a in all_ancestors
            )

            if weight_sum_union == 0:
                return 0.0

            return weight_sum_intersection / weight_sum_union

        except Exception as e:
            logger.error(
                f"Error computing ontology similarity between {ontology_id1} and {ontology_id2}: {e}"
            )
            return self.compute_simple_similarity(ontology_id1, ontology_id2)

    def compute_batch_similarities(
        self, ontology_pairs: list[tuple[str, str]]
    ) -> list[float]:
        """
        Compute similarities for multiple ontology term pairs.

        Args:
            ontology_pairs: List of (ontology_id1, ontology_id2) tuples

        Returns:
            List of similarity scores
        """
        similarities = []
        for ont1, ont2 in ontology_pairs:
            similarity = self.compute_ontology_similarity(ont1, ont2)
            similarities.append(similarity)

        logger.info(f"Computed {len(similarities)} ontology similarities")
        return similarities
