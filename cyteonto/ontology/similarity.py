# cyteonto/ontology/similarity.py
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
from owlready2 import Ontology, ThingClass, get_ontology  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from ..logger_config import logger


class OntologySimilarity:
    """Computes ontology-based similarity between cell types."""

    def __init__(
        self,
        owl_file_path: Path | None = None,
        embeddings_path: Path | None = None,
    ):
        """
        Initialize ontology similarity calculator.

        Args:
            owl_file_path: Path to Cell Ontology OWL file.
            embeddings_path: Path to ontology term embeddings NPZ file.
        """
        self.owl_file_path = owl_file_path
        self.embeddings_path = embeddings_path
        self._ontology: Ontology | None = None
        self._ontology_loaded = False

        # Caches and attributes for advanced similarity metrics
        self.embedding_map: dict[str, np.ndarray] = {}
        self.embedding_map_cosine_sim: pd.DataFrame | None = None
        self.embedding_labels: list[str] = []
        self.embedding_max: float = 0.0
        self._class_cache: dict[str, ThingClass] = {}
        self._ancestor_cache: dict[str, set[ThingClass]] = {}
        self._depth_cache: dict[str, int] = {}
        self.root_class = None

        # Load resources on initialization
        self._load_ontology()
        if self.embeddings_path:
            logger.info("Trying to get embedding map")
            self._load_embeddings()

    def _load_ontology_robust(
        self, ontology_url_or_path, max_retries=3
    ) -> tuple[Ontology, str]:
        """
        Robustly load an ontology with multiple fallback strategies.

        Args:
            ontology_url_or_path: URL or local path to ontology
            max_retries: Maximum number of retry attempts

        Returns:
            tuple: (ontology_object, loading_method_used)
        """
        onto = get_ontology(ontology_url_or_path)
        # Try local-only loading
        try:
            onto.load(only_local=True)
            print(
                "Loaded using only_local=True (some imported ontologies may be missing)"
            )
            return onto, "local_only"
        except Exception as e:
            print(f"Local-only loading failed: {e}")
        # Try with reload flag
        try:
            onto.load(reload=True)
            return onto, "reload_forced"
        except Exception as e:
            print(f"Forced reload failed: {e}")
        # Try local with reload
        try:
            onto.load(only_local=True, reload=True)
            print("Using local-only with forced reload (minimal imports)")
            return onto, "local_reload"
        except Exception as e:
            print(f"All loading strategies failed: {e}")

        # Last resort: Try normal loading
        for attempt in range(max_retries):
            try:
                onto.load()
                return onto, f"normal_load_attempt_{attempt + 1}"
            except Exception as e:
                print(f"Normal load attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    continue
            return None, "failed"
        return None, "failed"

    def _load_ontology(self) -> bool:
        """Load Cell Ontology OWL file."""
        if self._ontology_loaded:
            return self._ontology is not None
        try:
            path = (
                f"file://{self.owl_file_path.absolute()}"
                if self.owl_file_path and self.owl_file_path.exists()
                else "http://purl.obolibrary.org/obo/cl.owl"
            )
            self._ontology, method = self._load_ontology_robust(path)
            if self._ontology:
                logger.info(
                    f"Cell Ontology loaded successfully using method: {method}."
                )
                self.root_class = self._find_class_cached("CL:0000000")
            else:
                logger.error("Failed to load Cell Ontology after all attempts.")

        except Exception as e:
            logger.error(f"Failed to load Cell Ontology: {e}")
            self._ontology = None
        finally:
            self._ontology_loaded = True
        return self._ontology is not None

    def _load_embeddings(self):
        """Loads ontology term embeddings from an NPZ file."""
        if not self.embeddings_path:
            logger.info("[_load_embeddings] Embedding path not found")
            return
        try:
            logger.info(
                f"[_load_embeddings] Loading embeddings from {self.embeddings_path}..."
            )
            data = np.load(self.embeddings_path, allow_pickle=True)
            embeddings = data["embeddings"]
            labels = data["ontology_ids"]
            assert len(embeddings) == len(labels), (
                "Embeddings and labels length mismatch."
            )
            self.embedding_map = {_id: emb for _id, emb in zip(labels, embeddings)}
            self.embedding_labels = labels.tolist()

            sim_matrix = cosine_similarity(embeddings)
            self.embedding_map_cosine_sim = pd.DataFrame(
                sim_matrix,
                index=self.embedding_labels,
                columns=self.embedding_labels,
            )
            second_max = []
            for _, row in self.embedding_map_cosine_sim.iterrows():
                row_sorted = row.sort_values(ascending=False)
                second_max.append(row_sorted.iloc[1])
            self.second_max = np.array(second_max)
            self.embedding_max = self.second_max.max()

        except FileNotFoundError:
            logger.warning(f"Embeddings file not found at {self.embeddings_path}.")
        except Exception as e:
            logger.error(f"An error occurred while loading embeddings: {e}")

    # Caching Helpers
    def _find_class_cached(self, ontology_id: str) -> ThingClass | None:
        """Finds a class in the ontology using a cache."""
        if ontology_id in self._class_cache:
            return self._class_cache[ontology_id]
        if not self._ontology:
            return None
        iri_id = ontology_id.replace(":", "_")
        cls = self._ontology.search_one(iri=f"*{iri_id}")
        self._class_cache[ontology_id] = cls
        return cls

    def _get_ancestors_cached(self, cls: ThingClass) -> set:
        """Gets ancestors of a class using a cache (excludes the class itself)."""
        if not cls or not hasattr(cls, "iri"):
            return set()
        if cls.iri in self._ancestor_cache:
            return self._ancestor_cache[cls.iri]
        ancestors = {anc for anc in cls.ancestors() if "CL_" in str(anc.iri)}
        # Discard self to avoid trivial similarity
        if cls in ancestors:
            ancestors.discard(cls)
        self._ancestor_cache[cls.iri] = ancestors
        return ancestors

    def _get_depth(self, cls: ThingClass) -> int:
        """Calculates the depth of a class (longest path to root)."""
        if not cls or not hasattr(cls, "is_a"):
            return 0
        if cls in self._depth_cache:
            return self._depth_cache[cls]
        if cls == self.root_class:
            return 0

        parents = [
            p for p in cls.is_a if isinstance(p, ThingClass) and "CL_" in str(p.iri)
        ]
        if not parents:
            self._depth_cache[cls] = 0
            return 0

        max_depth = max((self._get_depth(p) for p in parents), default=0)
        self._depth_cache[cls] = max_depth + 1
        return max_depth + 1

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        if v1 is None or v2 is None:
            return 0.0
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        # return np.dot(v1, v2) / (norm_v1 * norm_v2)
        return cosine_similarity([v1], [v2])[0][0]

    @staticmethod
    def gaussian_hill(x, center=1, width=0.2, amplitude=1):
        """
        Computes a Gaussian hill function.
        Args:
            x: Input value.
            center: Center of the Gaussian.
            width: Width (standard deviation) of the Gaussian.
            amplitude: Amplitude (height) of the Gaussian.
        Returns:
            Gaussian hill value at x.
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))

    # --- Similarity Calculation Primitives ---

    def _set_sim(self, ancestors1: set, ancestors2: set, method: str):
        intersection = ancestors1 & ancestors2
        union = ancestors1 | ancestors2
        if not union:
            return 0.0
        if method == "jaccard":
            return len(intersection) / len(union)
        if method == "cosine":
            if not ancestors1 or not ancestors2:
                return 0.0
            return len(intersection) / np.sqrt(len(ancestors1) * len(ancestors2))
        if method == "weighted_jaccard":
            weights1 = {a: self._get_depth(a) for a in ancestors1}
            weights2 = {a: self._get_depth(a) for a in ancestors2}
            weight_intersection = sum(
                min(weights1.get(a, 0), weights2.get(a, 0)) for a in union
            )
            weight_union = sum(
                max(weights1.get(a, 0), weights2.get(a, 0)) for a in union
            )
            return weight_intersection / weight_union if weight_union > 0 else 0.0
        raise ValueError(f"Unknown set similarity method: {method}")

    def _weighted_sim(self, cl_id1, cl_id2, ancestors1, ancestors2, method):
        union = ancestors1 | ancestors2
        if not union:
            return 0.0
        weights1, weights2 = {}, {}
        if method == "num_ancestors":
            for a in union:
                weight = 1.0 / max(len(self._get_ancestors_cached(a)) + 1, 1)
                if a in ancestors1:
                    weights1[a] = weight
                if a in ancestors2:
                    weights2[a] = weight
        elif method == "specificity":
            max_depth1 = self._get_depth(self._find_class_cached(cl_id1)) or 1
            max_depth2 = self._get_depth(self._find_class_cached(cl_id2)) or 1
            for a in union:
                depth = self._get_depth(a)
                if a in ancestors1:
                    weights1[a] = depth / max_depth1
                if a in ancestors2:
                    weights2[a] = depth / max_depth2
        elif method == "embedding_cosine":
            if not self.embedding_map:
                return 0.0
            emb1, emb2 = self.embedding_map.get(cl_id1), self.embedding_map.get(cl_id2)
            if emb1 is None or emb2 is None:
                return 0.0
            for a in union:
                a_emb = self.embedding_map.get(a.name.replace("_", ":"))
                if a_emb is None:
                    continue
                if a in ancestors1:
                    weights1[a] = self._cosine_similarity(a_emb, emb1)
                if a in ancestors2:
                    weights2[a] = self._cosine_similarity(a_emb, emb2)
        else:
            raise ValueError(f"Unknown weighted similarity method: {method}")

        weight_intersection = sum(
            min(weights1.get(a, 0), weights2.get(a, 0)) for a in union
        )
        weight_union = sum(max(weights1.get(a, 0), weights2.get(a, 0)) for a in union)
        return weight_intersection / weight_union if weight_union > 0 else 0.0

    def _path_sim(self, cl_id1: ThingClass, cl_id2: ThingClass) -> float:
        ancestors1 = self._get_ancestors_cached(cl_id1)
        ancestors2 = self._get_ancestors_cached(cl_id2)
        intersection = ancestors1 & ancestors2
        if not intersection:
            return 0.0
        lca = max(intersection, key=lambda a: self._get_depth(a))
        lca_depth = self._get_depth(lca)
        # Using depth directly instead of ancestor count for path calculation
        # d1 = self._get_depth(list(ancestors1 - intersection)[0]) if (ancestors1 - intersection) else lca_depth
        # d2 = self._get_depth(list(ancestors2 - intersection)[0]) if (ancestors2 - intersection) else lca_depth
        d1 = self._get_depth(cl_id1) - lca_depth
        d2 = self._get_depth(cl_id2) - lca_depth
        if d1 + d2 == 0:
            return 0.0
        avg_depth = (d1 + d2) / 2
        return 1.0 / avg_depth

    # --- User-Facing Methods ---

    def compute_simple_similarity(self, term1: str, term2: str) -> float:
        """Compute simple string-based similarity."""
        term1_norm = term1.lower().replace("_", " ").replace("-", " ")
        term2_norm = term2.lower().replace("_", " ").replace("-", " ")
        return SequenceMatcher(None, term1_norm, term2_norm).ratio()

    def compute_ontology_similarity(
        self,
        ontology_id1: str,
        ontology_id2: str,
        ontology_score1: float = 1.0,
        ontology_score2: float = 1.0,
        metric: str = "cosine_kernel",
        metric_params: dict | None = None,
    ) -> float:
        """
        Compute similarity between two ontology terms using a specified metric.

        Args:
            ontology_id1: First ontology term ID (e.g., "CL:0000000").
            ontology_id2: Second ontology term ID (e.g., "CL:0000001").
            metric: The similarity metric to use.
                - Set-based: 'set:jaccard', 'set:cosine', 'set:weighted_jaccard'
                - Weighted: 'weighted:num_ancestors', 'weighted:specificity', 'weighted:embedding_cosine'
                - Path-based: 'path'
                - Embedding-based: 'cosine' (direct embedding similarity)
                - Ensemble: 'cosine_ensemble'
                - Ensemble: 'cosine_kernel' (DEFAULT; embedding with Gaussian hill)
                - Combined: 'final'

            metric_params: Optional dictionary of parameters for the metric.
                For 'cosine_kernel', supported keys are:
                - center: Center of the Gaussian (default: 1)
                - width: Width of the Gaussian (default: 0.25)
                - amplitude: Amplitude of the Gaussian (default: 1)

        Returns:
            Similarity score between 0 and 1.
        """
        if not isinstance(ontology_id1, str) or not isinstance(ontology_id2, str):
            return 0.0
        if ontology_id1 == ontology_id2:
            return 1.0
        if self._ontology is None:
            logger.warning(
                "Ontology not loaded. Falling back to simple string similarity."
            )
            return self.compute_simple_similarity(ontology_id1, ontology_id2)

        class1 = self._find_class_cached(ontology_id1)
        class2 = self._find_class_cached(ontology_id2)

        if not class1 or not class2:
            missing = f"{ontology_id1 if not class1 else ''} {ontology_id2 if not class2 else ''}"
            logger.warning(
                f"Classes not found in ontology: {missing.strip()}. Using string similarity."
            )
            return self.compute_simple_similarity(ontology_id1, ontology_id2)

        try:
            if metric == "simple":
                return self.compute_simple_similarity(ontology_id1, ontology_id2)

            if metric == "cosine_ensemble":
                # Ensemble of similarities for assigned terms (d1, d2)
                # and the cosine similarity of their embeddings (d3)
                # divided by max similarity between all pairs of assigned terms (dm)
                # similarity = ((d1 + d2 + d3) / (3*dm)
                d1 = ontology_score1
                d2 = ontology_score2
                emb1 = self.embedding_map.get(ontology_id1)
                emb2 = self.embedding_map.get(ontology_id2)
                if emb1 is None or emb2 is None:
                    return 0.0
                d3 = self._cosine_similarity(emb1, emb2)
                dm = self.embedding_max
                if dm == 0:
                    return 0.0
                return (d1 + d2 + d3) / (3 * dm)

            if metric == "cosine_kernel":
                embd1 = self.embedding_map[ontology_id1.replace("_", ":")]
                embd2 = self.embedding_map[ontology_id2.replace("_", ":")]
                logger.debug(f"Embeddings: {embd1[:5]}, {embd2[:5]}")
                d3 = self._cosine_similarity(embd1, embd2)  # type:ignore
                logger.debug(f"Embedding Cosine: {d3}")

                # Get parameters with defaults
                params = metric_params or {}
                center = params.get("center", 1)
                width = params.get("width", 0.25)
                amplitude = params.get("amplitude", 1)

                d3_hill = self.gaussian_hill(
                    d3, center=center, width=width, amplitude=amplitude
                )
                logger.debug(f"Embedding Cosine Hill: {d3_hill}")
                return d3_hill

            # Direct embedding cosine similarity
            if metric == "cosine_direct":
                emb1 = self.embedding_map[ontology_id1]
                emb2 = self.embedding_map[ontology_id2]
                return self._cosine_similarity(emb1, emb2)

            # Get ancestors (excluding self) for hierarchical metrics
            ancestors1 = self._get_ancestors_cached(class1)
            ancestors2 = self._get_ancestors_cached(class2)

            if metric == "final":
                # For 'final' score, we use a predefined combination
                # Ancestors for set sim should include the class itself
                set_ancestors1 = ancestors1 | {class1}
                set_ancestors2 = ancestors2 | {class2}
                jaccard = self._set_sim(set_ancestors1, set_ancestors2, "jaccard")
                specificity = self._weighted_sim(
                    ontology_id1, ontology_id2, ancestors1, ancestors2, "specificity"
                )
                path = self._path_sim(class1, class2)
                return 0.3 * jaccard + 0.5 * specificity + 0.2 * path

            # Dispatch to appropriate metric calculation
            metric_group, _, metric_method = metric.partition(":")

            if metric_group == "set":
                # Set-based metrics traditionally include the term itself.
                set_ancestors1 = ancestors1 | {class1}
                set_ancestors2 = ancestors2 | {class2}
                return self._set_sim(set_ancestors1, set_ancestors2, metric_method)
            elif metric_group == "weighted":
                return self._weighted_sim(
                    ontology_id1, ontology_id2, ancestors1, ancestors2, metric_method
                )
            elif metric_group == "path":
                return self._path_sim(class1, class2)
            else:
                logger.error(
                    f"Unknown metric '{metric}'. Falling back to simple similarity."
                )
                return self.compute_simple_similarity(ontology_id1, ontology_id2)

        except Exception as e:
            logger.error(
                f"Error computing '{metric}' for {ontology_id1} vs {ontology_id2}: {e}"
            )
            return self.compute_simple_similarity(ontology_id1, ontology_id2)
