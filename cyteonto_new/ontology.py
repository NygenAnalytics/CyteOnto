"""Cell Ontology mapping and similarity computations."""

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from owlready2 import Ontology, ThingClass, get_ontology  # type: ignore

from .logger import logger


class OntologyMapping:
    """Wrapper around the cell_ontology CSV mapping (label <-> CL id)."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None
        self._label_to_id: dict[str, str] = {}
        self._id_to_labels: dict[str, list[str]] = {}
        self._loaded = False

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            df = pd.read_csv(self.csv_path)
            self._df = df
            for _, row in df.iterrows():
                label = str(row["label"])
                oid = str(row["ontology_id"])
                self._id_to_labels.setdefault(oid, []).append(label)
                self._label_to_id.setdefault(label, oid)
            self._loaded = True
            logger.info(f"Loaded {len(df)} ontology mappings from {self.csv_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ontology mapping: {e}")
            return False

    @property
    def df(self) -> pd.DataFrame:
        if not self._loaded:
            self.load()
        assert self._df is not None
        return self._df

    def label_to_id(self, label: str) -> str | None:
        if not self._loaded:
            self.load()
        return self._label_to_id.get(label)

    def labels_for_id(self, ontology_id: str) -> list[str]:
        if not self._loaded:
            self.load()
        return self._id_to_labels.get(ontology_id, [])

    def ids_and_joined_labels(self) -> tuple[list[str], list[str]]:
        """Return (ontology_ids, labels_joined_by_semicolon) for description generation."""
        if not self._loaded:
            self.load()
        assert self._df is not None
        grouped = self._df.groupby("ontology_id")["label"].apply(";".join).reset_index()
        return grouped["ontology_id"].tolist(), grouped["label"].tolist()


class OntologySimilarity:
    """Compute similarity between two CL terms under several metrics.

    Supported ``metric`` values:

    * ``simple``                  string SequenceMatcher (fallback only)
    * ``cosine_direct``           raw cosine similarity of term embeddings
    * ``cosine_kernel`` (default) ``cosine_direct`` warped by a Gaussian hill
    * ``path``                    1 / average depth from the lowest common ancestor
    * ``set:jaccard``             jaccard of ancestor sets
    * ``set:cosine``              cosine of ancestor set sizes
    * ``set:weighted_jaccard``    jaccard weighted by ancestor depth
    * ``weighted:num_ancestors``  jaccard weighted by inverse ancestor count
    * ``weighted:specificity``    jaccard weighted by relative depth
    * ``weighted:embedding_cosine`` jaccard weighted by ancestor-to-term cosine
    """

    def __init__(
        self,
        owl_path: Path | None,
        embeddings_path: Path | None = None,
    ) -> None:
        self.owl_path = owl_path
        self.embeddings_path = embeddings_path

        self._ontology: Ontology | None = None
        self._root: ThingClass | None = None
        self._class_cache: dict[str, ThingClass | None] = {}
        self._ancestor_cache: dict[str, set[ThingClass]] = {}
        self._depth_cache: dict[Any, int] = {}

        self.embedding_map: dict[str, np.ndarray] = {}

        self._load_ontology()
        if embeddings_path:
            self._load_embeddings()

    def _load_ontology(self) -> None:
        if not self.owl_path or not self.owl_path.exists():
            logger.warning(f"OWL file not found at {self.owl_path}")
            return
        try:
            onto = get_ontology(f"file://{self.owl_path.absolute()}")
            try:
                onto.load(only_local=True)
            except Exception:
                onto.load()
            self._ontology = onto
            self._root = self._find_class("CL:0000000")
            logger.info(f"Loaded Cell Ontology from {self.owl_path}")
        except Exception as e:
            logger.error(f"Failed to load Cell Ontology: {e}")

    def _load_embeddings(self) -> None:
        if not self.embeddings_path or not self.embeddings_path.exists():
            logger.info(f"No ontology embeddings at {self.embeddings_path}")
            return
        try:
            data = np.load(self.embeddings_path, allow_pickle=True)
            embeddings = data["embeddings"]
            ids = data["ontology_ids"].tolist()
            self.embedding_map = {str(i): emb for i, emb in zip(ids, embeddings)}
            logger.info(f"Loaded {len(self.embedding_map)} ontology embeddings")
        except Exception as e:
            logger.error(f"Failed to load ontology embeddings: {e}")

    def _find_class(self, ontology_id: str) -> ThingClass | None:
        if ontology_id in self._class_cache:
            return self._class_cache[ontology_id]
        if not self._ontology:
            self._class_cache[ontology_id] = None
            return None
        iri_id = ontology_id.replace(":", "_")
        cls = self._ontology.search_one(iri=f"*{iri_id}")
        self._class_cache[ontology_id] = cls
        return cls

    def _ancestors(self, cls: ThingClass) -> set[ThingClass]:
        if not cls or not hasattr(cls, "iri"):
            return set()
        if cls.iri in self._ancestor_cache:
            return self._ancestor_cache[cls.iri]
        ancs = {a for a in cls.ancestors() if "CL_" in str(a.iri)}
        ancs.discard(cls)
        self._ancestor_cache[cls.iri] = ancs
        return ancs

    def _depth(self, cls: ThingClass) -> int:
        if not cls or not hasattr(cls, "is_a"):
            return 0
        if cls in self._depth_cache:
            return self._depth_cache[cls]
        if cls == self._root:
            return 0
        parents = [
            p for p in cls.is_a if isinstance(p, ThingClass) and "CL_" in str(p.iri)
        ]
        if not parents:
            self._depth_cache[cls] = 0
            return 0
        d = max((self._depth(p) for p in parents), default=0) + 1
        self._depth_cache[cls] = d
        return d

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    @staticmethod
    def _gaussian_hill(
        x: float, center: float = 1.0, width: float = 0.25, amplitude: float = 1.0
    ) -> float:
        return float(amplitude * np.exp(-((x - center) ** 2) / (2 * width**2)))

    @staticmethod
    def _simple(term1: str, term2: str) -> float:
        a = term1.lower().replace("_", " ").replace("-", " ")
        b = term2.lower().replace("_", " ").replace("-", " ")
        return float(SequenceMatcher(None, a, b).ratio())

    def _set_sim(
        self, anc1: set[ThingClass], anc2: set[ThingClass], method: str
    ) -> float:
        union = anc1 | anc2
        if not union:
            return 0.0
        inter = anc1 & anc2
        if method == "jaccard":
            return len(inter) / len(union)
        if method == "cosine":
            if not anc1 or not anc2:
                return 0.0
            return len(inter) / float(np.sqrt(len(anc1) * len(anc2)))
        if method == "weighted_jaccard":
            w1 = {a: self._depth(a) for a in anc1}
            w2 = {a: self._depth(a) for a in anc2}
            num = sum(min(w1.get(a, 0), w2.get(a, 0)) for a in union)
            den = sum(max(w1.get(a, 0), w2.get(a, 0)) for a in union)
            return num / den if den > 0 else 0.0
        raise ValueError(f"Unknown set method: {method}")

    def _weighted_sim(
        self,
        id1: str,
        id2: str,
        anc1: set[ThingClass],
        anc2: set[ThingClass],
        method: str,
    ) -> float:
        union = anc1 | anc2
        if not union:
            return 0.0
        w1: dict[Any, float] = {}
        w2: dict[Any, float] = {}

        if method == "num_ancestors":
            for a in union:
                w = 1.0 / max(len(self._ancestors(a)) + 1, 1)
                if a in anc1:
                    w1[a] = w
                if a in anc2:
                    w2[a] = w
        elif method == "specificity":
            d1 = self._depth(self._find_class(id1)) or 1
            d2 = self._depth(self._find_class(id2)) or 1
            for a in union:
                da = self._depth(a)
                if a in anc1:
                    w1[a] = da / d1
                if a in anc2:
                    w2[a] = da / d2
        elif method == "embedding_cosine":
            if not self.embedding_map:
                return 0.0
            e1 = self.embedding_map.get(id1)
            e2 = self.embedding_map.get(id2)
            if e1 is None or e2 is None:
                return 0.0
            for a in union:
                a_id = a.name.replace("_", ":") if hasattr(a, "name") else None
                a_emb = self.embedding_map.get(a_id) if a_id else None
                if a_emb is None:
                    continue
                if a in anc1:
                    w1[a] = self._cosine(a_emb, e1)
                if a in anc2:
                    w2[a] = self._cosine(a_emb, e2)
        else:
            raise ValueError(f"Unknown weighted method: {method}")

        num = sum(min(w1.get(a, 0.0), w2.get(a, 0.0)) for a in union)
        den = sum(max(w1.get(a, 0.0), w2.get(a, 0.0)) for a in union)
        return num / den if den > 0 else 0.0

    def _path_sim(self, cls1: ThingClass, cls2: ThingClass) -> float:
        anc1 = self._ancestors(cls1)
        anc2 = self._ancestors(cls2)
        inter = anc1 & anc2
        if not inter:
            return 0.0
        lca = max(inter, key=lambda a: self._depth(a))
        lca_depth = self._depth(lca)
        d1 = self._depth(cls1) - lca_depth
        d2 = self._depth(cls2) - lca_depth
        if d1 + d2 == 0:
            return 0.0
        return 1.0 / ((d1 + d2) / 2)

    def similarity(
        self,
        id1: str,
        id2: str,
        metric: str = "cosine_kernel",
        metric_params: dict[str, Any] | None = None,
    ) -> float:
        if not isinstance(id1, str) or not isinstance(id2, str):
            return 0.0
        if id1 == id2:
            return 1.0

        params = metric_params or {}

        if metric == "simple":
            return self._simple(id1, id2)

        if metric == "cosine_direct":
            e1 = self.embedding_map.get(id1)
            e2 = self.embedding_map.get(id2)
            if e1 is None or e2 is None:
                logger.debug(f"Missing embedding for {id1} or {id2}")
                return 0.0
            return self._cosine(e1, e2)

        if metric == "cosine_kernel":
            e1 = self.embedding_map.get(id1)
            e2 = self.embedding_map.get(id2)
            if e1 is None or e2 is None:
                logger.debug(f"Missing embedding for {id1} or {id2}")
                return 0.0
            return self._gaussian_hill(
                self._cosine(e1, e2),
                center=params.get("center", 1.0),
                width=params.get("width", 0.25),
                amplitude=params.get("amplitude", 1.0),
            )

        if self._ontology is None:
            logger.warning("Ontology not loaded; falling back to simple similarity")
            return self._simple(id1, id2)

        cls1 = self._find_class(id1)
        cls2 = self._find_class(id2)
        if cls1 is None or cls2 is None:
            logger.warning(
                f"Class not found for {id1} or {id2}; using simple similarity"
            )
            return self._simple(id1, id2)

        anc1 = self._ancestors(cls1)
        anc2 = self._ancestors(cls2)

        if metric == "path":
            return self._path_sim(cls1, cls2)

        group, _, method = metric.partition(":")
        try:
            if group == "set":
                return self._set_sim(anc1 | {cls1}, anc2 | {cls2}, method)
            if group == "weighted":
                return self._weighted_sim(id1, id2, anc1, anc2, method)
        except ValueError as e:
            logger.error(f"Metric error: {e}; falling back to simple similarity")
            return self._simple(id1, id2)

        logger.error(f"Unknown metric '{metric}'; falling back to simple similarity")
        return self._simple(id1, id2)
