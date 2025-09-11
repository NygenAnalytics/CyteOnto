# cyteonto/ontology/extractor.py

from pathlib import Path

import pandas as pd  # type: ignore

from ..logger_config import logger


class OntologyExtractor:
    """Extracts terms and mappings from Cell Ontology data."""

    def __init__(self, mapping_csv_path: Path):
        """
        Initialize ontology extractor.

        Args:
            mapping_csv_path: Path to cell ontology mapping CSV file
        """
        self.mapping_csv_path = mapping_csv_path
        self._mapping_df: pd.DataFrame | None = None
        self._ontology_to_labels: dict[str, list[str]] | None = None
        self._label_to_ontology: dict[str, str] | None = None

    def load_mapping(self) -> bool:
        """
        Load ontology mapping from CSV file.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._mapping_df = pd.read_csv(self.mapping_csv_path)
            logger.info(f"Loaded {len(self._mapping_df)} ontology mappings")
            return True
        except Exception as e:
            logger.error(f"Failed to load ontology mapping: {e}")
            return False

    def get_all_ontology_terms(self) -> list[str]:
        """
        Get all unique ontology IDs.

        Returns:
            List of unique ontology IDs
        """
        if self._mapping_df is None:
            if not self.load_mapping():
                return []

        if self._mapping_df is not None:
            ontology_ids = self._mapping_df["ontology_id"].unique().tolist()
        else:
            ontology_ids = []
        logger.info(f"Found {len(ontology_ids)} unique ontology terms")
        return ontology_ids

    def get_all_labels(self) -> list[str]:
        """
        Get all unique cell type labels.

        Returns:
            List of unique labels
        """
        if self._mapping_df is None:
            if not self.load_mapping():
                return []

        if self._mapping_df is not None:
            labels = self._mapping_df["label"].unique().tolist()
        else:
            labels = []
        logger.info(f"Found {len(labels)} unique labels")
        return labels

    def build_mappings(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        """
        Build bidirectional mappings between ontology IDs and labels.

        Returns:
            Tuple of (ontology_to_labels, label_to_ontology) dictionaries

        ontology_to_labels structure:
        {
            "CL:0000014": ["germ line stem cell", "germline stem cell"],
            "CL:0000035": ["single fate stem cell", "unipotent stem cell", "unipotential stem cell"],
            ...
        }

        label_to_ontology structure:
        {
            "germ line stem cell": "CL:0000014",
            "germline stem cell": "CL:0000014",
            "single fate stem cell": "CL:0000035",
            "unipotent stem cell": "CL:0000035",
            "unipotential stem cell": "CL:0000035",
            ...
        }
        """
        if self._mapping_df is None:
            if not self.load_mapping():
                return {}, {}

        if self._mapping_df is None:
            return {}, {}

        # Build ontology ID to labels mapping (one-to-many)
        ontology_to_labels: dict[str, list[str]] = {}
        for _, row in self._mapping_df.iterrows():
            ontology_id = row["ontology_id"]
            label = row["label"]

            if ontology_id not in ontology_to_labels:
                ontology_to_labels[ontology_id] = []
            ontology_to_labels[ontology_id].append(label)

        # Build label to ontology ID mapping (many-to-one, using first occurrence)
        label_to_ontology: dict[str, str] = {}
        for _, row in self._mapping_df.iterrows():
            label = row["label"]
            ontology_id = row["ontology_id"]

            if label not in label_to_ontology:
                label_to_ontology[label] = ontology_id

        self._ontology_to_labels = ontology_to_labels
        self._label_to_ontology = label_to_ontology
        return ontology_to_labels, label_to_ontology

    def get_ontology_id_for_label(self, label: str) -> str | None:
        """
        Get ontology ID for a given label.

        Args:
            label: Cell type label

        Returns:
            Ontology ID or None if not found
        """
        if self._label_to_ontology is None:
            self.build_mappings()

        if self._label_to_ontology is not None:
            return self._label_to_ontology.get(label)
        return None

    def get_labels_for_ontology_id(self, ontology_id: str) -> list[str]:
        """
        Get all labels for a given ontology ID.

        Args:
            ontology_id: Ontology term ID

        Returns:
            List of labels for this ontology ID
        """
        if self._ontology_to_labels is None:
            self.build_mappings()

        if self._ontology_to_labels is not None:
            return self._ontology_to_labels.get(ontology_id, [])
        return []

    def get_ontology_terms_for_description_generation(
        self,
    ) -> tuple[list[str], list[str]]:
        """
        Get ontology terms that need description generation.
        Returns unique ontology IDs with their primary labels.

        Returns:
            List of (ontology_id, primary_label) tuples
        """
        if self._mapping_df is None:
            if not self.load_mapping():
                return [], []

        if self._mapping_df is None:
            return [], []

        # Group by ontology_id and join labels with ;
        unique_df = (
            self._mapping_df.groupby("ontology_id")["label"]
            .apply(";".join)
            .reset_index()
        )
        ontology_ids = unique_df["ontology_id"].tolist()
        ontology_terms = unique_df["label"].tolist()

        logger.info(
            f"Identified {len(ontology_ids)} ontology terms for description generation"
        )
        return ontology_ids, ontology_terms
