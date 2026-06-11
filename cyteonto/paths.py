from pathlib import Path

from .models import ModelArtifactKey


def artifact_key_segment(key: ModelArtifactKey) -> str:
    """Filename-safe segment ``{provider}_{company}-{modelName}``."""
    return key.filename_segment()


def _clean_identifier(name: str) -> str:
    """Normalise a free-form identifier (run id, algo name)."""
    return name.replace("/", "-").replace(":", "-").replace(" ", "_").replace(".", "_")


class PathConfig:
    """Resolves every file path used by the package.

    Layout produced under ``data_dir``::

        cell_ontology/cell_to_cell_ontology.csv
        cell_ontology/cl.owl
        embedding/cell_ontology/embeddings_<llmKey>_<embdKey>.npz
        embedding/descriptions/descriptions_<llmKey>.json

    And under ``user_dir`` (defaults to ``data_dir/user_files``)::

        embeddings/<run_id>/<kind>/<identifier>_embeddings_<llmKey>_<embdKey>.npz
        descriptions/<run_id>/<kind>/<identifier>_descriptions_<llmKey>.json
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.user_dir = Path(user_dir) if user_dir else self.data_dir / "user_files"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ontology_csv(self) -> Path:
        return self.data_dir / "cell_ontology" / "cell_to_cell_ontology.csv"

    @property
    def ontology_owl(self) -> Path:
        return self.data_dir / "cell_ontology" / "cl.owl"

    def ontology_embeddings(
        self, llm_key: ModelArtifactKey, embd_key: ModelArtifactKey
    ) -> Path:
        name = (
            f"embeddings_{artifact_key_segment(llm_key)}_"
            f"{artifact_key_segment(embd_key)}.npz"
        )
        path = self.data_dir / "embedding" / "cell_ontology" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def ontology_descriptions(self, llm_key: ModelArtifactKey) -> Path:
        name = f"descriptions_{artifact_key_segment(llm_key)}.json"
        path = self.data_dir / "embedding" / "descriptions" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def user_embeddings(
        self,
        run_id: str,
        kind: str,
        identifier: str,
        llm_key: ModelArtifactKey,
        embd_key: ModelArtifactKey,
    ) -> Path:
        name = (
            f"{_clean_identifier(identifier)}_embeddings_"
            f"{artifact_key_segment(llm_key)}_{artifact_key_segment(embd_key)}.npz"
        )
        path = self.user_dir / "embeddings" / _clean_identifier(run_id) / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def user_descriptions(
        self,
        run_id: str,
        kind: str,
        identifier: str,
        llm_key: ModelArtifactKey,
    ) -> Path:
        name = (
            f"{_clean_identifier(identifier)}_descriptions_"
            f"{artifact_key_segment(llm_key)}.json"
        )
        path = self.user_dir / "descriptions" / _clean_identifier(run_id) / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run_embedding_dirs(self, run_id: str) -> list[Path]:
        """Top-level directories under which all artifacts for ``run_id`` live."""
        clean = _clean_identifier(run_id)
        return [
            self.user_dir / "embeddings" / clean,
            self.user_dir / "descriptions" / clean,
        ]

    def run_kind_dirs(self, run_id: str, kind: str) -> list[Path]:
        clean = _clean_identifier(run_id)
        return [
            self.user_dir / "embeddings" / clean / kind,
            self.user_dir / "descriptions" / clean / kind,
        ]

    def core_files_present(self) -> dict[str, bool]:
        return {
            "ontology_csv": self.ontology_csv.exists(),
            "ontology_owl": self.ontology_owl.exists(),
        }
