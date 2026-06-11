"""Tests for cyteonto.paths.PathConfig (schema v3 artifact keys)."""

from cyteonto.models import EmbdConfig, LlmConfig
from cyteonto.paths import PathConfig, artifact_key_segment


class TestPathConfig:
    def test_init_default_paths(self):
        config = PathConfig()
        assert config.data_dir.exists()
        assert config.user_dir.exists()

    def test_init_custom_paths(self, temp_dir):
        base_path = str(temp_dir / "custom_base")
        user_path = str(temp_dir / "custom_user")
        config = PathConfig(base_path, user_path)
        assert config.data_dir.name == "custom_base"
        assert config.user_dir.name == "custom_user"

    def test_artifact_key_segment(self):
        llm = LlmConfig(provider="together", model="moonshotai/Kimi-K2.6").to_artifact_key()
        assert artifact_key_segment(llm) == "together_moonshotai-Kimi-K2.6"

    def test_core_files_present(self, temp_dir):
        config = PathConfig(str(temp_dir))
        ontology_dir = temp_dir / "cell_ontology"
        ontology_dir.mkdir(parents=True)
        (ontology_dir / "cell_to_cell_ontology.csv").touch()
        (ontology_dir / "cl.owl").touch()
        result = config.core_files_present()
        assert result["ontology_csv"] is True
        assert result["ontology_owl"] is True

    def test_path_uniqueness(self, temp_dir):
        config = PathConfig(str(temp_dir), str(temp_dir / "user"))
        llm1 = LlmConfig(provider="together", model="moonshotai/Kimi-K2.5").to_artifact_key()
        llm2 = LlmConfig(provider="together", model="moonshotai/Kimi-K2.6").to_artifact_key()
        embd = EmbdConfig().to_artifact_key()
        assert config.ontology_embeddings(llm1, embd) != config.ontology_embeddings(
            llm2, embd
        )
