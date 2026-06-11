import json

import numpy as np

from cyteonto.config import Config
from cyteonto.models import (
    CellDescription,
    EmbdConfig,
    LlmConfig,
    ModelArtifactKey,
    ModelPairUsage,
)
from cyteonto.paths import PathConfig
from cyteonto import storage

SCHEMA_VERSION = Config.SCHEMA_VERSION


class TestModelArtifactKey:
    def test_parse_slash_model(self):
        key = ModelArtifactKey(provider="FIREWORKS", model="moonshotai/Kimi-K2.6")
        assert key.provider == "fireworks"
        assert key.company == "moonshotai"
        assert key.modelName == "Kimi-K2.6"
        assert key.filename_segment() == "fireworks_moonshotai-Kimi-K2.6"

    def test_parse_model_without_slash(self):
        key = ModelArtifactKey(provider="openai", model="gpt-4o")
        assert key.company == "gpt-4o"
        assert key.modelName == "gpt-4o"
        assert key.filename_segment() == "openai_gpt-4o-gpt-4o"

    def test_embedding_key(self):
        key = EmbdConfig(provider="openrouter", model="qwen/qwen3-embedding-8b").to_artifact_key()
        assert key.filename_segment() == "openrouter_qwen-qwen3-embedding-8b"

    def test_llm_config_to_artifact_key(self):
        llm = LlmConfig(provider="together", model="moonshotai/Kimi-K2.6")
        assert llm.to_artifact_key().filename_segment() == "together_moonshotai-Kimi-K2.6"


class TestPathConfig:
    def test_ontology_paths(self, temp_dir):
        paths = PathConfig(str(temp_dir))
        llm = LlmConfig(provider="together", model="moonshotai/Kimi-K2.6").to_artifact_key()
        embd = EmbdConfig().to_artifact_key()

        desc = paths.ontology_descriptions(llm)
        emb = paths.ontology_embeddings(llm, embd)

        assert desc.name == "descriptions_together_moonshotai-Kimi-K2.6.json"
        assert emb.name == (
            "embeddings_together_moonshotai-Kimi-K2.6_"
            "openrouter_qwen-qwen3-embedding-8b.npz"
        )

    def test_user_paths(self, temp_dir):
        paths = PathConfig(str(temp_dir), str(temp_dir / "user"))
        llm = LlmConfig(provider="fireworks", model="moonshotai/Kimi-K2.6").to_artifact_key()
        embd = EmbdConfig().to_artifact_key()

        desc = paths.user_descriptions("run.1", "algorithm", "algo-a", llm)
        emb = paths.user_embeddings("run.1", "algorithm", "algo-a", llm, embd)

        assert "run_1" in str(desc)
        assert desc.name == "algo-a_descriptions_fireworks_moonshotai-Kimi-K2.6.json"
        assert "algo-a_embeddings_fireworks_moonshotai-Kimi-K2.6_" in emb.name


class TestStorageV3:
    def test_descriptions_round_trip(self, temp_dir):
        path = temp_dir / "descriptions.json"
        key = ModelArtifactKey(provider="together", model="moonshotai/Kimi-K2.6")
        descriptions = {
            "CL:1": CellDescription(
                initialLabel="stem cell",
                descriptiveName="Stem cell",
                function="Self-renewal",
                diseaseRelevance="Not established",
                developmentalStage="Not established",
            )
        }

        assert storage.save_descriptions(path, descriptions, key)
        loaded = storage.load_descriptions(path)
        assert loaded is not None
        assert "CL:1" in loaded
        assert loaded["CL:1"].descriptiveName == "Stem cell"

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        assert raw["schemaVersion"] == SCHEMA_VERSION
        assert raw["artifactKey"]["provider"] == "together"

    def test_reject_legacy_flat_json(self, temp_dir):
        path = temp_dir / "legacy.json"
        path.write_text(
            json.dumps(
                {
                    "stem cell": {
                        "initialLabel": "stem cell",
                        "descriptiveName": "Stem cell",
                        "function": "x",
                        "diseaseRelevance": "Not established",
                        "developmentalStage": "Not established",
                    }
                }
            )
        )
        assert storage.load_descriptions(path) is None

    def test_ontology_embeddings_metadata(self, temp_dir):
        path = temp_dir / "onto.npz"
        llm = ModelArtifactKey(provider="together", model="moonshotai/Kimi-K2.6")
        embd = EmbdConfig().to_artifact_key()
        embeddings = np.zeros((2, 4), dtype=np.float32)

        storage.save_ontology_embeddings(
            path, embeddings, ["CL:1", "CL:2"], llm, embd
        )
        loaded = storage.load_ontology_embeddings(path)
        assert loaded is not None
        _, _, meta = loaded
        assert meta["schemaVersion"] == SCHEMA_VERSION
        assert meta["llm"]["provider"] == "together"
        assert meta["embedding"]["provider"] == "openrouter"


class TestModelPairs:
    def test_primary_preset_segments(self):
        cfg = Config()
        llm = LlmConfig(
            provider=cfg.PRIMARY_LLM_PROVIDER, model=cfg.PRIMARY_LLM_MODEL
        ).to_artifact_key()
        embd = EmbdConfig(
            provider=cfg.PRIMARY_EMBEDDING_PROVIDER,  # type: ignore[arg-type]
            model=cfg.PRIMARY_EMBEDDING_MODEL,
            modelSettings={},
        ).to_artifact_key()
        assert llm.filename_segment() == "nebius_moonshotai-Kimi-K2.6"
        assert embd.filename_segment() == "nebius_Qwen-Qwen3-Embedding-8B"

    def test_fallback_preset_segments(self):
        cfg = Config()
        llm = LlmConfig(
            provider=cfg.FALLBACK_LLM_PROVIDER, model=cfg.FALLBACK_LLM_MODEL
        ).to_artifact_key()
        embd = EmbdConfig(
            provider=cfg.FALLBACK_EMBEDDING_PROVIDER,  # type: ignore[arg-type]
            model=cfg.FALLBACK_EMBEDDING_MODEL,
        ).to_artifact_key()
        assert llm.filename_segment() == "fireworks_accounts-fireworks-models-kimi-k2p6"
        assert embd.filename_segment() == "openrouter_Qwen-Qwen3-Embedding-8B"

    def test_model_pair_usage_status_dict(self):
        llm = LlmConfig(provider="nebius", model="moonshotai/Kimi-K2.6")
        emb = EmbdConfig(provider="nebius", model="Qwen/Qwen3-Embedding-8B", modelSettings={})  # type: ignore[arg-type]
        usage = ModelPairUsage(llm=llm, embedding=emb)
        usage.llmTier = "mixed"
        usage.embeddingTier = "fallback"
        d = usage.to_status_dict()
        assert d["llmTier"] == "mixed"
        assert d["embeddingProvider"] == "nebius"
