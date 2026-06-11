"""Tests for cyteonto.models (schema v3 pydantic models)."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cyteonto.config import Config
from cyteonto.models import (
    AgentUsage,
    CellDescription,
    DescriptionFileEnvelope,
    EmbdConfig,
    LlmConfig,
    ModelArtifactKey,
    ModelPairUsage,
)


class TestCellDescription:
    def test_blank_is_blank(self):
        blank = CellDescription.blank("Monocyte")
        assert blank.initialLabel == "Monocyte"
        assert blank.is_blank() is True

    def test_get_blank_alias(self):
        assert CellDescription.get_blank().is_blank() is True

    def test_populated_is_not_blank(self, sample_cell_description):
        assert sample_cell_description.is_blank() is False

    def test_to_sentence(self, sample_cell_description):
        sentence = sample_cell_description.to_sentence()
        assert "T cell is CD4+ helper T lymphocyte" in sentence
        assert "Coordinates immune responses" in sentence
        assert "Critical in autoimmune diseases" in sentence
        assert "Mature adaptive immune cell" in sentence

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            CellDescription()


class TestModelArtifactKey:
    def test_provider_normalized_lowercase(self):
        key = ModelArtifactKey(provider="  FIREWORKS  ", model="moonshotai/Kimi-K2.6")
        assert key.provider == "fireworks"

    def test_company_and_model_name_split(self):
        key = ModelArtifactKey(provider="fireworks", model="moonshotai/Kimi-K2.6")
        assert key.company == "moonshotai"
        assert key.modelName == "Kimi-K2.6"

    def test_model_without_slash(self):
        key = ModelArtifactKey(provider="openai", model="gpt-4o")
        assert key.company == "gpt-4o"
        assert key.modelName == "gpt-4o"

    def test_filename_segment_preserves_dots(self):
        key = ModelArtifactKey(provider="fireworks", model="moonshotai/Kimi-K2.6")
        assert key.filename_segment() == "fireworks_moonshotai-Kimi-K2.6"

    def test_filename_segment_sanitizes_separators(self):
        key = ModelArtifactKey(provider="together", model="org/model name:v1")
        segment = key.filename_segment()
        assert "/" not in segment
        assert ":" not in segment
        assert " " not in segment

    def test_from_provider_model(self):
        key = ModelArtifactKey.from_provider_model("nebius", "Qwen/Qwen3-Embedding-8B")
        assert key.provider == "nebius"
        assert key.modelName == "Qwen3-Embedding-8B"


class TestLlmConfig:
    def test_to_artifact_key(self):
        llm = LlmConfig(provider="together", model="moonshotai/Kimi-K2.6")
        assert llm.to_artifact_key().filename_segment() == "together_moonshotai-Kimi-K2.6"


class TestEmbdConfig:
    def test_default_provider_applies_routing(self):
        cfg = EmbdConfig()
        assert cfg.provider == "openrouter"
        assert cfg.modelSettings is not None
        assert cfg.modelSettings["provider"]["order"] == ["deepinfra"]

    def test_explicit_empty_model_settings_disables_routing(self):
        cfg = EmbdConfig(provider="openrouter", modelSettings={})
        assert cfg.modelSettings == {}

    def test_non_openrouter_has_no_routing(self):
        cfg = EmbdConfig(provider="nebius", model="Qwen/Qwen3-Embedding-8B")
        assert cfg.modelSettings is None

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            EmbdConfig(provider="not-a-provider")  # type: ignore[arg-type]

    def test_max_concurrent_must_be_positive(self):
        with pytest.raises(ValidationError):
            EmbdConfig(maxConcurrent=0)

    def test_to_artifact_key(self):
        key = EmbdConfig(provider="openrouter", model="qwen/qwen3-embedding-8b").to_artifact_key()
        assert key.filename_segment() == "openrouter_qwen-qwen3-embedding-8b"


class TestDescriptionFileEnvelope:
    def test_default_schema_version(self):
        key = ModelArtifactKey(provider="together", model="moonshotai/Kimi-K2.6")
        envelope = DescriptionFileEnvelope(artifactKey=key)
        assert envelope.schemaVersion == Config.SCHEMA_VERSION
        assert envelope.descriptions == {}

    def test_round_trip(self, sample_cell_description):
        key = ModelArtifactKey(provider="together", model="moonshotai/Kimi-K2.6")
        envelope = DescriptionFileEnvelope(
            artifactKey=key, descriptions={"CL:1": sample_cell_description}
        )
        payload = envelope.model_dump(mode="json")
        restored = DescriptionFileEnvelope.model_validate(payload)
        assert restored.descriptions["CL:1"].descriptiveName == "CD4+ helper T lymphocyte"


class TestAgentUsage:
    def test_record_accumulates(self):
        usage = AgentUsage(agentName="CyteOnto")
        run = SimpleNamespace(
            requests=2, input_tokens=100, output_tokens=50, total_tokens=150
        )
        usage.record("model-a", run, {"pubmed": 1})
        assert usage.modelName == "model-a"
        assert usage.requests == 2
        assert usage.inputTokens == 100
        assert usage.outputTokens == 50
        assert usage.totalTokens == 150
        assert usage.toolUsage == {"pubmed": 1}

    def test_record_handles_none_tokens(self):
        usage = AgentUsage()
        run = SimpleNamespace(
            requests=1, input_tokens=None, output_tokens=None, total_tokens=None
        )
        usage.record("model-b", run)
        assert usage.requests == 1
        assert usage.inputTokens == 0
        assert usage.outputTokens == 0

    def test_merge(self):
        a = AgentUsage(requests=1, inputTokens=10, outputTokens=5, totalTokens=15)
        a.toolUsage = {"pubmed": 1}
        b = AgentUsage(requests=2, inputTokens=20, outputTokens=10, totalTokens=30)
        b.toolUsage = {"pubmed": 2, "other": 1}
        a.merge(b)
        assert a.requests == 3
        assert a.inputTokens == 30
        assert a.totalTokens == 45
        assert a.toolUsage == {"pubmed": 3, "other": 1}


class TestModelPairUsage:
    def test_status_dict(self):
        llm = LlmConfig(provider="nebius", model="moonshotai/Kimi-K2.6")
        emb = EmbdConfig(provider="nebius", model="Qwen/Qwen3-Embedding-8B")
        usage = ModelPairUsage(llm=llm, embedding=emb)
        usage.llmTier = "mixed"
        usage.embeddingTier = "fallback"
        d = usage.to_status_dict()
        assert d["llmTier"] == "mixed"
        assert d["embeddingTier"] == "fallback"
        assert d["llmProvider"] == "nebius"
        assert d["embeddingModel"] == "Qwen/Qwen3-Embedding-8B"

    def test_default_tiers_primary(self):
        llm = LlmConfig(provider="nebius", model="moonshotai/Kimi-K2.6")
        emb = EmbdConfig(provider="nebius", model="Qwen/Qwen3-Embedding-8B")
        usage = ModelPairUsage(llm=llm, embedding=emb)
        assert usage.llmTier == "primary"
        assert usage.embeddingTier == "primary"
