from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, computed_field, model_validator
from pydantic_ai.usage import RunUsage

from .config import Config

EmbdProvider: TypeAlias = Literal[
    "deepinfra",
    "fireworks",
    "nebius",
    "ollama",
    "openai",
    "google",
    "openrouter",
    "together",
]

config = Config()

ModelPairTier: TypeAlias = Literal["primary", "fallback", "mixed"]

def _sanitize_filename_part(value: str) -> str:
    """Normalise a single path segment. Preserves case and dots."""
    return value.replace("/", "-").replace(":", "-").replace(" ", "_")


class ModelArtifactKey(BaseModel):
    """Canonical cache key for an LLM or embedding model configuration."""

    provider: str = Field(
        description="Inference provider slug, e.g. fireworks, together"
    )
    model: str = Field(description="Full model id, e.g. moonshotai/Kimi-K2.6")

    @model_validator(mode="after")
    def _normalize_provider(self) -> "ModelArtifactKey":
        self.provider = self.provider.strip().lower()
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def company(self) -> str:
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return self.model

    @computed_field  # type: ignore[prop-decorator]
    @property
    def modelName(self) -> str:
        if "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model

    def filename_segment(self) -> str:
        """``{provider}_{company}-{modelName}`` after filesystem sanitization."""
        company = _sanitize_filename_part(self.company)
        model_name = _sanitize_filename_part(self.modelName)
        return f"{self.provider}_{company}-{model_name}"

    @classmethod
    def from_provider_model(cls, provider: str, model: str) -> "ModelArtifactKey":
        return cls(provider=provider, model=model)


class LlmConfig(BaseModel):
    """LLM routing identity for description generation caches."""

    provider: str = Field(
        description="Provider slug, e.g. fireworks, together, openrouter"
    )
    model: str = Field(description="Full model id passed to the agent")

    def to_artifact_key(self) -> ModelArtifactKey:
        return ModelArtifactKey.from_provider_model(self.provider, self.model)


class CellDescription(BaseModel):
    """Structured description of a single cell type."""

    initialLabel: str = Field(description="The initial label for the cell")
    descriptiveName: str = Field(description="A detailed descriptive name")
    function: str = Field(description="The primary function of the cell")
    diseaseRelevance: str = Field(description="Disease relevance of the cell")
    developmentalStage: str = Field(description="Developmental stage of the cell")

    @classmethod
    def blank(cls, label: str = "") -> "CellDescription":
        return cls(
            initialLabel=label,
            descriptiveName="",
            function="",
            diseaseRelevance="",
            developmentalStage="",
        )

    def is_blank(self) -> bool:
        return (
            not self.descriptiveName
            and not self.function
            and not self.diseaseRelevance
            and not self.developmentalStage
        )

    def to_sentence(self) -> str:
        return (
            f"{self.initialLabel} is {self.descriptiveName}. "
            f"Function: {self.function} "
            f"Disease relevance: {self.diseaseRelevance} "
            f"Developmental stage: {self.developmentalStage}"
        )

    @classmethod
    def get_blank(cls) -> "CellDescription":
        return cls.blank()


class DescriptionFileEnvelope(BaseModel):
    """On-disk wrapper for a descriptions JSON file (schema v3)."""

    schemaVersion: str = Field(default=config.SCHEMA_VERSION)
    artifactKey: ModelArtifactKey
    updatedAt: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    descriptions: dict[str, CellDescription] = Field(default_factory=dict)


class EmbdConfig(BaseModel):
    """Parameters used to call an embedding provider.

    Defaults target OpenRouter with ``qwen/qwen3-embedding-8b`` routed primarily
    to DeepInfra. Pass ``modelSettings={}`` to disable the default routing, or
    override it with your own ``{"provider": {...}}`` block.
    """

    provider: EmbdProvider = Field(default="openrouter")
    model: str = Field(default="qwen/qwen3-embedding-8b")
    apiKey: str | None = Field(default=None)
    modelSettings: dict[str, Any] | None = Field(default=None)
    maxConcurrent: int = Field(default=config.MAX_CONCURRENT_EMBEDDINGS, ge=1)

    @model_validator(mode="after")
    def _apply_default_routing(self) -> "EmbdConfig":
        if self.provider == "openrouter" and self.modelSettings is None:
            self.modelSettings = dict(config.OPENROUTER_DEEPINFRA_ROUTING)
        return self

    def to_artifact_key(self) -> ModelArtifactKey:
        return ModelArtifactKey.from_provider_model(self.provider, self.model)


class ModelPairUsage(BaseModel):
    """Which model pair tiers were used during a run."""

    llmTier: ModelPairTier = "primary"
    embeddingTier: ModelPairTier = "primary"
    llm: LlmConfig
    embedding: EmbdConfig

    def to_status_dict(self) -> dict[str, str]:
        return {
            "llmTier": self.llmTier,
            "embeddingTier": self.embeddingTier,
            "llmProvider": self.llm.provider,
            "llmModel": self.llm.model,
            "embeddingProvider": self.embedding.provider,
            "embeddingModel": self.embedding.model,
        }


class AgentUsage(BaseModel):
    """Tally of LLM calls made through a given agent."""

    agentName: str = Field(default="")
    modelName: str = Field(default="")
    requests: int = Field(default=0)
    inputTokens: int = Field(default=0)
    outputTokens: int = Field(default=0)
    totalTokens: int = Field(default=0)
    toolUsage: dict[str, int] = Field(default_factory=dict)

    def record(
        self,
        model_name: str,
        run_usage: RunUsage,
        tool_counts: dict[str, int] | None = None,
    ) -> None:
        self.modelName = model_name
        self.requests += run_usage.requests
        self.inputTokens += run_usage.input_tokens or 0
        self.outputTokens += run_usage.output_tokens or 0
        self.totalTokens += run_usage.total_tokens or 0
        if tool_counts:
            for tool, n in tool_counts.items():
                self.toolUsage[tool] = self.toolUsage.get(tool, 0) + n

    def merge(self, other: "AgentUsage") -> None:
        self.requests += other.requests
        self.inputTokens += other.inputTokens
        self.outputTokens += other.outputTokens
        self.totalTokens += other.totalTokens
        for tool, n in other.toolUsage.items():
            self.toolUsage[tool] = self.toolUsage.get(tool, 0) + n
