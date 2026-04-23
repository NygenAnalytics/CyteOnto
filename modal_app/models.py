"""Request and response models for the CyteOnto HTTP API."""

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from .config import AppConfig

app_config = AppConfig()

LlmProvider: TypeAlias = Literal["openrouter", "together", "openai"]
EmbdProvider: TypeAlias = Literal[
    "deepinfra", "ollama", "openai", "google", "openrouter", "together"
]
RunState: TypeAlias = Literal["queued", "running", "completed", "failed"]


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authorLabels: list[str]
    algorithms: dict[str, list[str]]

    llmProvider: LlmProvider = app_config.DEFAULT_LLM_PROVIDER
    llmModel: str = app_config.DEFAULT_LLM_MODEL
    llmApiKey: str

    embeddingProvider: EmbdProvider = app_config.DEFAULT_EMBEDDING_PROVIDER
    embeddingModel: str = app_config.DEFAULT_EMBEDDING_MODEL
    embeddingApiKey: str
    embeddingModelSettings: dict[str, Any] | None = None
    embeddingMaxConcurrent: int = Field(default=100, ge=1)

    maxDescriptionConcurrency: int = Field(default=100, ge=1)
    usePubmedTool: bool = False
    reasoning: bool = False

    metric: str = "cosine_kernel"
    metricParams: dict[str, Any] | None = None
    minMatchSimilarity: float = Field(default=0.1, ge=0.0, le=1.0)
    useCache: bool = True


class CompareResponse(BaseModel):
    runId: str
    state: RunState


class StatusResponse(BaseModel):
    runId: str
    state: RunState
    createdAt: str
    startedAt: str | None = None
    completedAt: str | None = None
    error: str | None = None
    numAuthorLabels: int
    numAlgorithms: int
    numRows: int | None = None
    resultCsvPath: str | None = None
    resultJsonPath: str | None = None
