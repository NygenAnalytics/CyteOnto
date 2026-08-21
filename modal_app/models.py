"""Request and response models for the CyteOnto HTTP API."""

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from .config import AppConfig

app_config = AppConfig()

LlmProvider: TypeAlias = Literal[
    "openrouter", "together", "openai", "nebius", "fireworks"
]
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
ModelPairTier: TypeAlias = Literal["primary", "fallback", "mixed"]
RunState: TypeAlias = Literal["queued", "running", "completed", "failed"]
SimilarityMethod: TypeAlias = Literal[
    "cytescore",
    "cytescore_compound",
    "string_similarity",
    "partial_match",
    "no_matches",
    "empty",
]


class CompareRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "authorLabels": [
                        "alveolar macrophage",
                        "regulatory T cell",
                    ],
                    "algorithms": {
                        "methodA": [
                            "lung macrophage",
                            "Treg",
                        ]
                    },
                }
            ]
        },
    )

    authorLabels: list[str] = Field(
        description="Reference cell labels, with one label for each comparison row."
    )
    algorithms: dict[str, list[str]] = Field(
        description=(
            "Algorithm names mapped to predicted cell labels. Each list must contain "
            "the same number of labels as authorLabels."
        )
    )

    llmProvider: LlmProvider = Field(
        default=app_config.DEFAULT_LLM_PROVIDER,
        description="Provider used to generate descriptions for cell labels.",
    )
    llmModel: str = Field(
        default=app_config.DEFAULT_LLM_MODEL,
        description="Model identifier accepted by the selected LLM provider.",
    )
    llmApiKey: str | None = Field(
        default=None,
        description=(
            "Optional provider API key. It is required when the service does not "
            "have a key configured for the selected LLM provider."
        ),
        json_schema_extra={"writeOnly": True},
    )

    embeddingProvider: EmbdProvider = Field(
        default=app_config.DEFAULT_EMBEDDING_PROVIDER,
        description="Provider used to create label embeddings.",
    )
    embeddingModel: str = Field(
        default=app_config.DEFAULT_EMBEDDING_MODEL,
        description="Model identifier accepted by the selected embedding provider.",
    )
    embeddingApiKey: str | None = Field(
        default=None,
        description=(
            "Optional provider API key. It is required when the service does not "
            "have a key configured for the selected embedding provider."
        ),
        json_schema_extra={"writeOnly": True},
    )
    embeddingModelSettings: dict[str, Any] | None = Field(
        default=None,
        description="Optional provider-specific settings sent with embedding requests.",
    )
    embeddingMaxConcurrent: int = Field(
        default=100,
        ge=1,
        description="Maximum number of concurrent embedding requests.",
    )

    maxDescriptionConcurrency: int = Field(
        default=100,
        ge=1,
        description="Maximum number of concurrent description-generation requests.",
    )
    usePubmedTool: bool = Field(
        default=False,
        description="Allow PubMed abstract searches while descriptions are generated.",
    )
    reasoning: bool = Field(
        default=False,
        description="Request provider reasoning when the selected model supports it.",
    )

    metric: str = Field(
        default="cosine_kernel",
        description="Similarity metric used to score matched ontology terms.",
    )
    metricParams: dict[str, Any] | None = Field(
        default=None,
        description="Optional parameters passed to the selected similarity metric.",
    )
    minMatchSimilarity: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum embedding similarity required for an ontology match.",
    )
    compoundScoring: Literal["max", "hungarian_mean"] = Field(
        default="max",
        description="Method used to combine scores for compound cell labels.",
    )
    useCache: bool = Field(
        default=True,
        description="Reuse cached descriptions and embeddings when available.",
    )


class CompareResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "runId": "run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3",
                    "state": "queued",
                }
            ]
        }
    )

    runId: str = Field(description="Identifier used to poll status and fetch results.")
    state: RunState = Field(description="Initial state of the submitted job.")


class ModelPairUsageResponse(BaseModel):
    llmTier: ModelPairTier = Field(
        description="Whether the primary, fallback, or both LLM configurations ran."
    )
    embeddingTier: ModelPairTier = Field(
        description=(
            "Whether the primary, fallback, or both embedding configurations ran."
        )
    )
    llmProvider: str = Field(description="LLM provider used by the completed job.")
    llmModel: str = Field(description="LLM model used by the completed job.")
    embeddingProvider: str = Field(
        description="Embedding provider used by the completed job."
    )
    embeddingModel: str = Field(
        description="Embedding model used by the completed job."
    )


class StatusResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "runId": "run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3",
                    "state": "completed",
                    "createdAt": "2026-04-22T22:11:04.501Z",
                    "startedAt": "2026-04-22T22:11:05.812Z",
                    "completedAt": "2026-04-22T22:12:48.133Z",
                    "error": None,
                    "numAuthorLabels": 2,
                    "numAlgorithms": 1,
                    "numRows": 2,
                    "resultCsvPath": (
                        "user_files/run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3/result.csv"
                    ),
                    "resultJsonPath": (
                        "user_files/run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3/"
                        "result.json"
                    ),
                }
            ]
        }
    )

    runId: str = Field(description="Identifier returned when the job was submitted.")
    state: RunState = Field(description="Current lifecycle state of the job.")
    createdAt: str = Field(
        description="UTC ISO 8601 timestamp recorded when the job was accepted."
    )
    startedAt: str | None = Field(
        default=None,
        description="UTC ISO 8601 timestamp recorded when processing began.",
    )
    completedAt: str | None = Field(
        default=None,
        description="UTC ISO 8601 timestamp recorded when the job reached a final state.",
    )
    error: str | None = Field(
        default=None,
        description="Failure type and message when state is failed.",
    )
    numAuthorLabels: int = Field(
        description="Number of reference labels in the request."
    )
    numAlgorithms: int = Field(description="Number of algorithms in the request.")
    numRows: int | None = Field(
        default=None,
        description="Number of result rows after the job completes.",
    )
    resultCsvPath: str | None = Field(
        default=None,
        description="Internal result path populated after the job completes.",
    )
    resultJsonPath: str | None = Field(
        default=None,
        description="Internal result path populated after the job completes.",
    )
    modelPairUsage: ModelPairUsageResponse | None = Field(
        default=None,
        description="Primary and fallback model usage for a completed job.",
    )


class HealthResponse(BaseModel):
    ok: bool = Field(description="Whether the API process is available.")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable explanation of the error.")


class ResultRow(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "run_id": "run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3",
                    "algorithm": "methodA",
                    "pair_index": 0,
                    "author_label": "alveolar macrophage",
                    "algorithm_label": "lung macrophage",
                    "author_ontology_id": "CL:0000583",
                    "author_ontology_name": "alveolar macrophage",
                    "author_embedding_similarity": 0.94,
                    "algorithm_ontology_id": "CL:0000583",
                    "algorithm_ontology_name": "alveolar macrophage",
                    "algorithm_embedding_similarity": 0.91,
                    "cytescore_similarity": 1.0,
                    "similarity_method": "cytescore",
                }
            ]
        },
    )

    runId: str = Field(
        alias="run_id",
        description="Identifier of the job that produced this row.",
    )
    algorithm: str = Field(description="Algorithm name supplied in the request.")
    pairIndex: int = Field(
        alias="pair_index",
        description="Zero-based position of the label pair in the request.",
    )
    authorLabel: str = Field(
        alias="author_label",
        description="Reference cell label for this comparison.",
    )
    algorithmLabel: str = Field(
        alias="algorithm_label",
        description="Algorithm-provided cell label for this comparison.",
    )
    authorOntologyId: str = Field(
        alias="author_ontology_id",
        description="Matched Cell Ontology identifiers for the reference label.",
    )
    authorOntologyName: str = Field(
        alias="author_ontology_name",
        description="Names of the matched reference Cell Ontology terms.",
    )
    authorEmbeddingSimilarity: float | str = Field(
        alias="author_embedding_similarity",
        description="Embedding similarity for each reference label part.",
    )
    algorithmOntologyId: str = Field(
        alias="algorithm_ontology_id",
        description="Matched Cell Ontology identifiers for the algorithm label.",
    )
    algorithmOntologyName: str = Field(
        alias="algorithm_ontology_name",
        description="Names of the matched algorithm Cell Ontology terms.",
    )
    algorithmEmbeddingSimilarity: float | str = Field(
        alias="algorithm_embedding_similarity",
        description="Embedding similarity for each algorithm label part.",
    )
    cytescoreSimilarity: float = Field(
        alias="cytescore_similarity",
        description="Similarity score produced by the selected metric.",
    )
    similarityMethod: SimilarityMethod = Field(
        alias="similarity_method",
        description="Method used to produce the similarity score.",
    )
