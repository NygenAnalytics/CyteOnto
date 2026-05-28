import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_ai import (
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.usage import UsageLimits

load_dotenv()


class Config:
    SCHEMA_VERSION: str = "3.0"
    PRIMARY_LLM_PROVIDER: str = "nebius"
    PRIMARY_LLM_MODEL: str = "moonshotai/Kimi-K2.6"
    PRIMARY_EMBEDDING_PROVIDER: str = "nebius"
    PRIMARY_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-8B"

    FALLBACK_LLM_PROVIDER: str = "fireworks"
    FALLBACK_LLM_MODEL: str = "moonshotai/Kimi-K2.6"
    FALLBACK_EMBEDDING_PROVIDER: str = "openrouter"
    FALLBACK_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-8B"

    PROVIDER_API_KEY_ENV: dict[str, str] = {
        "nebius": "NEBIUS_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHER_API_KEY",
        "deepinfra": "DEEPINFRA_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_MODEL_API_KEY", "")
    NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")
    LOG_LEVEL: str = os.getenv("CYTEONTO_LOG_LEVEL", "DEBUG")
    LOG_FILE: str | None = os.getenv("CYTEONTO_LOG_FILE") or None

    MAX_PUBMED_CALLS = 3
    DEFAULT_USAGE_LIMITS = UsageLimits(request_limit=50, input_tokens_limit=60_000)

    RETRY_ATTEMPTS = 5
    RETRY_WAIT_MIN = 1.0
    RETRY_WAIT_MAX = 15.0
    PER_ATTEMPT_TIMEOUT = 120.0

    _RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
        ModelHTTPError,
        UnexpectedModelBehavior,
        ValidationError,
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

    OPENROUTER_DEEPINFRA_ROUTING: dict[str, Any] = {
        "provider": {"order": ["deepinfra"], "allow_fallbacks": True}
    }

    _PROVIDER_URL: dict[str, str] = {
        "deepinfra": "https://api.deepinfra.com/v1/openai/embeddings",
        "openrouter": "https://openrouter.ai/api/v1/embeddings",
        "ollama": "http://localhost:11434/api/embed",
        "openai": "https://api.openai.com/v1/embeddings",
        "google": "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent",
        "together": "https://api.together.xyz/v1/embeddings",
        "nebius": "https://api.tokenfactory.nebius.com/v1/embeddings",
    }

    RESULT_COLUMNS: list[str] = [
        "run_id",
        "algorithm",
        "pair_index",
        "author_label",
        "algorithm_label",
        "author_ontology_id",
        "author_embedding_similarity",
        "algorithm_ontology_id",
        "algorithm_embedding_similarity",
        "cytescore_similarity",
        "similarity_method",
    ]

    MAX_CONCURRENT_DESCRIPTIONS = 10
    MAX_CONCURRENT_EMBEDDINGS = 10
