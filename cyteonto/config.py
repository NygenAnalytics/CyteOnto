# cyteonto/config.py

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
    NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # Recommended for higher rate limits

    # Embedding Config
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY", "")
    EMBEDDING_MODEL_PROVIDER = os.getenv("EMBEDDING_MODEL_PROVIDER", "deepinfra")

    # LLM Config
    BASE_AGENT = os.getenv("BASE_AGENT", "grok-3-mini")
    BASE_AGENT_API_KEY = os.getenv("BASE_AGENT_API_KEY", "")
    BASE_AGENT_PROVIDER = os.getenv("BASE_AGENT_PROVIDER", "xai")


CONFIG = Config()
