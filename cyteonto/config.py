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
    TEXT_MODEL = os.getenv("TEXT_MODEL", "grok-3-mini")
    TEXT_MODEL_API_KEY = os.getenv("TEXT_MODEL_API_KEY", "")
    TEXT_MODEL_PROVIDER = os.getenv("TEXT_MODEL_PROVIDER", "xai")


CONFIG = Config()
