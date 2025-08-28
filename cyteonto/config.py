# cyteonto/config.py

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    API keys configuration.
    """

    LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
    NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # Recommended for higher rate limits
    EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY", "")


CONFIG = Config()
