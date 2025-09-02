# cyteonto/config.py

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    API keys configuration.
    """

    NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
    EMBEDDING_MODEL_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY", "")
    LOGGING_LEVEL = "INFO"
    LOG_FILE = None


CONFIG = Config()
