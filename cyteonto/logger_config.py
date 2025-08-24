# cyteonto/logger_config.py

import sys

import logfire
from loguru import logger

from .config import CONFIG

logger.remove()
logger.add(sys.stdout, format="{level}: {message}", level="DEBUG")

try:
    logfire.configure(
        token=CONFIG.LOGFIRE_API_KEY, scrubbing=False, service_name="CyteOnto"
    )
    logger.configure(handlers=[logfire.loguru_handler()])
except Exception:
    pass
