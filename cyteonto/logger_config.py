# cyteonto/logger_config.py

import sys

from loguru import logger

from .config import CONFIG

logger.remove()
logger.add(sys.stdout, format="{level}: {message}", level=CONFIG.LOGGING_LEVEL)

try:
    if CONFIG.LOG_FILE:
        logger.add(
            CONFIG.LOG_FILE,
            format="{time} {level} {message}",
            level=CONFIG.LOGGING_LEVEL,
            rotation="10 MB",
        )
except Exception:
    pass
