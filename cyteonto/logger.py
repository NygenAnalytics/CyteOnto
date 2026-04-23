import sys

from loguru import logger

from .config import Config

config = Config()
logger.remove()
logger.add(sys.stdout, format="{level}: {message}", level=config.LOG_LEVEL)

if config.LOG_FILE:
    try:
        logger.add(
            config.LOG_FILE,
            format="{time} {level} {message}",
            level=config.LOG_LEVEL,
            rotation="10 MB",
        )
    except Exception:
        pass

__all__ = ["logger"]
