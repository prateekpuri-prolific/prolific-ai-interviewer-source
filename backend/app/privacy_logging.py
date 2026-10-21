"""Conservative logging defaults for research data paths.

This is not a substitute for sink/provider inspection. Do not attach request
bodies or participant text to logs at any level.
"""
import logging
import sys
from loguru import logger


def configure_privacy_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", diagnose=False, backtrace=False)
    # Provider debug logs can include prompts/transcriptions. Suppress their
    # routine logging; application health reports errors without payloads.
    for name in ("openai", "httpx", "httpcore", "sqlalchemy.engine"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logger.disable("pipecat")
