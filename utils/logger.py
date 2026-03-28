"""
utils/logger.py
───────────────
Centralized structured logging for DermaScan.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Prediction completed", extra={"class": "Melanoma", "confidence": 0.85})
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger with consistent formatting.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
               Defaults to INFO.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured (avoid duplicate handlers)

    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
