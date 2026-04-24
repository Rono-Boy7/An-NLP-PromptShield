"""
src/promptshield/utils/logging.py

Purpose
-------
Centralized logging setup for PromptShield.

Why this file matters
---------------------
ML/security projects need clear logs because many things can fail quietly:

- dataset paths
- config loading
- model training
- scanner decisions
- API requests
- evaluation runs

Instead of using scattered print statements, this module gives the project one
consistent logging setup.

What this module does
---------------------
1. Configure Loguru with a clean terminal format
2. Support log levels from config or environment variables
3. Provide a reusable logger across the project
4. Avoid duplicate log handlers during repeated script/API runs

Design scope
------------
This file only configures logging.

It does not decide what gets logged by scanners, models, or API routes.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from loguru import logger

DEFAULT_LOG_LEVEL = "INFO"

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging(log_level: str | None = None) -> None:
    """Configure project logging with a single terminal sink."""

    level = normalize_log_level(
        log_level or os.getenv("PROMPTSHIELD_LOG_LEVEL") or DEFAULT_LOG_LEVEL
    )

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )


def normalize_log_level(log_level: str) -> str:
    """Normalize a log level string and validate common levels."""

    normalized = log_level.strip().upper()

    valid_levels = {
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if normalized not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Expected one of: {sorted(valid_levels)}")

    return normalized


def get_logger(**context: Any):
    """Return a logger optionally bound with structured context."""

    if context:
        return logger.bind(**context)

    return logger


__all__ = [
    "configure_logging",
    "get_logger",
    "normalize_log_level",
]
