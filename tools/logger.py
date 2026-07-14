"""Logging setup for GovResearch-AI."""

from __future__ import annotations

import logging


def configure_logging(log_level: str) -> None:
    """Configure the application root logger once.

    Args:
        log_level: Standard Python logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
