"""Enhanced logging setup for GovResearch-AI with execution metrics."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Generator


def configure_logging(log_level: str) -> None:
    """Configure the application root logger once.

    Args:
        log_level: Standard Python logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@contextmanager
def log_node_execution(node_name: str) -> Generator[dict[str, object], None, None]:
    """Context manager for logging node execution with timing and metrics.

    Args:
        node_name: Name of the workflow node being executed.

    Yields:
        Dictionary to store execution metrics.
    """
    logger = logging.getLogger(__name__)
    metrics: dict[str, object] = {}
    start_time = perf_counter()

    logger.info("Node '%s' started execution.", node_name)

    try:
        yield metrics
    finally:
        elapsed = perf_counter() - start_time
        metrics["elapsed_seconds"] = round(elapsed, 3)

        response_length = metrics.get("response_length", 0)
        logger.info(
            "Node '%s' completed in %.3f seconds (response length: %d chars).",
            node_name,
            elapsed,
            response_length,
        )
