"""Utilities for execution timing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context manager for measuring execution time.

    Args:
        label: Description of the operation being timed.

    Yields:
        None. On context exit, logs the elapsed time.
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        # Timer result can be accessed via logging
        timer.last_elapsed = elapsed  # type: ignore[attr-defined]


def get_elapsed() -> float:
    """Get the last recorded elapsed time.

    Returns:
        Seconds elapsed in the most recent timer context.
    """
    return getattr(timer, "last_elapsed", 0.0)
