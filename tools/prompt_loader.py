"""Utilities for loading prompt templates from disk."""

from __future__ import annotations

from pathlib import Path


PROMPTS_DIRECTORY = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(prompt_name: str) -> str:
    """Load a named text prompt from the prompts directory.

    Args:
        prompt_name: Prompt filename, for example `planner.txt`.

    Returns:
        Prompt text without trailing whitespace.
    """
    return (PROMPTS_DIRECTORY / prompt_name).read_text(encoding="utf-8").strip()
