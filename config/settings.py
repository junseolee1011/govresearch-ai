"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the application."""

    app_name: str
    log_level: str
    reports_directory: Path
    default_research_topic: str
    ollama_model: str
    ollama_base_url: str
    temperature: float
    top_p: float
    max_tokens: int


def get_settings() -> Settings:
    """Load settings from a local `.env` file and environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")
    reports_directory = Path(os.getenv("REPORTS_DIRECTORY", "reports"))
    if not reports_directory.is_absolute():
        reports_directory = PROJECT_ROOT / reports_directory
    return Settings(
        app_name=os.getenv("APP_NAME", "GovResearch-AI"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        reports_directory=reports_directory,
        default_research_topic=os.getenv(
            "DEFAULT_RESEARCH_TOPIC",
            "AI services used by public institutions",
        ),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
    )
