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
    tavily_api_key: str
    search_max_results: int
    search_topic: str
    embedding_model: str
    chroma_path: Path
    retrieval_k: int
    similarity_threshold: float
    min_retrieved_docs: int


def get_settings() -> Settings:
    """Load settings from a local `.env` file and environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")
    reports_directory = Path(os.getenv("REPORTS_DIRECTORY", "reports"))
    if not reports_directory.is_absolute():
        reports_directory = PROJECT_ROOT / reports_directory
    chroma_path = Path(os.getenv("CHROMA_PATH", "chroma"))
    if not chroma_path.is_absolute():
        chroma_path = PROJECT_ROOT / chroma_path
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
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        search_max_results=int(os.getenv("SEARCH_MAX_RESULTS", "5")),
        search_topic=os.getenv("SEARCH_TOPIC", "general"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chroma_path=chroma_path,
        retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.75")),
        min_retrieved_docs=int(os.getenv("MIN_RETRIEVED_DOCS", "3")),
    )
