"""Ollama LLM client for local model inference."""

from __future__ import annotations

import logging

from langchain_ollama import OllamaLLM

from config.settings import Settings


LOGGER = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when Ollama server is unreachable."""

    pass


def get_llm(settings: Settings) -> OllamaLLM:
    """Create and return a configured Ollama LLM instance.

    Args:
        settings: Application settings with Ollama configuration.

    Returns:
        Configured OllamaLLM instance.

    Raises:
        OllamaConnectionError: If Ollama server is not running or unreachable.
    """
    try:
        llm = OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            top_p=settings.top_p,
            num_predict=settings.max_tokens,
        )
        # Test connectivity by invoking the model with a minimal prompt
        test_response = llm.invoke("test")
        if not test_response:
            raise OllamaConnectionError(
                f"Ollama server at {settings.ollama_base_url} did not respond properly."
            )
        LOGGER.info(
            "Ollama LLM initialized: model=%s, base_url=%s",
            settings.ollama_model,
            settings.ollama_base_url,
        )
        return llm
    except ConnectionError as e:
        raise OllamaConnectionError(
            f"Ollama server is not running at {settings.ollama_base_url}. "
            "Please start Ollama and ensure the model is available."
        ) from e
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str:
            raise OllamaConnectionError(
                f"Ollama server is not running at {settings.ollama_base_url}. "
                "Please start Ollama and ensure the model is available."
            ) from e
        if "model" in error_str or "not found" in error_str or "pull" in error_str:
            raise OllamaConnectionError(
                f"Model '{settings.ollama_model}' is not installed.\n"
                f"Run:\n"
                f"ollama pull {settings.ollama_model}"
            ) from e
        raise


def invoke_with_encoding(llm: OllamaLLM, prompt: str) -> str:
    """Invoke LLM with proper UTF-8 encoding handling.
    
    Args:
        llm: OllamaLLM instance.
        prompt: Prompt string.
        
    Returns:
        LLM response with proper encoding.
    """
    response = llm.invoke(prompt)
    # Ensure response is properly decoded as UTF-8
    if isinstance(response, bytes):
        response = response.decode('utf-8', errors='replace')
    return response
