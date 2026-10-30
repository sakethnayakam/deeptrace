"""Application configuration loaded from the backend environment file."""

import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_TAVILY_SEARCH_TIMEOUT = 30.0
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_GEMINI_REQUEST_TIMEOUT = 60.0
load_dotenv(BACKEND_DIRECTORY / ".env")


def get_gemini_api_key() -> str | None:
    """Return the configured Gemini API key, if one is available."""
    api_key = os.getenv("GEMINI_API_KEY")
    return api_key.strip() if api_key else None


def get_gemini_model() -> str:
    """Return the configured Gemini model, preserving the existing default."""
    model = os.getenv("GEMINI_MODEL")
    return model.strip() if model and model.strip() else DEFAULT_GEMINI_MODEL


def get_gemini_request_timeout() -> float:
    """Return the Gemini request timeout in seconds, with a safe default."""
    value = os.getenv("GEMINI_REQUEST_TIMEOUT")
    if not value:
        return DEFAULT_GEMINI_REQUEST_TIMEOUT

    try:
        timeout = float(value)
    except ValueError:
        return DEFAULT_GEMINI_REQUEST_TIMEOUT

    return timeout if timeout > 0 else DEFAULT_GEMINI_REQUEST_TIMEOUT


def get_gemini_max_output_tokens() -> int | None:
    """Return a positive configured Gemini output-token limit, if supplied."""
    value = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
    if not value:
        return None

    try:
        max_output_tokens = int(value)
    except ValueError:
        return None

    return max_output_tokens if max_output_tokens > 0 else None


def get_gemini_thinking_budget() -> int | None:
    """Return a configured Gemini 2.5 Flash thinking budget, if supplied."""
    value = os.getenv("GEMINI_THINKING_BUDGET")
    if not value:
        return None

    try:
        thinking_budget = int(value)
    except ValueError:
        return None

    return thinking_budget if thinking_budget >= -1 else None


def get_tavily_api_key() -> str | None:
    """Return the configured Tavily API key, if one is available."""
    api_key = os.getenv("TAVILY_API_KEY")
    return api_key.strip() if api_key else None


def get_tavily_search_timeout() -> float:
    """Return the configured Tavily timeout in seconds, with a safe default."""
    value = os.getenv("TAVILY_SEARCH_TIMEOUT")
    if not value:
        return DEFAULT_TAVILY_SEARCH_TIMEOUT

    try:
        timeout = float(value)
    except ValueError:
        return DEFAULT_TAVILY_SEARCH_TIMEOUT

    return timeout if timeout > 0 else DEFAULT_TAVILY_SEARCH_TIMEOUT
