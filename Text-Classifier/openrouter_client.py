"""OpenRouter API client for Text Classifier application.

Handles communication with the OpenRouter chat completions endpoint with
comprehensive error handling, timeout management, and credential protection.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

# Search for .env in current folder, project folder, and parent directory
env_paths = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.cwd() / ".env",
]
for p in env_paths:
    if p.exists():
        load_dotenv(dotenv_path=p)
        break
else:
    load_dotenv()

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Predefined curated model choices for easy selection
POPULAR_MODELS: List[Dict[str, str]] = [
    {"name": "GPT-4o Mini (OpenAI)", "id": "openai/gpt-4o-mini"},
    {"name": "Claude 3.5 Haiku (Anthropic)", "id": "anthropic/claude-3.5-haiku"},
    {"name": "Gemini 2.0 Flash (Google)", "id": "google/gemini-2.0-flash-001"},
    {"name": "Llama 3.3 70B Instruct (Meta)", "id": "meta-llama/llama-3.3-70b-instruct"},
    {"name": "DeepSeek V3 (DeepSeek)", "id": "deepseek/deepseek-chat"},
    {"name": "Mistral Small 24B (Mistral)", "id": "mistralai/mistral-small-24b-instruct-2501"},
]


class OpenRouterError(Exception):
    """Base exception class for OpenRouter API interactions."""


class AuthenticationError(OpenRouterError):
    """Raised when the API key is missing, invalid, or unauthorized."""


class RateLimitError(OpenRouterError):
    """Raised when the rate limit or quota has been exceeded."""


class ModelUnavailableError(OpenRouterError):
    """Raised when the requested model is invalid or unavailable."""


class NetworkError(OpenRouterError):
    """Raised for network connectivity issues or timeouts."""


def get_api_key() -> Optional[str]:
    """Retrieve the OpenRouter API key from the environment, reloading if necessary."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        # Reload in case .env was added while the app was running
        for p in env_paths:
            if p.exists():
                load_dotenv(dotenv_path=p, override=True)
                break
        key = os.getenv("OPENROUTER_API_KEY")

    if key:
        key = key.strip().strip('"').strip("'")
    return key if key else None


def is_api_key_configured() -> bool:
    """Check if a valid non-empty API key is present."""
    return get_api_key() is not None


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 500,
    timeout: int = 30,
) -> str:
    """Send a chat completion request to OpenRouter and return the text content.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        model: OpenRouter model identifier string.
        temperature: Sampling temperature (0.0 for deterministic classification).
        max_tokens: Maximum tokens in response.
        timeout: Network timeout in seconds.

    Returns:
        The text content of the assistant message.

    Raises:
        AuthenticationError: If API key is missing or invalid.
        RateLimitError: If rate limit is hit.
        ModelUnavailableError: If model is not found or unsupported.
        NetworkError: If connection fails or times out.
        OpenRouterError: For other API failures.
    """
    api_key = get_api_key()
    if not api_key:
        raise AuthenticationError(
            "OpenRouter API key is not configured. "
            "Please set OPENROUTER_API_KEY in your .env file or environment variables."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/project-03-text-classifier",
        "X-Title": "Text Classifier Learning Project",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            OPENROUTER_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise NetworkError("API request timed out. Please try again.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise NetworkError(
            "Unable to connect to OpenRouter. Please check your internet connection."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise NetworkError(f"Network error while communicating with OpenRouter: {exc}") from exc

    if response.status_code == 401:
        raise AuthenticationError(
            "OpenRouter authentication failed (HTTP 401). "
            "Please check that your API key is valid and has active credits."
        )
    if response.status_code == 429:
        raise RateLimitError(
            "OpenRouter rate limit or quota exceeded (HTTP 429). "
            "Please wait a moment or verify your account balance."
        )
    if response.status_code == 404:
        raise ModelUnavailableError(
            f"The model '{model}' is not found or currently unavailable (HTTP 404). "
            "Please select a different model."
        )
    if response.status_code == 400:
        error_msg = "Bad request (HTTP 400)."
        try:
            err_data = response.json()
            if "error" in err_data and "message" in err_data["error"]:
                error_msg = f"API error: {err_data['error']['message']}"
        except Exception:
            pass
        raise ModelUnavailableError(error_msg)

    if not response.ok:
        raise OpenRouterError(
            f"OpenRouter returned status {response.status_code}. "
            "Classification could not be completed."
        )

    try:
        data = response.json()
    except Exception as exc:
        raise OpenRouterError("Received an invalid JSON response from the API.") from exc

    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        raise OpenRouterError("API response contained no completion choices.")

    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content")

    if not content or not content.strip():
        raise OpenRouterError("Model returned an empty response.")

    return content.strip()
