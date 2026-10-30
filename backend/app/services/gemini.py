"""Small wrapper around the Google Gen AI SDK."""

from google import genai
from google.genai import types

from app import config


MODEL_NAME = "gemini-flash-latest"
MILLISECONDS_PER_SECOND = 1_000


class GeminiServiceError(Exception):
    """Raised when Gemini cannot provide a usable answer."""


class GeminiService:
    """Ask the Gemini API text-only questions."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.request_timeout = (
            request_timeout
            if request_timeout is not None
            else config.get_gemini_request_timeout()
        )
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(self.request_timeout * MILLISECONDS_PER_SECOND)
            ),
        )
        self.model = model if model is not None else config.get_gemini_model()
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else config.get_gemini_max_output_tokens()
        )
        self.thinking_budget = (
            thinking_budget
            if thinking_budget is not None
            else config.get_gemini_thinking_budget()
        )

    def ask(self, question: str) -> str:
        """Generate an answer without exposing SDK errors to API clients."""
        try:
            request_arguments: dict[str, object] = {
                "model": self.model,
                "contents": question,
            }
            generation_config = self._build_generation_config()
            if generation_config is not None:
                request_arguments["config"] = generation_config
            response = self.client.models.generate_content(**request_arguments)
            answer = response.text
        except Exception as error:
            raise GeminiServiceError("Gemini request failed") from error

        if not answer:
            raise GeminiServiceError("Gemini returned an empty response")

        return answer

    def generate_json(self, prompt: str) -> str:
        """Generate text that the Gemini API labels as JSON."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._build_generation_config(
                    response_mime_type="application/json"
                ),
            )
            answer = response.text
        except Exception as error:
            raise GeminiServiceError("Gemini request failed") from error

        if not answer:
            raise GeminiServiceError("Gemini returned an empty response")

        return answer

    def close(self) -> None:
        """Close the SDK client and release its HTTP resources."""
        self.client.close()

    def _build_generation_config(
        self,
        response_mime_type: str | None = None,
    ) -> types.GenerateContentConfig | None:
        """Build settings supported by the configured model and installed SDK."""
        settings: dict[str, object] = {}
        if response_mime_type is not None:
            settings["response_mime_type"] = response_mime_type
        if self.max_output_tokens is not None:
            settings["max_output_tokens"] = self.max_output_tokens
        if self._supports_thinking_budget() and self.thinking_budget is not None:
            settings["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        return types.GenerateContentConfig(**settings) if settings else None

    def _supports_thinking_budget(self) -> bool:
        """Return whether this explicit model supports the SDK's budget setting."""
        return self.model.startswith("gemini-2.5-flash")
