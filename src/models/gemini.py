from collections.abc import Callable
from typing import Any

from src.helpers.config import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL_ID,
    GEMINI_TEMPERATURE,
)
from src.models.language_model import ChatMessage


def build_gemini_config(**settings: Any) -> Any:
    """Build SDK configuration lazily so imports do not require the SDK."""
    from google.genai import types

    return types.GenerateContentConfig(**settings)


class GeminiModel:
    provider = "gemini"

    def __init__(
        self,
        client: Any,
        model_id: str = GEMINI_MODEL_ID,
        config_builder: Callable[..., Any] = build_gemini_config,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.config_builder = config_builder

    def generate(self, messages: list[ChatMessage]) -> str:
        system_instruction = "\n\n".join(
            message["content"]
            for message in messages
            if message["role"] == "system"
        ) or None
        contents = "\n\n".join(
            message["content"]
            for message in messages
            if message["role"] != "system"
        )
        config = self.config_builder(
            system_instruction=system_instruction,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            temperature=GEMINI_TEMPERATURE,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise RuntimeError("Gemini generation failed.") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty text response.")
        return text
