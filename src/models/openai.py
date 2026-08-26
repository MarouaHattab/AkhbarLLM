from typing import Any

from src.helpers.config import (
    OPENAI_MODEL_ID,
    OPENAI_RESPONSE_FORMAT,
    OPENAI_TEMPERATURE,
)
from src.models.language_model import ChatMessage


class OpenAIModel:
    provider = "openai"

    def __init__(self, client: Any, model_id: str = OPENAI_MODEL_ID) -> None:
        self.client = client
        self.model_id = model_id

    def generate(self, messages: list[ChatMessage]) -> str:
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_id,
                temperature=OPENAI_TEMPERATURE,
                response_format=OPENAI_RESPONSE_FORMAT,
            )
        except Exception as exc:
            raise RuntimeError("OpenAI generation failed.") from exc

        choices = getattr(completion, "choices", None) or []
        content = (
            getattr(getattr(choices[0], "message", None), "content", None)
            if choices
            else None
        )
        text = (content or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty text response.")
        return text
