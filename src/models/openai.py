from typing import Any

from src.helpers.config import (
    OPENAI_MODEL_ID,
    OPENAI_RESPONSE_FORMAT,
    OPENAI_TEMPERATURE,
)
from src.models.distillation import OpenAIGeneration, TokenUsage
from src.models.language_model import ChatMessage


class OpenAIModel:
    provider = "openai"

    def __init__(
        self,
        client: Any,
        model_id: str = OPENAI_MODEL_ID,
        temperature: float | None = OPENAI_TEMPERATURE,
        response_format: dict[str, str] = OPENAI_RESPONSE_FORMAT,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.temperature = temperature
        self.response_format = response_format

    def generate(self, messages: list[ChatMessage]) -> str:
        text = self.generate_with_usage(messages).text
        if not text:
            raise RuntimeError("OpenAI returned an empty text response.")
        return text

    def generate_with_usage(
        self,
        messages: list[ChatMessage],
    ) -> OpenAIGeneration:
        request: dict[str, Any] = {
            "messages": messages,
            "model": self.model_id,
            "response_format": self.response_format,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature

        try:
            completion = self.client.chat.completions.create(**request)
        except Exception as exc:
            raise RuntimeError("OpenAI generation failed.") from exc

        choices = getattr(completion, "choices", None) or []
        content = (
            getattr(getattr(choices[0], "message", None), "content", None)
            if choices
            else None
        )
        text = (content or "").strip()

        raw_usage = getattr(completion, "usage", None)
        prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
        usage = TokenUsage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            cached_prompt_tokens=(
                getattr(prompt_details, "cached_tokens", 0) or 0
            ),
            completion_tokens=(
                getattr(raw_usage, "completion_tokens", 0) or 0
            ),
        )
        return OpenAIGeneration(text=text, usage=usage)
