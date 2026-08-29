from collections.abc import Callable
from typing import Any

from src.helpers.config import (
    VLLM_API_BASE_URL,
    VLLM_LOCAL_API_KEY,
    VLLM_MAX_TOKENS,
    VLLM_MODEL_ID,
    VLLM_REQUEST_TIMEOUT_SECONDS,
    VLLM_TEMPERATURE,
)
from src.models.language_model import ChatMessage
from src.utils.text import contains_chinese_characters


class VLLMModel:
    provider = "vllm"

    def __init__(
        self,
        client: Any,
        model_id: str = VLLM_MODEL_ID,
        max_tokens: int = VLLM_MAX_TOKENS,
        temperature: float = VLLM_TEMPERATURE,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def load(
        cls,
        *,
        base_url: str = VLLM_API_BASE_URL,
        api_key: str = VLLM_LOCAL_API_KEY,
        timeout: float = VLLM_REQUEST_TIMEOUT_SECONDS,
        client_factory: Callable[..., Any] | None = None,
    ) -> "VLLMModel":
        if client_factory is None:
            from openai import OpenAI

            client_factory = OpenAI
        client = client_factory(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        return cls(client)

    def generate(self, messages: list[ChatMessage]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            raise RuntimeError(
                "vLLM generation failed; check the local server and model readiness."
            ) from exc

        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError(
                "vLLM returned no choices in its empty completion response."
            )

        try:
            message = getattr(choices[0], "message", None)
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "vLLM returned an invalid completion response with no message."
            ) from exc
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "vLLM returned missing or blank message content."
            )

        if contains_chinese_characters(content):
            raise RuntimeError(
                "vLLM mandatory suppression is broken: generated content contains CJK characters."
            )
        return content
