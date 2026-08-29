from dataclasses import dataclass
from typing import Literal

from src.helpers.config import (
    VLLM_API_BASE_URL,
    VLLM_LOCAL_API_KEY,
    VLLM_MAX_TOKENS,
    VLLM_MODEL_ID,
    VLLM_TEMPERATURE,
)
from src.helpers.environment import read_optional_setting


ProviderName = Literal["vllm", "finetuned"]


@dataclass(frozen=True)
class InferenceSettings:
    provider: ProviderName
    base_url: str = ""
    api_key: str = ""
    model_id: str = ""
    temperature: float = VLLM_TEMPERATURE
    max_tokens: int = VLLM_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.provider == "vllm":
            if not self.base_url.strip():
                raise ValueError("vLLM API base URL cannot be blank.")
            if not self.model_id.strip():
                raise ValueError("vLLM model ID cannot be blank.")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0 and 1.")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError(
                "Maximum output tokens must be between 1 and 4096."
            )

    @classmethod
    def direct(cls) -> "InferenceSettings":
        return cls(provider="finetuned")


def load_default_settings() -> InferenceSettings:
    provider = (
        read_optional_setting("NEWS_MODEL_PROVIDER") or "finetuned"
    ).casefold()
    if provider != "vllm":
        return InferenceSettings.direct()
    return InferenceSettings(
        provider="vllm",
        base_url=(
            read_optional_setting("VLLM_API_BASE_URL")
            or VLLM_API_BASE_URL
        ),
        api_key=(
            read_optional_setting("VLLM_API_KEY")
            or VLLM_LOCAL_API_KEY
        ),
        model_id=(
            read_optional_setting("VLLM_MODEL_ID")
            or VLLM_MODEL_ID
        ),
    )
