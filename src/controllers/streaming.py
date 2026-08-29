import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal, cast

from src.helpers.environment import read_optional_setting
from src.models.factory import load_language_model
from src.models.language_model import (
    LanguageModel,
    StreamingLanguageModel,
)
from src.models.news import NewsDetails
from src.models.translation import TranslatedStory
from src.tasks import build_extraction_messages, build_translation_messages


TaskName = Literal["extraction", "translation"]
ValidatedTaskResponse = NewsDetails | TranslatedStory


@dataclass
class NewsStreamingController:
    runtime: StreamingLanguageModel
    _generation_lock: Lock = field(default_factory=Lock, repr=False)

    def stream_task(
        self,
        task: TaskName,
        story: str,
        *,
        source_language: str = "Arabic",
        target_language: str = "English",
    ) -> Iterator[str]:
        if task == "extraction":
            messages = build_extraction_messages(story)
        elif task == "translation":
            messages = build_translation_messages(
                story,
                source_language=source_language,
                target_language=target_language,
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        with self._generation_lock:
            yield from self.runtime.stream(messages)

    def validate_task_response(
        self,
        task: TaskName,
        raw_response: str,
    ) -> ValidatedTaskResponse:
        payload = json.loads(raw_response)
        if task == "extraction":
            return NewsDetails.model_validate(payload)
        if task == "translation":
            return TranslatedStory.model_validate(payload)
        raise ValueError(f"Unsupported task: {task}")


def build_streaming_controller(
    provider: str | None = None,
    *,
    runtime_loader: Callable[..., LanguageModel] = load_language_model,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
    vllm_model_id: str | None = None,
    vllm_temperature: float | None = None,
    vllm_max_tokens: int | None = None,
) -> NewsStreamingController:
    selected_provider = (
        provider
        or read_optional_setting("NEWS_MODEL_PROVIDER")
        or "finetuned"
    ).strip().casefold()
    runtime_options = {
        key: value
        for key, value in {
            "vllm_base_url": vllm_base_url,
            "vllm_api_key": vllm_api_key,
            "vllm_model_id": vllm_model_id,
            "vllm_temperature": vllm_temperature,
            "vllm_max_tokens": vllm_max_tokens,
        }.items()
        if value is not None
    }
    runtime = runtime_loader(selected_provider, **runtime_options)
    if not isinstance(runtime, StreamingLanguageModel):
        raise TypeError(
            f"Model provider {selected_provider!r} does not support streaming."
        )
    return NewsStreamingController(cast(StreamingLanguageModel, runtime))
