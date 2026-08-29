import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import pytest
from pydantic import ValidationError

from src.controllers.streaming import (
    NewsStreamingController,
    build_streaming_controller,
)
from src.models.news import NewsDetails
from src.models.translation import TranslatedStory


class FakeRuntime:
    provider = "fake"
    model_id = "fake-news"

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        return "".join(self.chunks)

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.messages = messages
        yield from self.chunks


EXTRACTION = {
    "story_title": "عنوان إخباري واضح",
    "story_keywords": ["خبر"],
    "story_summary": ["ملخص الخبر"],
    "story_category": "politics",
    "story_entities": [
        {"entity_value": "تونس", "entity_type": "location"}
    ],
}

TRANSLATION = {
    "translated_title": "A complete translated title",
    "translated_content": "A complete translated news story.",
}


def test_extraction_stream_reuses_existing_prompt_builder() -> None:
    runtime = FakeRuntime([json.dumps(EXTRACTION, ensure_ascii=False)])
    controller = NewsStreamingController(runtime)

    chunks = list(controller.stream_task("extraction", "قصة إخبارية"))

    assert chunks
    assert runtime.messages is not None
    assert runtime.messages[0]["role"] == "system"
    assert "## Output Schema:" in runtime.messages[1]["content"]
    assert "قصة إخبارية" in runtime.messages[1]["content"]


def test_translation_stream_passes_languages_to_existing_builder() -> None:
    runtime = FakeRuntime([json.dumps(TRANSLATION)])
    controller = NewsStreamingController(runtime)

    list(
        controller.stream_task(
            "translation",
            "قصة إخبارية",
            source_language="Arabic",
            target_language="French",
        )
    )

    assert runtime.messages is not None
    assert "Arabic" in runtime.messages[0]["content"]
    assert "## Target Language:\nFrench" in runtime.messages[1]["content"]


def test_validation_returns_the_selected_schema() -> None:
    controller = NewsStreamingController(FakeRuntime([]))
    extraction = controller.validate_task_response(
        "extraction", json.dumps(EXTRACTION, ensure_ascii=False)
    )
    translation = controller.validate_task_response(
        "translation", json.dumps(TRANSLATION)
    )
    assert isinstance(extraction, NewsDetails)
    assert isinstance(translation, TranslatedStory)


def test_validation_preserves_json_and_schema_errors() -> None:
    controller = NewsStreamingController(FakeRuntime([]))
    with pytest.raises(json.JSONDecodeError):
        controller.validate_task_response("extraction", "not json")
    with pytest.raises(ValidationError):
        controller.validate_task_response("translation", "{}")


def test_controller_rejects_unknown_task() -> None:
    controller = NewsStreamingController(FakeRuntime([]))
    with pytest.raises(ValueError, match="Unsupported task"):
        list(controller.stream_task("summarization", "story"))  # type: ignore[arg-type]


def test_controller_serializes_access_to_one_runtime() -> None:
    class ConcurrencyTrackingRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__(["ok"])
            self.active = 0
            self.maximum_active = 0
            self.guard = Lock()

        def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
            with self.guard:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            sleep(0.03)
            yield "ok"
            with self.guard:
                self.active -= 1

    runtime = ConcurrencyTrackingRuntime()
    controller = NewsStreamingController(runtime)
    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(
            executor.map(
                lambda _: list(controller.stream_task("extraction", "story")),
                range(2),
            )
        )

    assert outputs == [["ok"], ["ok"]]
    assert runtime.maximum_active == 1


def test_builder_defaults_to_finetuned_and_requires_streaming_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEWS_MODEL_PROVIDER", raising=False)
    requested: list[str] = []

    def loader(provider: str) -> object:
        requested.append(provider)
        return FakeRuntime(["ok"])

    controller = build_streaming_controller(runtime_loader=loader)
    assert controller.runtime.provider == "fake"
    assert requested == ["finetuned"]


def test_builder_forwards_vllm_runtime_overrides() -> None:
    captured: dict[str, object] = {}

    def loader(provider: str, **kwargs: object) -> object:
        captured.update(provider=provider, **kwargs)
        return FakeRuntime(["ok"])

    controller = build_streaming_controller(
        "vllm",
        vllm_base_url="http://server/v1",
        vllm_api_key="secret",
        vllm_model_id="news",
        vllm_temperature=0.15,
        vllm_max_tokens=640,
        runtime_loader=loader,
    )

    assert controller.runtime.provider == "fake"
    assert captured == {
        "provider": "vllm",
        "vllm_base_url": "http://server/v1",
        "vllm_api_key": "secret",
        "vllm_model_id": "news",
        "vllm_temperature": 0.15,
        "vllm_max_tokens": 640,
    }
