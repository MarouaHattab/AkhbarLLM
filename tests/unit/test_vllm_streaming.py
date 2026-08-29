from types import SimpleNamespace
from typing import Any

import pytest

from src.models.language_model import StreamingLanguageModel
from src.models.vllm import VLLMModel


class FakeCompletions:
    def __init__(self, events: object = None, error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.events


def event(content: object) -> SimpleNamespace:
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def runtime_with(events: object) -> tuple[VLLMModel, FakeCompletions]:
    completions = FakeCompletions(events)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return VLLMModel(client, model_id="news-lora"), completions


def test_vllm_is_a_streaming_language_model() -> None:
    runtime, _ = runtime_with([event("ok")])
    assert isinstance(runtime, StreamingLanguageModel)


def test_load_applies_explicit_generation_settings() -> None:
    captured: dict[str, Any] = {}

    def client_factory(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    model = VLLMModel.load(
        base_url="http://server/v1",
        api_key="secret",
        model_id="news",
        temperature=0.15,
        max_tokens=640,
        client_factory=client_factory,
    )

    assert model.temperature == 0.15
    assert model.max_tokens == 640
    assert model.model_id == "news"
    assert captured["base_url"] == "http://server/v1"
    assert captured["api_key"] == "secret"


def test_vllm_stream_yields_delta_content() -> None:
    runtime, completions = runtime_with(
        [event(None), event('{"translated'), event('_title": "خبر"}')]
    )

    assert list(runtime.stream([{"role": "user", "content": "x"}])) == [
        '{"translated',
        '_title": "خبر"}',
    ]
    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["model"] == "news-lora"


def test_vllm_stream_rejects_blank_output() -> None:
    runtime, _ = runtime_with([event(None), event("")])
    with pytest.raises(RuntimeError, match="blank streamed content"):
        list(runtime.stream([{"role": "user", "content": "x"}]))


def test_vllm_stream_rejects_malformed_delta() -> None:
    runtime, _ = runtime_with([SimpleNamespace(choices=[SimpleNamespace()])])
    with pytest.raises(RuntimeError, match="invalid streaming delta"):
        list(runtime.stream([{"role": "user", "content": "x"}]))


def test_vllm_stream_checks_complete_output_for_cjk() -> None:
    runtime, _ = runtime_with([event("safe"), event("中文")])
    generated = runtime.stream([{"role": "user", "content": "x"}])
    assert next(generated) == "safe"
    assert next(generated) == "中文"
    with pytest.raises(RuntimeError, match="contains CJK"):
        next(generated)
