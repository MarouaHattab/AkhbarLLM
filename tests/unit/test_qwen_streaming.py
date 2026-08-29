from queue import Queue
from typing import Any

import pytest

from src.helpers.config import qwen_generation_kwargs
from src.models.language_model import StreamingLanguageModel
from src.models.qwen import QwenModel


class FakeInputs(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=[[1, 2, 3]])
        self.input_ids = [[1, 2, 3]]
        self.target_device: str | None = None

    def to(self, device: str) -> "FakeInputs":
        self.target_device = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "</s>"

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "rendered prompt"

    def __call__(self, texts: list[str], **kwargs: object) -> FakeInputs:
        assert texts == ["rendered prompt"]
        assert kwargs == {"return_tensors": "pt", "padding": True}
        return FakeInputs()


class FakeStreamer:
    _STOP = object()

    def __init__(self) -> None:
        self.items: Queue[object] = Queue()
        self.end_calls = 0

    def emit(self, text: str) -> None:
        self.items.put(text)

    def end(self) -> None:
        self.end_calls += 1
        self.items.put(self._STOP)

    def __iter__(self) -> "FakeStreamer":
        return self

    def __next__(self) -> str:
        item = self.items.get(timeout=2)
        if item is self._STOP:
            raise StopIteration
        return str(item)


class FakeModel:
    device = "cpu"

    def __init__(
        self,
        chunks: list[str],
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        streamer = kwargs["streamer"]
        for chunk in self.chunks:
            streamer.emit(chunk)
        if self.error is not None:
            raise self.error
        streamer.end()


MESSAGES = [{"role": "user", "content": "خبر عربي"}]


def build_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunks: list[str],
    error: Exception | None = None,
) -> tuple[QwenModel, FakeModel, FakeStreamer]:
    streamer = FakeStreamer()
    monkeypatch.setattr(
        "src.models.qwen.TextIteratorStreamer",
        lambda *args, **kwargs: streamer,
    )
    model = FakeModel(chunks, error)
    runtime = QwenModel(
        model=model,
        tokenizer=FakeTokenizer(),
        logits_processors=[lambda input_ids, scores: scores],
    )
    return runtime, model, streamer


def test_qwen_is_a_streaming_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, _ = build_runtime(monkeypatch, chunks=["ok"])
    assert isinstance(runtime, StreamingLanguageModel)


def test_stream_yields_real_chunks_and_preserves_generation_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, model, _ = build_runtime(
        monkeypatch,
        chunks=["{", '"story_title": "خبر"', "}"],
    )

    assert list(runtime.stream(MESSAGES)) == [
        "{",
        '"story_title": "خبر"',
        "}",
    ]
    assert model.kwargs is not None
    assert model.kwargs["max_new_tokens"] == qwen_generation_kwargs()[
        "max_new_tokens"
    ]
    assert model.kwargs["logits_processor"] == runtime.logits_processors
    assert "streamer" in model.kwargs


def test_stream_drains_partial_output_then_raises_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, streamer = build_runtime(
        monkeypatch,
        chunks=["partial"],
        error=ValueError("decoder failed"),
    )
    generated = runtime.stream(MESSAGES)

    assert next(generated) == "partial"
    with pytest.raises(RuntimeError, match="Qwen streaming generation failed"):
        next(generated)
    assert streamer.end_calls == 1


def test_stream_rejects_blank_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, _ = build_runtime(monkeypatch, chunks=["   "])

    with pytest.raises(RuntimeError, match="blank output"):
        list(runtime.stream(MESSAGES))
