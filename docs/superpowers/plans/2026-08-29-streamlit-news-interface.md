# Streamlit News Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a clean Streamlit application that performs information extraction and translation with the existing fine-tuned Qwen adapter while streaming genuine model output.

**Architecture:** Add a streaming protocol beside the existing synchronous model contract, implement it for Transformers and vLLM, and place a task-focused controller between those runtimes and Streamlit. Streamlit loads the controller lazily through `st.cache_resource`, owns only input/result presentation, and can switch between a direct fine-tuned runtime and a separately deployed vLLM server through environment variables.

**Tech Stack:** Python 3.12, Streamlit, Transformers `TextIteratorStreamer`, Qwen2.5, PEFT/LoRA, OpenAI-compatible vLLM client, Pydantic 2, pytest, uv, Docker/Hugging Face Spaces.

---

## File Map

New files:

- `app.py`: minimal Streamlit entry point.
- `src/controllers/streaming.py`: task prompt selection, serialized streaming, and final schema validation.
- `src/ui/__init__.py`: UI package marker.
- `src/ui/input.py`: upload decoding and paste/upload precedence without a Streamlit dependency.
- `src/ui/streamlit_app.py`: page composition, cached controller, live output, structured results, and user-facing errors.
- `tests/unit/test_qwen_streaming.py`: local Transformers streaming behavior.
- `tests/unit/test_vllm_streaming.py`: OpenAI-compatible delta streaming behavior.
- `tests/unit/test_streaming_configuration.py`: environment and adapter-source behavior.
- `tests/unit/test_streaming_controller.py`: extraction/translation routing, locking, and schema validation.
- `tests/unit/test_ui_input.py`: upload decoding and precedence.
- `tests/unit/test_streamlit_app.py`: initial UI, task switching, lazy loading, and resource caching.
- `tests/integration/test_finetuned_streaming.py`: opt-in real adapter extraction/translation smoke tests.
- `Dockerfile`: Hugging Face Docker Space image.
- `.dockerignore`: prevent local weights, caches, secrets, and generated outputs from entering the image.
- `deployment/huggingface/README.md`: Space YAML/front-matter template.
- `docs/streamlit-deployment.md`: local, Docker, direct-GPU Space, and remote-vLLM deployment guide.

Modified files:

- `src/models/language_model.py`: additive `StreamingLanguageModel` protocol.
- `src/models/qwen.py`: shared prompt/input preparation and real Transformers streaming.
- `src/models/vllm.py`: real chat-completion streaming and configurable served model ID.
- `src/models/finetuned_qwen.py`: accept either a verified local adapter or a valid Hub repository ID.
- `src/helpers/environment.py`: optional non-secret environment setting reader.
- `src/models/factory.py`: preserve defaults while honoring adapter and vLLM environment overrides.
- `pyproject.toml`: isolated `app` dependency group and smoke-test marker.
- `uv.lock`: locked Streamlit application dependencies.

The existing dirty `README.md`, `configs/llamafactory/news_finetune.yaml`, and `tests/load/locustfile.py` must not be staged, reverted, or reformatted.

### Task 1: Add the streaming contract and Transformers implementation

**Files:**

- Modify: `src/models/language_model.py`
- Modify: `src/models/qwen.py`
- Create: `tests/unit/test_qwen_streaming.py`

- [ ] **Step 1: Write the failing Transformers streaming tests**

Create `tests/unit/test_qwen_streaming.py` with queue-backed fakes so no model is downloaded:

```python
from queue import Queue
from types import SimpleNamespace
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
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run:

```powershell
uv run --group evaluation --group test pytest tests/unit/test_qwen_streaming.py -v -p no:cacheprovider
```

Expected: collection fails because `StreamingLanguageModel` and `QwenModel.stream` do not exist.

- [ ] **Step 3: Add the additive protocol**

Update `src/models/language_model.py` to import `Iterator` and add this protocol after `LanguageModel`:

```python
from collections.abc import Iterator
from typing import Protocol, TypedDict, runtime_checkable


@runtime_checkable
class StreamingLanguageModel(LanguageModel, Protocol):
    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield decoded model output as generation produces it."""
        ...
```

Keep `ChatMessage`, `LanguageModel`, and `TokenizedLanguageModel` otherwise unchanged.

- [ ] **Step 4: Implement real Transformers streaming**

In `src/models/qwen.py`, add these imports:

```python
from collections.abc import Callable, Iterator
from queue import Queue
from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
```

Add two private helpers to `QwenModel` and make synchronous generation use them:

```python
    def _prepare_generation_inputs(
        self,
        messages: list[ChatMessage],
    ) -> Any:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

    def _generation_kwargs(self) -> dict[str, Any]:
        generation_kwargs = qwen_generation_kwargs()
        if self.logits_processors:
            generation_kwargs["logits_processor"] = self.logits_processors
        return generation_kwargs
```

Replace the duplicated preparation at the start of `generate()` with:

```python
        model_inputs = self._prepare_generation_inputs(messages)
        generation_kwargs = self._generation_kwargs()
```

Add this method after `generate()`:

```python
    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield genuine decoder output from a background generation thread."""
        model_inputs = self._prepare_generation_inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = self._generation_kwargs()
        generation_kwargs["streamer"] = streamer
        worker_errors: Queue[BaseException] = Queue(maxsize=1)

        def generate_in_worker() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **model_inputs,
                        **generation_kwargs,
                    )
            except BaseException as exc:
                worker_errors.put(exc)
                streamer.end()

        worker = Thread(
            target=generate_in_worker,
            name="qwen-streaming-generation",
            daemon=True,
        )
        worker.start()
        emitted: list[str] = []
        try:
            for chunk in streamer:
                if chunk:
                    emitted.append(chunk)
                    yield chunk
        finally:
            worker.join()

        if not worker_errors.empty():
            cause = worker_errors.get_nowait()
            raise RuntimeError("Qwen streaming generation failed.") from cause
        if not "".join(emitted).strip():
            raise RuntimeError("Qwen streaming generation returned blank output.")
```

- [ ] **Step 5: Run focused and synchronous regression tests**

Run:

```powershell
uv run --group evaluation --group test pytest tests/unit/test_qwen_streaming.py -v -p no:cacheprovider
uv run --group evaluation python -m compileall -q src/models
```

Expected: four tests pass and compilation exits 0.

- [ ] **Step 6: Commit the streaming protocol and Qwen runtime**

```powershell
git add src/models/language_model.py src/models/qwen.py tests/unit/test_qwen_streaming.py
git commit -m "feat: stream fine-tuned Qwen generation"
```

### Task 2: Add vLLM delta streaming

**Files:**

- Modify: `src/models/vllm.py`
- Create: `tests/unit/test_vllm_streaming.py`

- [ ] **Step 1: Write failing vLLM stream tests**

Create `tests/unit/test_vllm_streaming.py`:

```python
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


def test_vllm_stream_yields_delta_content() -> None:
    runtime, completions = runtime_with(
        [event(None), event("{\"translated"), event("_title\": \"خبر\"}")]
    )

    assert list(runtime.stream([{"role": "user", "content": "x"}])) == [
        "{\"translated",
        "_title\": \"خبر\"}",
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
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
uv run --group serve --group test pytest tests/unit/test_vllm_streaming.py -v -p no:cacheprovider
```

Expected: tests fail because `VLLMModel.stream` is missing.

- [ ] **Step 3: Implement vLLM streaming and configurable model loading**

Add `Iterator` to the imports and extend `VLLMModel.load()` with `model_id`:

```python
from collections.abc import Callable, Iterator

    @classmethod
    def load(
        cls,
        *,
        base_url: str = VLLM_API_BASE_URL,
        api_key: str = VLLM_LOCAL_API_KEY,
        model_id: str = VLLM_MODEL_ID,
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
        return cls(client, model_id=model_id)
```

Add `stream()` after the existing synchronous `generate()`:

```python
    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        try:
            events = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            emitted: list[str] = []
            for event in events:
                choices = getattr(event, "choices", None)
                if choices == []:
                    continue
                if not choices:
                    raise RuntimeError(
                        "vLLM returned an invalid streaming response with no choices."
                    )
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    raise RuntimeError("vLLM returned an invalid streaming delta.")
                content = getattr(delta, "content", None)
                if content is None:
                    continue
                if not isinstance(content, str):
                    raise RuntimeError("vLLM returned non-text streaming content.")
                if content:
                    emitted.append(content)
                    yield content
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "vLLM streaming generation failed; check server readiness and credentials."
            ) from exc

        complete = "".join(emitted)
        if not complete.strip():
            raise RuntimeError("vLLM returned blank streamed content.")
        if contains_chinese_characters(complete):
            raise RuntimeError(
                "vLLM mandatory suppression is broken: generated content contains CJK characters."
            )
```

- [ ] **Step 4: Run vLLM tests and compile**

Run:

```powershell
uv run --group serve --group test pytest tests/unit/test_vllm_streaming.py -v -p no:cacheprovider
uv run --group serve python -m compileall -q src/models/vllm.py
```

Expected: five tests pass and compilation exits 0.

- [ ] **Step 5: Commit vLLM streaming**

```powershell
git add src/models/vllm.py tests/unit/test_vllm_streaming.py
git commit -m "feat: stream responses from vLLM"
```

### Task 3: Add deployment-safe runtime configuration and Hub adapter loading

**Files:**

- Modify: `src/helpers/environment.py`
- Modify: `src/models/finetuned_qwen.py`
- Modify: `src/models/factory.py`
- Create: `tests/unit/test_streaming_configuration.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tests/unit/test_streaming_configuration.py`:

```python
from pathlib import Path
from typing import Any

import pytest

from src.helpers.environment import read_optional_setting
from src.models.finetuned_qwen import resolve_adapter_source


def test_optional_setting_prefers_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SETTING=file-value\n", encoding="utf-8")
    assert read_optional_setting(
        "SETTING",
        env_path=env_file,
        environ={"SETTING": " process-value "},
    ) == "process-value"


def test_optional_setting_uses_file_then_none(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SETTING=file-value\n", encoding="utf-8")
    assert read_optional_setting(
        "SETTING", env_path=env_file, environ={}
    ) == "file-value"
    assert read_optional_setting(
        "MISSING", env_path=env_file, environ={}
    ) is None


def test_resolve_adapter_source_validates_existing_local_adapter(
    tmp_path: Path,
) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert resolve_adapter_source(tmp_path) == str(tmp_path.resolve())


def test_resolve_adapter_source_accepts_hub_repository() -> None:
    assert resolve_adapter_source(
        "marouaHattab/ArabLLM-news"
    ) == "marouaHattab/ArabLLM-news"


@pytest.mark.parametrize(
    "source",
    ["", "missing-local-directory", "outputs/models/missing"],
)
def test_resolve_adapter_source_rejects_invalid_source(source: str) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_adapter_source(source)


def test_factory_passes_adapter_environment_to_finetuned_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load(token: str, adapter_path: str) -> object:
        captured.update(token=token, adapter_path=adapter_path)
        return object()

    monkeypatch.setenv(
        "FINETUNED_ADAPTER_SOURCE", "marouaHattab/ArabLLM-news"
    )
    monkeypatch.setattr(
        "src.models.finetuned_qwen.FineTunedQwenModel.load", fake_load
    )
    from src.models.factory import _load_finetuned

    _load_finetuned("token")
    assert captured == {
        "token": "token",
        "adapter_path": "marouaHattab/ArabLLM-news",
    }


def test_factory_passes_vllm_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("VLLM_API_BASE_URL", "https://inference.example/v1")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("VLLM_MODEL_ID", "deployed-news")
    monkeypatch.setattr("src.models.vllm.VLLMModel.load", fake_load)
    from src.models.factory import _load_vllm

    _load_vllm()
    assert captured == {
        "base_url": "https://inference.example/v1",
        "api_key": "secret",
        "model_id": "deployed-news",
    }
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
uv run --group evaluation --group serve --group test pytest tests/unit/test_streaming_configuration.py -v -p no:cacheprovider
```

Expected: collection fails because `read_optional_setting` and `resolve_adapter_source` do not exist.

- [ ] **Step 3: Add the optional setting helper**

Append to `src/helpers/environment.py`:

```python
def read_optional_setting(
    key_name: str,
    env_path: str | Path = ENV_PATH,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Read a setting from the process or dotenv file without requiring it."""
    target_environ = os.environ if environ is None else environ
    file_values = dotenv_values(env_path)
    value = target_environ.get(key_name)
    if value is None:
        value = file_values.get(key_name)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
```

- [ ] **Step 4: Resolve local and Hub adapter sources**

In `src/models/finetuned_qwen.py`, add:

```python
def resolve_adapter_source(adapter_source: str | Path) -> str:
    """Return a validated local path or a normalized owner/repository Hub ID."""
    if isinstance(adapter_source, Path):
        return str(validate_adapter_path(adapter_source))

    source = adapter_source.strip()
    if not source:
        raise ValueError("Fine-tuned adapter source must not be empty.")
    candidate = Path(source)
    if candidate.exists():
        return str(validate_adapter_path(candidate))
    if "\\" in source or source.startswith((".", "/")) or source.count("/") != 1:
        raise FileNotFoundError(f"Fine-tuned adapter source not found: {source}")
    owner, repository = source.split("/", maxsplit=1)
    if not owner.strip() or not repository.strip():
        raise ValueError(f"Invalid Hugging Face adapter repository: {source}")
    return source
```

Then change the start of `FineTunedQwenModel.load()` and its adapter call:

```python
        adapter_source = resolve_adapter_source(adapter_path)
        runtime = super().load(
            token=token,
            model_id=model_id,
            target=target,
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
        )

        try:
            runtime.model.load_adapter(adapter_source, token=token)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the fine-tuned adapter from: "
                f"{adapter_source}"
            ) from exc

        runtime.model.eval()
        runtime.model_id = adapter_source
```

- [ ] **Step 5: Wire environment overrides into the existing factory**

Add imports in `src/models/factory.py`:

```python
from src.helpers.config import (
    FINETUNED_MODEL_DIR,
    VLLM_API_BASE_URL,
    VLLM_LOCAL_API_KEY,
    VLLM_MODEL_ID,
)
from src.helpers.environment import read_optional_setting, require_api_key
```

Replace `_load_finetuned()` and `_load_vllm()` with:

```python
def _load_finetuned(token: str) -> LanguageModel:
    from src.models.finetuned_qwen import FineTunedQwenModel

    adapter_source = (
        read_optional_setting("FINETUNED_ADAPTER_SOURCE")
        or FINETUNED_MODEL_DIR
    )
    return FineTunedQwenModel.load(token, adapter_path=adapter_source)


def _load_vllm() -> LanguageModel:
    from src.models.vllm import VLLMModel

    return VLLMModel.load(
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
```

- [ ] **Step 6: Run configuration and factory regressions**

Run:

```powershell
uv run --group evaluation --group serve --group test pytest tests/unit/test_streaming_configuration.py tests/unit/test_qwen_streaming.py tests/unit/test_vllm_streaming.py -v -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 7: Commit runtime configuration**

```powershell
git add src/helpers/environment.py src/models/finetuned_qwen.py src/models/factory.py tests/unit/test_streaming_configuration.py
git commit -m "feat: configure streaming inference backends"
```

### Task 4: Add the task streaming controller

**Files:**

- Create: `src/controllers/streaming.py`
- Create: `tests/unit/test_streaming_controller.py`

- [ ] **Step 1: Write failing controller tests**

Create `tests/unit/test_streaming_controller.py`:

```python
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
```

- [ ] **Step 2: Run the controller tests and verify failure**

Run:

```powershell
uv run --group evaluation --group test pytest tests/unit/test_streaming_controller.py -v -p no:cacheprovider
```

Expected: collection fails because `src.controllers.streaming` does not exist.

- [ ] **Step 3: Implement the controller**

Create `src/controllers/streaming.py`:

```python
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
    runtime_loader: Callable[[str], LanguageModel] = load_language_model,
) -> NewsStreamingController:
    selected_provider = (
        provider
        or read_optional_setting("NEWS_MODEL_PROVIDER")
        or "finetuned"
    ).strip().casefold()
    runtime = runtime_loader(selected_provider)
    if not isinstance(runtime, StreamingLanguageModel):
        raise TypeError(
            f"Model provider {selected_provider!r} does not support streaming."
        )
    return NewsStreamingController(cast(StreamingLanguageModel, runtime))
```

- [ ] **Step 4: Run controller and task regressions**

Run:

```powershell
uv run --group evaluation --group test pytest tests/unit/test_streaming_controller.py -v -p no:cacheprovider
uv run --group evaluation python -c "from src.tasks import build_extraction_messages, build_translation_messages; assert build_extraction_messages('x'); assert build_translation_messages('x')"
```

Expected: seven tests pass and both prompt builders remain callable.

- [ ] **Step 5: Commit the controller**

```powershell
git add src/controllers/streaming.py tests/unit/test_streaming_controller.py
git commit -m "feat: add streaming news task controller"
```

### Task 5: Add upload/input handling

**Files:**

- Create: `src/ui/__init__.py`
- Create: `src/ui/input.py`
- Create: `tests/unit/test_ui_input.py`

- [ ] **Step 1: Write failing upload/input tests**

Create `tests/unit/test_ui_input.py`:

```python
import pytest

from src.ui.input import decode_uploaded_story, resolve_story_input


def test_decode_uploaded_story_accepts_utf8_bom() -> None:
    assert decode_uploaded_story("story.txt", b"\xef\xbb\xbf\xd8\xae\xd8\xa8\xd8\xb1") == "خبر"


@pytest.mark.parametrize("filename", ["story.txt", "story.md", "STORY.TXT"])
def test_decode_uploaded_story_accepts_supported_extensions(filename: str) -> None:
    assert decode_uploaded_story(filename, "قصة".encode("utf-8")) == "قصة"


def test_pasted_story_takes_precedence() -> None:
    assert resolve_story_input(
        " pasted story ",
        uploaded_name="story.txt",
        uploaded_bytes=b"uploaded story",
    ) == "pasted story"


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("story.pdf", b"content", "TXT or Markdown"),
        ("story.txt", b"", "empty"),
        ("story.txt", b"\xff", "UTF-8"),
    ],
)
def test_decode_uploaded_story_reports_precise_input_errors(
    filename: str,
    data: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_uploaded_story(filename, data)


def test_resolve_story_input_rejects_empty_submission() -> None:
    with pytest.raises(ValueError, match="Paste a story or upload"):
        resolve_story_input("", uploaded_name=None, uploaded_bytes=None)
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```powershell
uv run --group test pytest tests/unit/test_ui_input.py -v -p no:cacheprovider
```

Expected: collection fails because `src.ui.input` does not exist.

- [ ] **Step 3: Implement upload/input handling**

Create an empty `src/ui/__init__.py`, then create `src/ui/input.py`:

```python
from pathlib import Path


SUPPORTED_STORY_SUFFIXES = {".txt", ".md"}


def decode_uploaded_story(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_STORY_SUFFIXES:
        raise ValueError("Upload a UTF-8 TXT or Markdown file.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    try:
        story = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded file must use UTF-8 encoding.") from exc
    if not story:
        raise ValueError("The uploaded file contains no story text.")
    return story


def resolve_story_input(
    pasted_story: str,
    *,
    uploaded_name: str | None,
    uploaded_bytes: bytes | None,
) -> str:
    pasted = pasted_story.strip()
    if pasted:
        return pasted
    if uploaded_name is not None and uploaded_bytes is not None:
        return decode_uploaded_story(uploaded_name, uploaded_bytes)
    raise ValueError("Paste a story or upload a UTF-8 TXT/Markdown file.")
```

- [ ] **Step 4: Run the upload tests**

Run:

```powershell
uv run --group test pytest tests/unit/test_ui_input.py -v -p no:cacheprovider
```

Expected: all parameterized cases pass.

- [ ] **Step 5: Commit upload handling**

```powershell
git add src/ui/__init__.py src/ui/input.py tests/unit/test_ui_input.py
git commit -m "feat: validate Streamlit story inputs"
```

### Task 6: Add Streamlit dependencies and the live interface

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `app.py`
- Create: `src/ui/streamlit_app.py`
- Create: `tests/unit/test_streamlit_app.py`

- [ ] **Step 1: Add and lock the application dependency group**

Add this group under `[dependency-groups]` in `pyproject.toml`:

```toml
app = [
    "accelerate>=1.2.1,<2",
    "huggingface-hub>=0.36.0,<2",
    "openai>=1.61,<2",
    "peft==0.18.1",
    "streamlit>=1.42,<2",
    "torch>=2.5.1",
    "transformers==4.48.3",
]
```

Then run:

```powershell
uv lock
uv sync --group app --group test
```

Expected: `uv.lock` changes and `uv run --group app streamlit version` prints an installed Streamlit version.

- [ ] **Step 2: Write failing Streamlit tests**

Create `tests/unit/test_streamlit_app.py`:

```python
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import streamlit_app


APP_PATH = Path(__file__).resolve().parents[2] / "app.py"


def test_initial_page_is_clean_and_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(provider: str) -> object:
        raise AssertionError(f"model loaded during initial render: {provider}")

    monkeypatch.setattr(
        streamlit_app,
        "build_streaming_controller",
        fail_if_loaded,
    )
    streamlit_app.get_controller.clear()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "ArabLLM Newsroom"
    assert app.radio[0].value == "Information Extraction"
    assert len(app.selectbox) == 0


def test_switching_to_translation_reveals_language_controls() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    app.radio[0].set_value("Translation").run(timeout=10)

    assert not app.exception
    assert len(app.selectbox) == 2
    assert app.selectbox[0].value == "Arabic"
    assert app.selectbox[1].value == "English"


def test_controller_resource_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    def fake_builder(provider: str) -> object:
        created.append(provider)
        return {"provider": provider}

    monkeypatch.setattr(
        streamlit_app,
        "build_streaming_controller",
        fake_builder,
    )
    streamlit_app.get_controller.clear()
    first = streamlit_app.get_controller("finetuned")
    second = streamlit_app.get_controller("finetuned")

    assert first is second
    assert created == ["finetuned"]
    streamlit_app.get_controller.clear()
```

- [ ] **Step 3: Add the minimal entry point and confirm the red state**

Create `app.py`:

```python
from src.ui.streamlit_app import main


main()
```

Run:

```powershell
uv run --group app --group test pytest tests/unit/test_streamlit_app.py -v -p no:cacheprovider
```

Expected: collection fails because `src.ui.streamlit_app` does not exist.

- [ ] **Step 4: Implement the Streamlit page**

Create `src/ui/streamlit_app.py` with the following complete public structure and behavior:

```python
import html
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any, cast

import streamlit as st
from pydantic import ValidationError

from src.controllers.streaming import (
    NewsStreamingController,
    TaskName,
    ValidatedTaskResponse,
    build_streaming_controller,
)
from src.helpers.environment import read_optional_setting
from src.models.news import NewsDetails
from src.models.translation import TranslatedStory
from src.ui.input import resolve_story_input


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["Arabic", "English", "French"]
TASK_LABELS: dict[str, TaskName] = {
    "Information Extraction": "extraction",
    "Translation": "translation",
}

PAGE_CSS = """
<style>
:root {
  --paper: #f4f1eb;
  --ink: #161616;
  --muted: #696762;
  --rule: #cbc6bc;
  --signal: #b3261e;
}
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -0.035em; }
h1 { font-size: clamp(2.2rem, 5vw, 4.8rem) !important; line-height: .95 !important; }
.eyebrow { color: var(--signal); font-size: .75rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; }
.deck { color: var(--muted); max-width: 720px; font-size: 1.02rem; }
.editorial-rule { border-top: 1px solid var(--ink); margin: 1.2rem 0 1.7rem; }
.result-card { border-top: 1px solid var(--ink); padding-top: 1rem; margin-top: .5rem; }
.result-label { color: var(--signal); font-size: .7rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
.auto-direction { direction: auto; unicode-bidi: plaintext; white-space: pre-wrap; line-height: 1.75; }
.empty-proof { border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); color: var(--muted); padding: 2.5rem 0; }
[data-testid="stFileUploaderDropzone"] { background: transparent; border-color: var(--rule); }
[data-testid="stChatMessage"] { background: #fff; border: 1px solid var(--rule); border-radius: 0; }
.stButton > button { border-radius: 0; border: 1px solid var(--ink); background: var(--ink); color: white; }
.stButton > button:hover { border-color: var(--signal); background: var(--signal); color: white; }
@media (max-width: 760px) {
  .block-container { padding: 1.2rem 1rem 3rem; }
  h1 { font-size: 2.7rem !important; }
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_controller(provider: str) -> NewsStreamingController:
    return build_streaming_controller(provider)


def _directional_text(text: str, css_class: str = "auto-direction") -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div class="{css_class}" dir="auto">{escaped}</div>'


def _runtime_error_message(exc: Exception) -> str:
    message = str(exc).casefold()
    if "out of memory" in message or "cuda" in message and "memory" in message:
        return "The GPU ran out of memory. Try a shorter story or use the vLLM backend."
    if isinstance(exc, FileNotFoundError) or "adapter" in message:
        return "The fine-tuned adapter could not be loaded. Check FINETUNED_ADAPTER_SOURCE and HF_TOKEN."
    if "vllm" in message or "connection" in message:
        return "The inference server is unavailable. Check its URL, API key, and model readiness."
    return "Generation failed. Check the server logs for details and try again."


def _validation_error_message(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"The model returned incomplete or invalid JSON: {exc.msg}."
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return f"The JSON does not match the expected schema at {location}: {first['msg']}."
    return str(exc)


def _render_extraction(result: NewsDetails) -> None:
    st.markdown('<p class="result-label">Extracted story</p>', unsafe_allow_html=True)
    st.markdown(_directional_text(result.story_title), unsafe_allow_html=True)
    category_col, keyword_col = st.columns([1, 2])
    category_col.metric("Category", result.story_category.replace("_", " ").title())
    keyword_col.write(" · ".join(result.story_keywords))
    st.markdown("#### Summary")
    for point in result.story_summary:
        st.markdown(_directional_text(f"— {point}"), unsafe_allow_html=True)
    st.markdown("#### Entities")
    st.dataframe(
        [entity.model_dump() for entity in result.story_entities],
        use_container_width=True,
        hide_index=True,
    )


def _render_translation(result: TranslatedStory) -> None:
    st.markdown('<p class="result-label">Translation</p>', unsafe_allow_html=True)
    st.markdown(_directional_text(result.translated_title), unsafe_allow_html=True)
    st.markdown('<div class="editorial-rule"></div>', unsafe_allow_html=True)
    st.markdown(_directional_text(result.translated_content), unsafe_allow_html=True)


def render_validated_result(result: ValidatedTaskResponse) -> None:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if isinstance(result, NewsDetails):
        _render_extraction(result)
    elif isinstance(result, TranslatedStory):
        _render_translation(result)
    st.markdown("</div>", unsafe_allow_html=True)


def _stream_with_status(
    controller: NewsStreamingController,
    task: TaskName,
    story: str,
    *,
    source_language: str,
    target_language: str,
    status: Any,
    chunks: list[str],
) -> Iterator[str]:
    first_chunk = True
    for chunk in controller.stream_task(
        task,
        story,
        source_language=source_language,
        target_language=target_language,
    ):
        if first_chunk:
            status.update(label="Receiving model output…", state="running")
            first_chunk = False
        chunks.append(chunk)
        yield chunk


def _render_previous_result(task: TaskName) -> None:
    previous = st.session_state.get("last_result")
    if not previous or previous.get("task") != task:
        st.markdown(
            '<div class="empty-proof">Your streamed model response will appear here.</div>',
            unsafe_allow_html=True,
        )
        return
    raw = cast(str, previous.get("raw", ""))
    error = previous.get("error")
    validated = previous.get("validated")
    if error:
        st.error(str(error))
    if validated is not None:
        render_validated_result(cast(ValidatedTaskResponse, validated))
    if raw:
        with st.expander("Raw model response"):
            st.code(raw, language="json", wrap_lines=True)


def main() -> None:
    st.set_page_config(
        page_title="ArabLLM Newsroom",
        page_icon="📰",
        layout="wide",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">Fine-tuned Arabic news model</p>', unsafe_allow_html=True)
    st.title("ArabLLM Newsroom")
    st.markdown(
        '<p class="deck">Extract structured editorial details or translate a story while the fine-tuned model writes its response live.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="editorial-rule"></div>', unsafe_allow_html=True)

    input_column, result_column = st.columns([0.95, 1.05], gap="large")
    with input_column:
        task_label = st.radio(
            "Task",
            list(TASK_LABELS),
            horizontal=True,
        )
        task = TASK_LABELS[task_label]
        pasted_story = st.text_area(
            "Story",
            height=300,
            placeholder="Paste the Arabic or source-language news story…",
        )
        uploaded = st.file_uploader(
            "Or upload a story",
            type=["txt", "md"],
            help="UTF-8 TXT or Markdown. Pasted text takes precedence.",
        )
        st.caption("If both are provided, the pasted story is used.")
        source_language = "Arabic"
        target_language = "English"
        if task == "translation":
            language_columns = st.columns(2)
            source_language = language_columns[0].selectbox(
                "Source language", LANGUAGES, index=0
            )
            target_language = language_columns[1].selectbox(
                "Target language", LANGUAGES, index=1
            )
        submitted = st.button(
            "Extract information" if task == "extraction" else "Translate story",
            type="primary",
            use_container_width=True,
        )

    with result_column:
        st.markdown("### Live proof")
        if not submitted:
            _render_previous_result(task)
            return

        try:
            story = resolve_story_input(
                pasted_story,
                uploaded_name=uploaded.name if uploaded is not None else None,
                uploaded_bytes=uploaded.getvalue() if uploaded is not None else None,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        provider = read_optional_setting("NEWS_MODEL_PROVIDER") or "finetuned"
        status = st.status("Loading inference runtime…", expanded=False)
        chunks: list[str] = []
        raw_response = ""
        try:
            controller = get_controller(provider)
            status.update(label="Generating…", state="running")
            with st.chat_message("assistant"):
                streamed = st.write_stream(
                    _stream_with_status(
                        controller,
                        task,
                        story,
                        source_language=source_language,
                        target_language=target_language,
                        status=status,
                        chunks=chunks,
                    )
                )
            raw_response = streamed if isinstance(streamed, str) else "".join(chunks)
            status.update(label="Generation complete", state="complete")
        except Exception as exc:
            LOGGER.exception("Streaming inference failed")
            raw_response = "".join(chunks)
            message = _runtime_error_message(exc)
            status.update(label="Generation failed", state="error")
            st.error(message)
            if raw_response:
                with st.expander("Partial model response", expanded=True):
                    st.code(raw_response, language="json", wrap_lines=True)
            st.session_state["last_result"] = {
                "task": task,
                "raw": raw_response,
                "validated": None,
                "error": message,
            }
            return

        try:
            validated = controller.validate_task_response(task, raw_response)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            message = _validation_error_message(exc)
            st.warning(message)
            with st.expander("Raw model response", expanded=True):
                st.code(raw_response, language="json", wrap_lines=True)
            st.session_state["last_result"] = {
                "task": task,
                "raw": raw_response,
                "validated": None,
                "error": message,
            }
            return

        render_validated_result(validated)
        with st.expander("Raw model response"):
            st.code(raw_response, language="json", wrap_lines=True)
        st.session_state["last_result"] = {
            "task": task,
            "raw": raw_response,
            "validated": validated,
            "error": None,
        }
```

- [ ] **Step 5: Run Streamlit and UI helper tests**

Run:

```powershell
uv run --group app --group test pytest tests/unit/test_streamlit_app.py tests/unit/test_ui_input.py -v -p no:cacheprovider
```

Expected: all tests pass; the initial and task-switching tests report no Streamlit exceptions.

- [ ] **Step 6: Run a headless startup health check**

Start the server:

```powershell
uv run --group app streamlit run app.py --server.headless=true --server.port=8501
```

In a second terminal, run:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health
```

Expected: HTTP 200 with body `ok`. Stop the server with Ctrl+C.

- [ ] **Step 7: Commit the Streamlit application**

```powershell
git add app.py src/ui/streamlit_app.py tests/unit/test_streamlit_app.py pyproject.toml uv.lock
git commit -m "feat: add streaming Streamlit newsroom"
```

### Task 7: Add Hugging Face Docker Space deployment assets

**Files:**

- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `deployment/huggingface/README.md`
- Create: `docs/streamlit-deployment.md`

- [ ] **Step 1: Create the Docker image definition**

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    NEWS_MODEL_PROVIDER=finetuned \
    FINETUNED_ADAPTER_SOURCE=marouaHattab/ArabLLM-news

WORKDIR /home/user/app
COPY --chown=user:user . .
RUN uv sync --frozen --group app --no-dev

EXPOSE 7860
CMD ["uv", "run", "--frozen", "--group", "app", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860"]
```

- [ ] **Step 2: Exclude caches, secrets, and generated artifacts**

Create `.dockerignore`:

```text
.git
.venv
.venv-wsl
.cache
.hf-cache
.pytest_cache
.uv-cache
src/.env
outputs
data/raw
data/datasets
LlamaFactor
__pycache__
*.pyc
```

- [ ] **Step 3: Add the Hugging Face Space metadata template**

Create `deployment/huggingface/README.md`:

```markdown
---
title: ArabLLM Newsroom
emoji: 📰
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
suggested_hardware: t4-small
models:
  - Qwen/Qwen2.5-1.5B-Instruct
  - marouaHattab/ArabLLM-news
---

# ArabLLM Newsroom

Streamlit interface for structured Arabic-news extraction and translation with
the `marouaHattab/ArabLLM-news` fine-tuned adapter.
```

- [ ] **Step 4: Write exact local and deployment instructions**

Create `docs/streamlit-deployment.md` with these sections and commands:

````markdown
# Streamlit application and deployment

## Recommended architecture

For local development or a low-traffic demonstration, run Streamlit and the
fine-tuned Transformers runtime in one process. The base Qwen model and LoRA
adapter are cached once per process, and requests are serialized to protect
VRAM.

For public or concurrent traffic, deploy Streamlit as a CPU frontend and keep
vLLM on a secured GPU service. Only the Streamlit server calls vLLM; browsers
never receive the inference API key.

## Local direct-model run

Create `src/.env` with:

```dotenv
HF_TOKEN=hf_your_read_token
NEWS_MODEL_PROVIDER=finetuned
```

The local adapter at `outputs/models/news-finetune` is used by default. To load
the public Hub adapter instead, add:

```dotenv
FINETUNED_ADAPTER_SOURCE=marouaHattab/ArabLLM-news
```

Install and run:

```powershell
uv sync --group app
uv run --group app streamlit run app.py
```

Open `http://localhost:8501`.

## Local vLLM-backed run

Start the existing WSL vLLM service, then configure:

```dotenv
NEWS_MODEL_PROVIDER=vllm
VLLM_API_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=local-vllm
VLLM_MODEL_ID=news-lora
```

Run the same Streamlit command. Extraction and translation prompts still come
from the existing task modules.

## Hugging Face Docker Space: direct model

1. Create a Docker Space and choose at least a T4 Small GPU for responsive
   generation. CPU hardware can start the model but is too slow for a good
   interactive experience.
2. Copy the YAML block from `deployment/huggingface/README.md` into the Space's
   root README, or let the Space creation form generate equivalent Docker
   metadata.
3. Push this repository, including `Dockerfile`, `app.py`, `pyproject.toml`, and
   `uv.lock`, to the Space repository.
4. Add `HF_TOKEN` as a Space Secret. Add these as Space Variables:

```text
NEWS_MODEL_PROVIDER=finetuned
FINETUNED_ADAPTER_SOURCE=marouaHattab/ArabLLM-news
```

The Space downloads the public LoRA adapter and its Qwen base model when the
container first loads the cached controller. A restarted Space downloads them
again unless persistent storage/cache is configured. Never commit `HF_TOKEN`.

Streamlit is not a native Hugging Face Space SDK; this project uses the Docker
SDK and exposes port 7860. Gradio ZeroGPU does not accelerate a Streamlit Docker
Space, so free Streamlit hosting is suitable only for a very slow CPU demo or
for a frontend connected to external inference.

## Hugging Face Docker Space: separate vLLM

Deploy the same Docker image on CPU hardware and set:

```text
NEWS_MODEL_PROVIDER=vllm
VLLM_API_BASE_URL=https://your-secured-inference-host/v1
VLLM_MODEL_ID=news-lora
```

Store `VLLM_API_KEY` as a Space Secret. The vLLM endpoint must use HTTPS,
authentication, request limits, and network restrictions where available. Do
not expose an unauthenticated raw vLLM port to the internet.

## Docker verification

```powershell
docker build -t arabllm-newsroom .
docker run --rm -p 7860:7860 `
  -e HF_TOKEN=$env:HF_TOKEN `
  arabllm-newsroom
```

Open `http://localhost:7860`.
````

- [ ] **Step 5: Validate Docker assets without changing running services**

Run:

```powershell
docker build -t arabllm-newsroom .
```

Expected: build exits 0. If Docker is unavailable, record the exact blocker and still run the Python verification in Task 9.

- [ ] **Step 6: Commit deployment assets**

```powershell
git add Dockerfile .dockerignore deployment/huggingface/README.md docs/streamlit-deployment.md
git commit -m "docs: add Streamlit Space deployment"
```

### Task 8: Add opt-in real-model smoke coverage

**Files:**

- Modify: `pyproject.toml`
- Create: `tests/integration/test_finetuned_streaming.py`

- [ ] **Step 1: Register the smoke marker**

Add this under `[tool.pytest.ini_options]` in `pyproject.toml`:

```toml
markers = [
    "model_smoke: downloads or loads the real fine-tuned model and runs generation",
]
```

- [ ] **Step 2: Add real extraction and translation smoke tests**

Create `tests/integration/test_finetuned_streaming.py`:

```python
import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import pytest
import torch

from src.controllers.streaming import NewsStreamingController
from src.helpers.config import DEFAULT_STORY_PATH
from src.helpers.huggingface import authenticate_huggingface
from src.models.finetuned_qwen import FineTunedQwenModel
from src.models.news import NewsDetails
from src.models.translation import TranslatedStory


pytestmark = [
    pytest.mark.model_smoke,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_SMOKE") != "1",
        reason="set RUN_MODEL_SMOKE=1 to run the real model",
    ),
]


@pytest.fixture(scope="module")
def controller() -> NewsStreamingController:
    token = authenticate_huggingface()
    runtime = FineTunedQwenModel.load(token)
    return NewsStreamingController(runtime)


@pytest.fixture(scope="module")
def story() -> str:
    return Path(DEFAULT_STORY_PATH).read_text(encoding="utf-8").strip()


def collect_timed_chunks(chunks: Iterable[str], label: str) -> list[str]:
    iterator = iter(chunks)
    started = perf_counter()
    first = next(iterator)
    first_chunk_seconds = perf_counter() - started
    collected = [first, *list(iterator)]
    total_seconds = perf_counter() - started
    peak_vram_gib = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"{label}: first_chunk={first_chunk_seconds:.3f}s "
        f"total={total_seconds:.3f}s peak_vram={peak_vram_gib:.3f}GiB"
    )
    return collected


def test_real_finetuned_extraction_streams_and_validates(
    controller: NewsStreamingController,
    story: str,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    chunks = collect_timed_chunks(
        controller.stream_task("extraction", story), "extraction"
    )
    assert chunks
    result = controller.validate_task_response("extraction", "".join(chunks))
    assert isinstance(result, NewsDetails)


def test_real_finetuned_translation_streams_and_validates(
    controller: NewsStreamingController,
    story: str,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    chunks = collect_timed_chunks(
        controller.stream_task(
            "translation",
            story,
            source_language="Arabic",
            target_language="English",
        ),
        "translation",
    )
    assert chunks
    result = controller.validate_task_response("translation", "".join(chunks))
    assert isinstance(result, TranslatedStory)
```

- [ ] **Step 3: Run the skipped-by-default integration test**

Run:

```powershell
uv run --group app --group test pytest tests/integration/test_finetuned_streaming.py -v -p no:cacheprovider
```

Expected: two tests are collected and skipped, proving the file is importable without loading the model.

- [ ] **Step 4: Run both real model tasks**

Run:

```powershell
$env:RUN_MODEL_SMOKE='1'
uv run --group app --group test pytest tests/integration/test_finetuned_streaming.py -v -s -p no:cacheprovider
Remove-Item Env:RUN_MODEL_SMOKE
```

Expected: the model loads once, both tasks emit at least one real chunk, and both final responses pass their Pydantic schemas. Record device, peak VRAM if CUDA is available, model-load time, extraction time-to-first-chunk, and translation time-to-first-chunk in the final handoff.

- [ ] **Step 5: Commit smoke coverage**

```powershell
git add pyproject.toml tests/integration/test_finetuned_streaming.py
git commit -m "test: smoke streaming news tasks"
```

### Task 9: Full regression, visual QA, and final handoff

**Files:**

- Inspect: all files changed in Tasks 1-8
- Preserve: `README.md`, `configs/llamafactory/news_finetune.yaml`, `tests/load/locustfile.py`

- [ ] **Step 1: Run all fast automated tests**

Run:

```powershell
uv run --group app --group serve --group test pytest tests/unit -q -p no:cacheprovider
```

Expected: all unit tests pass with exit code 0.

- [ ] **Step 2: Compile all Python paths**

Run:

```powershell
uv run --group app --group serve python -m compileall -q app.py src tests
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 3: Verify existing vLLM and Locust imports**

Run:

```powershell
uv run --group serve python -m src.workflows.infer_vllm --help
uv run --group serve locust -f tests/load/locustfile.py --list
```

Expected: the inference CLI prints help and Locust lists `CompletionLoadTest` without connecting to a server.

- [ ] **Step 4: Start Streamlit and inspect desktop and narrow layouts**

Run:

```powershell
uv run --group app streamlit run app.py --server.headless=true --server.port=8501
```

Open `http://localhost:8501`, confirm the newsroom header, restrained neutral/red visual system, two-column desktop composition, stacked narrow layout, task switching, language controls, upload behavior, status-before-first-token, genuine live output, raw JSON, structured extraction, structured translation, and readable RTL Arabic. Inspect browser console output for errors.

- [ ] **Step 5: Check the final diff and unrelated changes**

Run:

```powershell
git status --short
git diff --check
git diff --name-only HEAD~8..HEAD
```

Expected: only planned files are in the implementation commits; the user's pre-existing README, LlamaFactory, and Locust edits remain unstaged and unchanged by this work.

- [ ] **Step 6: Produce the final evidence-backed handoff**

Report:

- Direct fine-tuned Transformers streaming is the local default.
- vLLM remains optional and uses the same UI/controller contract.
- Exact unit, integration, compile, health, Docker, and visual checks run, including failures or environmental blockers.
- Local command: `uv run --group app streamlit run app.py`.
- Deployment guide path: `docs/streamlit-deployment.md`.
- No claim that Streamlit can use Hugging Face ZeroGPU; direct interactive deployment needs paid GPU hardware or a separate inference service.
