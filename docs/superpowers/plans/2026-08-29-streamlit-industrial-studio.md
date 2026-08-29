# Streamlit Industrial Inference Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current editorial Streamlit layout with an Industrial inference studio that adds a configuration-only sidebar while preserving genuine extraction and translation streaming.

**Architecture:** Introduce an immutable UI settings value object, pass explicit vLLM overrides through the existing controller and model factory, and isolate the server readiness check in a small controller. Keep the Streamlit page as the composition layer and keep prompts, generation, and schema validation in their current modules.

**Tech Stack:** Python 3.12, Streamlit 1.x, OpenAI-compatible vLLM client, Transformers/PEFT, Pydantic, pytest, Streamlit AppTest.

---

## File map

- Create `src/ui/settings.py`: validated, hashable session inference settings and environment-backed defaults.
- Create `src/ui/theme.py`: Industrial Streamlit CSS so page composition is not mixed with a large style string.
- Create `src/controllers/vllm_status.py`: real vLLM model-readiness check with dependency injection for unit tests.
- Modify `src/models/vllm.py`: accept explicit generation settings when constructing the client wrapper.
- Modify `src/models/factory.py`: accept explicit vLLM connection and generation overrides without mutating environment variables.
- Modify `src/controllers/streaming.py`: pass optional runtime overrides through the existing controller builder.
- Modify `src/ui/streamlit_app.py`: render the sidebar, task tabs, responsive workspace, and genuine live proof console.
- Modify `.streamlit/config.toml`: set the matching dark base theme.
- Modify `tests/unit/test_vllm_streaming.py`: cover explicit generation settings.
- Modify `tests/unit/test_streaming_configuration.py`: cover factory/controller override forwarding.
- Create `tests/unit/test_ui_settings.py`: cover defaults and validation.
- Create `tests/unit/test_vllm_status.py`: cover honest connection states.
- Modify `tests/unit/test_streamlit_app.py`: cover the new sidebar and task tabs while preserving lazy loading.
- Modify `docs/streamlit-local.md`: document sidebar overrides and the connection check.

### Task 1: Immutable UI inference settings

**Files:**
- Create: `src/ui/settings.py`
- Create: `tests/unit/test_ui_settings.py`

- [ ] **Step 1: Write failing settings tests**

```python
from src.ui.settings import InferenceSettings, load_default_settings


def test_defaults_use_environment_without_writing_it(monkeypatch):
    monkeypatch.setenv("NEWS_MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_API_BASE_URL", "http://server:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("VLLM_MODEL_ID", "news")
    settings = load_default_settings()
    assert settings == InferenceSettings(
        provider="vllm",
        base_url="http://server:8000/v1",
        api_key="secret",
        model_id="news",
        temperature=0.3,
        max_tokens=1000,
    )


def test_settings_reject_blank_vllm_endpoint():
    with pytest.raises(ValueError, match="API base URL"):
        InferenceSettings(
            provider="vllm",
            base_url=" ",
            api_key="local-vllm",
            model_id="news-lora",
            temperature=0.3,
            max_tokens=1000,
        )


def test_settings_are_hashable_for_streamlit_resource_cache():
    settings = InferenceSettings.direct()
    assert {settings: "cached"}[settings] == "cached"
```

- [ ] **Step 2: Run the new tests and verify they fail**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit/test_ui_settings.py -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: collection fails because `src.ui.settings` does not exist.

- [ ] **Step 3: Implement the settings value object**

```python
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
            raise ValueError("Maximum output tokens must be between 1 and 4096.")

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
        base_url=read_optional_setting("VLLM_API_BASE_URL") or VLLM_API_BASE_URL,
        api_key=read_optional_setting("VLLM_API_KEY") or VLLM_LOCAL_API_KEY,
        model_id=read_optional_setting("VLLM_MODEL_ID") or VLLM_MODEL_ID,
    )
```

- [ ] **Step 4: Run the settings tests**

Run the command from Step 2. Expected: all settings tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/ui/settings.py tests/unit/test_ui_settings.py
git commit -m "feat: add Streamlit inference settings"
```

### Task 2: Explicit vLLM runtime overrides

**Files:**
- Modify: `src/models/vllm.py`
- Modify: `src/models/factory.py`
- Modify: `src/controllers/streaming.py`
- Modify: `tests/unit/test_vllm_streaming.py`
- Modify: `tests/unit/test_streaming_configuration.py`

- [ ] **Step 1: Add failing vLLM construction tests**

```python
def test_load_applies_explicit_generation_settings():
    captured = {}

    def client_factory(**kwargs):
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
```

Add this factory test:

```python
def test_vllm_provider_forwards_explicit_settings():
    captured = {}
    runtime = object()

    def vllm_loader(**kwargs):
        captured.update(kwargs)
        return runtime

    loaded = load_language_model(
        "vllm",
        vllm_loader=vllm_loader,
        vllm_base_url="http://server/v1",
        vllm_api_key="secret",
        vllm_model_id="news",
        vllm_temperature=0.15,
        vllm_max_tokens=640,
    )

    assert loaded is runtime
    assert captured == {
        "base_url": "http://server/v1",
        "api_key": "secret",
        "model_id": "news",
        "temperature": 0.15,
        "max_tokens": 640,
    }
```

- [ ] **Step 2: Run targeted tests and verify signature failures**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit/test_vllm_streaming.py tests/unit/test_streaming_configuration.py -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: failures report unexpected generation or override arguments.

- [ ] **Step 3: Extend `VLLMModel.load`**

```python
@classmethod
def load(
    cls,
    *,
    base_url: str = VLLM_API_BASE_URL,
    api_key: str = VLLM_LOCAL_API_KEY,
    model_id: str = VLLM_MODEL_ID,
    timeout: float = VLLM_REQUEST_TIMEOUT_SECONDS,
    temperature: float = VLLM_TEMPERATURE,
    max_tokens: int = VLLM_MAX_TOKENS,
    client_factory: Callable[..., Any] | None = None,
) -> "VLLMModel":
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    client = client_factory(base_url=base_url, api_key=api_key, timeout=timeout)
    return cls(
        client,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )
```

- [ ] **Step 4: Forward overrides through the factory and controller**

Extend `load_language_model` and `build_streaming_controller` with:

```python
vllm_base_url: str | None = None,
vllm_api_key: str | None = None,
vllm_model_id: str | None = None,
vllm_temperature: float | None = None,
vllm_max_tokens: int | None = None,
```

Resolve explicit values before environment values and configured defaults. Pass
the resolved values to `VLLMModel.load`. Keep non-vLLM paths unchanged. Update
injected test loaders to accept `**kwargs` and assert exact forwarding.

- [ ] **Step 5: Run vLLM, factory, and controller tests**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit/test_vllm_streaming.py tests/unit/test_streaming_configuration.py tests/unit/test_streaming_controller.py -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: all targeted tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- src/models/vllm.py src/models/factory.py src/controllers/streaming.py tests/unit/test_vllm_streaming.py tests/unit/test_streaming_configuration.py tests/unit/test_streaming_controller.py
git commit -m "feat: configure vLLM from Streamlit sessions"
```

### Task 3: Honest vLLM connection status

**Files:**
- Create: `src/controllers/vllm_status.py`
- Create: `tests/unit/test_vllm_status.py`

- [ ] **Step 1: Write failing status tests**

```python
from src.controllers.vllm_status import check_vllm_connection


def test_connection_check_confirms_expected_model():
    client = fake_client_with_models(["news-lora"])
    result = check_vllm_connection(client, "news-lora")
    assert result.served_model_ids == ("news-lora",)
    assert result.connected is True


def test_connection_check_rejects_missing_model():
    client = fake_client_with_models(["base-model"])
    with pytest.raises(RuntimeError, match="news-lora"):
        check_vllm_connection(client, "news-lora")
```

The test file defines `fake_client_with_models` using a small object whose
`models.list().data` items each expose an `id` attribute.

- [ ] **Step 2: Run tests and verify the missing-module failure**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit/test_vllm_status.py -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: collection fails because `vllm_status` does not exist.

- [ ] **Step 3: Implement the status controller**

```python
from dataclasses import dataclass
from typing import Any

from src.controllers.serving import list_served_model_ids


@dataclass(frozen=True)
class VLLMConnectionResult:
    connected: bool
    served_model_ids: tuple[str, ...]


def check_vllm_connection(client: Any, model_id: str) -> VLLMConnectionResult:
    served = tuple(list_served_model_ids(client))
    if model_id not in served:
        raise RuntimeError(
            f"Configured model {model_id!r} is not served; available: {list(served)!r}."
        )
    return VLLMConnectionResult(connected=True, served_model_ids=served)
```

- [ ] **Step 4: Run the status tests**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit/test_vllm_status.py -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- src/controllers/vllm_status.py tests/unit/test_vllm_status.py
git commit -m "feat: check configured vLLM model readiness"
```

### Task 4: Industrial theme and configuration sidebar

**Files:**
- Create: `src/ui/theme.py`
- Modify: `.streamlit/config.toml`
- Modify: `src/ui/streamlit_app.py`
- Modify: `tests/unit/test_streamlit_app.py`

- [ ] **Step 1: Update AppTest expectations before changing the page**

```python
def test_vllm_sidebar_exposes_only_inference_configuration():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    backend = next(item for item in app.sidebar.selectbox if item.label == "Backend")
    backend.set_value("vLLM").run(timeout=10)
    labels = {item.label for item in app.sidebar.text_input}
    assert labels == {"API base URL", "Model ID", "API key override"}
    assert {item.label for item in app.sidebar.slider} == {
        "Temperature", "Maximum output tokens"
    }
    assert "History" not in [tab.label for tab in app.tabs]
    assert "Architecture" not in [tab.label for tab in app.tabs]
```

Also assert that initial rendering does not call `build_streaming_controller`.

- [ ] **Step 2: Run AppTest and verify the new expectations fail**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit/test_streamlit_app.py -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: failures show the old title and missing sidebar controls.

- [ ] **Step 3: Create the Industrial theme module**

Define `PAGE_CSS` in `src/ui/theme.py` using exactly these tokens:

```css
:root {
  --studio-bg: #0b0c0a;
  --studio-panel: #111410;
  --studio-line: #293027;
  --studio-ink: #ecf4ea;
  --studio-muted: #8b9788;
  --studio-signal: #00e676;
}
.stApp, [data-testid="stSidebar"] { background: var(--studio-bg); }
html, body, [class*="st-"] {
  font-family: "Cascadia Mono", "IBM Plex Mono", ui-monospace, monospace;
}
[data-testid="stSidebar"] { border-right: 1px solid var(--studio-line); }
.stButton > button {
  border: 1px solid var(--studio-signal);
  border-radius: 0;
  background: var(--studio-signal);
  color: #000;
  box-shadow: none;
}
[data-testid="stChatMessage"], [data-testid="stFileUploaderDropzone"] {
  border: 1px solid var(--studio-line);
  border-radius: 0;
  background: var(--studio-panel);
}
```

Add responsive rules that preserve 16px inputs and stack content under 900px.
Apply automatic direction to model content. Do not add gradients, glows, shadows,
rounded cards, emoji, or fabricated telemetry.

- [ ] **Step 4: Render the configuration sidebar**

Add `_render_sidebar(defaults: InferenceSettings) -> InferenceSettings`. Use a
vLLM/direct backend selector, conditional vLLM fields, a blank masked override
that falls back internally to the environment API key, and connection status
keyed by `(base_url, model_id)`.

On `Check connection`, reuse the configured vLLM controller, call
`check_vllm_connection(runtime.client, settings.model_id)`, and render exactly
one status: `Not checked`, `Connected`, or `Unavailable`. Never include the API
key in status or error strings.

- [ ] **Step 5: Apply the dark Streamlit base theme**

```toml
[theme]
base = "dark"
primaryColor = "#00e676"
backgroundColor = "#0b0c0a"
secondaryBackgroundColor = "#111410"
textColor = "#ecf4ea"
font = "monospace"
headingFont = "monospace"
baseRadius = "none"
buttonRadius = "none"
showWidgetBorder = true
```

- [ ] **Step 6: Run AppTest**

Run the command from Step 2. Expected: all sidebar and lazy-load tests pass.

- [ ] **Step 7: Commit Task 4**

```powershell
git add -- .streamlit/config.toml src/ui/theme.py src/ui/streamlit_app.py tests/unit/test_streamlit_app.py
git commit -m "feat: add Industrial inference sidebar"
```

### Task 5: Extraction and Translation task tabs

**Files:**
- Modify: `src/ui/streamlit_app.py`
- Modify: `tests/unit/test_streamlit_app.py`

- [ ] **Step 1: Add failing task-tab tests**

```python
def test_page_has_only_extraction_and_translation_tabs():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    assert [tab.label for tab in app.tabs] == ["Extraction", "Translation"]
    assert {area.key for area in app.text_area} == {
        "extraction_story", "translation_story"
    }
```

Add these tests for translation controls and early validation:

```python
def test_translation_tab_has_language_controls():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    assert [box.key for box in app.selectbox if box.label.endswith("language")] == [
        "translation_source_language",
        "translation_target_language",
    ]


def test_empty_extraction_stops_before_controller_loading(monkeypatch):
    def fail_if_loaded(settings):
        raise AssertionError(f"controller loaded for empty input: {settings}")

    monkeypatch.setattr(streamlit_app, "build_streaming_controller", fail_if_loaded)
    streamlit_app.get_controller.clear()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    extract = next(button for button in app.button if button.label == "Extract information")
    extract.click().run(timeout=10)
    assert not app.exception
    assert any("Paste a story" in error.value for error in app.error)
```

- [ ] **Step 2: Run AppTest and verify task-layout failures**

Run the Task 4 AppTest command. Expected: the radio-based layout fails.

- [ ] **Step 3: Split task rendering into focused helpers**

```python
def _render_story_controls(task: TaskName) -> tuple[str, Any]:
    pasted = st.text_area(
        "Story",
        height=260,
        placeholder="Paste the news story…",
        key=f"{task}_story",
    )
    uploaded = st.file_uploader(
        "Or upload a story",
        type=["txt", "md"],
        key=f"{task}_upload",
        help="UTF-8 TXT or Markdown. Pasted text takes precedence.",
    )
    return pasted, uploaded


def _run_streaming_task(
    task: TaskName,
    settings: InferenceSettings,
    story: str,
    *,
    source_language: str,
    target_language: str,
) -> None:
    status = st.status("Preparing inference…", expanded=False)
    chunks: list[str] = []
    try:
        controller = get_controller(settings)
        status.update(label="Waiting for the first token…", state="running")
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
        raw = streamed if isinstance(streamed, str) else "".join(chunks)
        status.update(label="Generation complete", state="complete")
    except Exception as exc:
        raw = "".join(chunks)
        _render_generation_failure(task, raw, exc, status)
        return

    _validate_render_and_store(controller, task, raw)


def _render_task_tab(task: TaskName, settings: InferenceSettings) -> None:
    input_column, output_column = st.columns([0.95, 1.05], gap="large")
    with input_column:
        pasted, uploaded = _render_story_controls(task)
        source_language, target_language = "Arabic", "English"
        if task == "translation":
            language_columns = st.columns(2)
            source_language = language_columns[0].selectbox(
                "Source language",
                LANGUAGES,
                key="translation_source_language",
            )
            target_language = language_columns[1].selectbox(
                "Target language",
                LANGUAGES,
                index=1,
                key="translation_target_language",
            )
        submitted = st.button(
            "Extract information" if task == "extraction" else "Translate story",
            key=f"{task}_submit",
            type="primary",
            use_container_width=True,
        )

    with output_column:
        st.markdown("### Live output")
        if not submitted:
            _render_previous_result(task)
            return
        try:
            story = resolve_story_input(
                pasted,
                uploaded_name=uploaded.name if uploaded is not None else None,
                uploaded_bytes=uploaded.getvalue() if uploaded is not None else None,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        _run_streaming_task(
            task,
            settings,
            story,
            source_language=source_language,
            target_language=target_language,
        )
```

Add these focused error and validation helpers, and update
`_render_previous_result` to read `st.session_state[f"last_result_{task}"]`:

```python
def _render_generation_failure(
    task: TaskName,
    raw: str,
    exc: Exception,
    status: Any,
) -> None:
    LOGGER.exception("Streaming inference failed")
    message = _runtime_error_message(exc)
    status.update(label="Generation failed", state="error")
    st.error(message)
    if raw:
        with st.expander("Partial model response", expanded=True):
            st.code(raw, language="json", wrap_lines=True)
    st.session_state[f"last_result_{task}"] = {
        "raw": raw,
        "validated": None,
        "error": message,
    }


def _validate_render_and_store(
    controller: NewsStreamingController,
    task: TaskName,
    raw: str,
) -> None:
    try:
        validated = controller.validate_task_response(task, raw)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        message = _validation_error_message(exc)
        st.warning(message)
        with st.expander("Raw model response", expanded=True):
            st.code(raw, language="json", wrap_lines=True)
        st.session_state[f"last_result_{task}"] = {
            "raw": raw,
            "validated": None,
            "error": message,
        }
        return

    render_validated_result(validated)
    with st.expander("Raw model response"):
        st.code(raw, language="json", wrap_lines=True)
    st.session_state[f"last_result_{task}"] = {
        "raw": raw,
        "validated": validated,
        "error": None,
    }
```

Use unique widget keys. Keep the existing order: validate input, resolve the
cached controller, stream once through `st.write_stream`, join those same chunks,
validate once, render the structured result, and store a task-specific previous
result. Preserve partial output and friendly error mapping on failure.

- [ ] **Step 4: Build task-specific cached controllers**

```python
@st.cache_resource(show_spinner=False)
def get_controller(settings: InferenceSettings) -> NewsStreamingController:
    return build_streaming_controller(
        settings.provider,
        vllm_base_url=settings.base_url or None,
        vllm_api_key=settings.api_key or None,
        vllm_model_id=settings.model_id or None,
        vllm_temperature=settings.temperature,
        vllm_max_tokens=settings.max_tokens,
    )
```

`InferenceSettings` is frozen, so it is a stable cache key. Direct settings are
constant, so task switching cannot reload model weights.

- [ ] **Step 5: Run Streamlit and controller unit tests**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit/test_streamlit_app.py tests/unit/test_streaming_controller.py tests/unit/test_ui_input.py -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add -- src/ui/streamlit_app.py tests/unit/test_streamlit_app.py
git commit -m "feat: add streamed inference task tabs"
```

### Task 6: Documentation and final verification

**Files:**
- Modify: `docs/streamlit-local.md`

- [ ] **Step 1: Document sidebar behavior**

Explain that environment variables provide defaults, sidebar overrides last for
the browser session only, `Check connection` verifies the served model, and
direct inference keeps deterministic generation defaults. Preserve the existing
two-terminal WSL vLLM and Windows Streamlit commands.

- [ ] **Step 2: Run the complete unit suite**

```powershell
uv run --cache-dir .uv-cache --no-sync pytest tests/unit -q -o cache_dir=.uv-cache/pytest-cache
```

Expected: every unit test passes.

- [ ] **Step 3: Run static and compatibility checks**

```powershell
uv run --cache-dir .uv-cache --no-sync python -m compileall -q app.py src tests
uv run --cache-dir .uv-cache --no-sync locust -f tests/load/locustfile.py --list
uv run --cache-dir .uv-cache --group serve python -m src.workflows.check_vllm
```

Expected: compilation exits zero; Locust lists `CompletionLoadTest`; the vLLM
check either reports `news-lora` ready or reports the external WSL server as
unavailable without changing application code.

- [ ] **Step 4: Run the Streamlit health check**

Start the app:

```powershell
uv run --cache-dir .uv-cache --group app streamlit run app.py --server.headless=true --server.port=8502 --server.address=127.0.0.1
```

Probe it from another terminal:

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8502/_stcore/health).Content
```

Expected: `ok`.

- [ ] **Step 5: Inspect the actual UI in a browser**

Open `http://127.0.0.1:8502`, inspect the sidebar and both task tabs at desktop
and 390px mobile widths, submit an empty task to verify inline validation, and
check browser logs. If WSL vLLM is ready, submit one short input for each task
and verify text appears before completion.

- [ ] **Step 6: Commit documentation and verified corrections**

```powershell
git add -f -- docs/streamlit-local.md
git add -- src tests .streamlit/config.toml
git commit -m "docs: explain the Streamlit inference studio"
```

- [ ] **Step 7: Confirm repository scope**

```powershell
git status --short --branch
git diff HEAD~1 --stat
```

Expected: only the user's pre-existing `README.md`, training configuration, and
Locust edits remain uncommitted; no Docker or unrelated changes are introduced.
