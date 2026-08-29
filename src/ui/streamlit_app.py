import html
import json
import logging
from collections.abc import Iterator
from typing import Any, cast

import streamlit as st
from pydantic import ValidationError

from src.controllers.streaming import (
    NewsStreamingController,
    TaskName,
    ValidatedTaskResponse,
    build_streaming_controller,
)
from src.controllers.vllm_status import (
    ConnectionState,
    check_vllm_connection,
    connection_state_from_check,
    connection_status_label,
)
from src.helpers.config import (
    FINETUNED_MODEL_DIR,
    QWEN_MODEL_ID,
    VLLM_API_BASE_URL,
    VLLM_LOCAL_API_KEY,
    VLLM_MAX_TOKENS,
    VLLM_MODEL_ID,
    VLLM_TEMPERATURE,
)
from src.helpers.environment import read_optional_setting
from src.models.news import NewsDetails
from src.models.translation import TranslatedStory
from src.ui.input import resolve_story_input
from src.ui.settings import InferenceSettings, ProviderName, load_default_settings
from src.ui.theme import PAGE_CSS


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["Arabic", "English", "French"]
BACKEND_OPTIONS = ("vLLM", "Direct fine-tuned model")
BACKEND_BY_LABEL: dict[str, ProviderName] = {
    "vLLM": "vllm",
    "Direct fine-tuned model": "finetuned",
}


def _settings_cache_key(settings: InferenceSettings) -> tuple[object, ...]:
    return (
        settings.provider,
        settings.base_url,
        settings.api_key,
        settings.model_id,
        settings.temperature,
        settings.max_tokens,
    )


@st.cache_resource(
    show_spinner=False,
    hash_funcs={InferenceSettings: _settings_cache_key},
)
def get_controller(settings: InferenceSettings) -> NewsStreamingController:
    if settings.provider != "vllm":
        return build_streaming_controller(settings.provider)
    return build_streaming_controller(
        settings.provider,
        vllm_base_url=settings.base_url,
        vllm_api_key=settings.api_key,
        vllm_model_id=settings.model_id,
        vllm_temperature=settings.temperature,
        vllm_max_tokens=settings.max_tokens,
    )


def _directional_text(text: str, css_class: str = "auto-direction") -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div class="{css_class}" dir="auto">{escaped}</div>'


def _without_secret(text: str, secret: str) -> str:
    if secret:
        return text.replace(secret, "***")
    return text


def _adapter_source() -> str:
    return read_optional_setting("FINETUNED_ADAPTER_SOURCE") or str(
        FINETUNED_MODEL_DIR
    )


def _vllm_environment_defaults(
    defaults: InferenceSettings,
) -> tuple[str, str, str, float, int]:
    if defaults.provider == "vllm":
        return (
            defaults.base_url,
            defaults.model_id,
            defaults.api_key,
            defaults.temperature,
            defaults.max_tokens,
        )
    return (
        read_optional_setting("VLLM_API_BASE_URL") or VLLM_API_BASE_URL,
        read_optional_setting("VLLM_MODEL_ID") or VLLM_MODEL_ID,
        read_optional_setting("VLLM_API_KEY") or VLLM_LOCAL_API_KEY,
        VLLM_TEMPERATURE,
        VLLM_MAX_TOKENS,
    )


def _runtime_identity(settings: InferenceSettings) -> str:
    if settings.provider == "vllm":
        return f"vLLM · {settings.model_id}"
    return f"Direct fine-tuned model · {_adapter_source()}"


def _connection_state_for(base_url: str, model_id: str) -> ConnectionState:
    stored = st.session_state.get("vllm_connection")
    if not isinstance(stored, dict):
        return "not_checked"
    if stored.get("key") != (base_url.strip(), model_id.strip()):
        return "not_checked"
    status = stored.get("status")
    if status in {"not_checked", "connected", "unavailable"}:
        return status
    return "not_checked"


def _runtime_error_message(
    exc: Exception,
    settings: InferenceSettings,
) -> str:
    message = str(exc).casefold()
    if "out of memory" in message or (
        "cuda" in message and "memory" in message
    ):
        return (
            "The GPU ran out of memory. Try a shorter story or use the "
            "vLLM backend."
        )
    if isinstance(exc, FileNotFoundError) or "adapter" in message:
        return (
            "The fine-tuned adapter could not be loaded. Check "
            "FINETUNED_ADAPTER_SOURCE and HF_TOKEN."
        )
    if (
        settings.provider == "vllm"
        or "vllm" in message
        or "connection" in message
    ):
        endpoint = settings.base_url or "the configured endpoint"
        return _without_secret(
            (
                f"The inference server at {endpoint} is unavailable. "
                "Check the URL, API key, and model readiness."
            ),
            settings.api_key,
        )
    return "Generation failed. Check the server logs for details and try again."


def _validation_error_message(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"The model returned incomplete or invalid JSON: {exc.msg}."
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return (
            "The JSON does not match the expected schema at "
            f"{location}: {first['msg']}."
        )
    return str(exc)


def _render_extraction(result: NewsDetails) -> None:
    st.markdown(
        '<p class="result-label">Extracted story</p>',
        unsafe_allow_html=True,
    )
    st.markdown(_directional_text(result.story_title), unsafe_allow_html=True)
    category_col, keyword_col = st.columns([1, 2])
    category_col.metric(
        "Category",
        result.story_category.replace("_", " ").title(),
    )
    keyword_col.caption("Keywords")
    keyword_col.write(" · ".join(result.story_keywords))
    st.markdown("#### Summary")
    for point in result.story_summary:
        st.markdown(
            _directional_text(f"— {point}"),
            unsafe_allow_html=True,
        )
    st.markdown("#### Entities")
    st.dataframe(
        [entity.model_dump() for entity in result.story_entities],
        use_container_width=True,
        hide_index=True,
    )


def _render_translation(result: TranslatedStory) -> None:
    st.markdown(
        '<p class="result-label">Translation</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        _directional_text(result.translated_title),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="result-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        _directional_text(result.translated_content),
        unsafe_allow_html=True,
    )


def render_validated_result(result: ValidatedTaskResponse) -> None:
    st.markdown('<div class="result-rule"></div>', unsafe_allow_html=True)
    if isinstance(result, NewsDetails):
        _render_extraction(result)
    elif isinstance(result, TranslatedStory):
        _render_translation(result)


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
            status.update(
                label="Streaming response…",
                state="running",
            )
            first_chunk = False
        chunks.append(chunk)
        yield chunk


def _render_previous_result(task: TaskName) -> None:
    previous = st.session_state.get(f"last_result_{task}")
    if not previous:
        st.markdown(
            '<div class="empty-proof">Your streamed model response will '
            "appear here.</div>",
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


def _store_task_result(
    task: TaskName,
    *,
    raw: str,
    validated: ValidatedTaskResponse | None,
    error: str | None,
) -> None:
    st.session_state[f"last_result_{task}"] = {
        "raw": raw,
        "validated": validated,
        "error": error,
    }


def _render_generation_failure(
    task: TaskName,
    raw: str,
    exc: Exception,
    status: Any,
    settings: InferenceSettings,
) -> None:
    LOGGER.exception("Streaming inference failed")
    message = _runtime_error_message(exc, settings)
    status.update(label="Generation failed", state="error")
    st.error(message)
    if raw:
        with st.expander("Partial model response", expanded=True):
            st.code(raw, language="json", wrap_lines=True)
    _store_task_result(task, raw=raw, validated=None, error=message)


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
        _store_task_result(task, raw=raw, validated=None, error=message)
        return

    render_validated_result(validated)
    with st.expander("Raw model response"):
        st.code(raw, language="json", wrap_lines=True)
    _store_task_result(task, raw=raw, validated=validated, error=None)


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
        _render_generation_failure(task, raw, exc, status, settings)
        return

    _validate_render_and_store(controller, task, raw)


def _render_task_tab(task: TaskName, settings: InferenceSettings) -> None:
    input_column, output_column = st.columns([0.95, 1.05], gap="large")
    with input_column:
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
        pasted, uploaded = _render_story_controls(task)
        submitted = st.button(
            (
                "Extract information"
                if task == "extraction"
                else "Translate story"
            ),
            key=f"{task}_submit",
            type="primary",
            use_container_width=True,
        )

    with output_column:
        st.markdown("### Live output")
        st.markdown(
            f'<p class="proof-identity">{html.escape(_runtime_identity(settings))}</p>',
            unsafe_allow_html=True,
        )
        if not submitted:
            _render_previous_result(task)
            return
        try:
            story = resolve_story_input(
                pasted,
                uploaded_name=uploaded.name if uploaded is not None else None,
                uploaded_bytes=(
                    uploaded.getvalue() if uploaded is not None else None
                ),
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


def _probe_vllm_connection(settings: InferenceSettings) -> None:
    error: BaseException | None = None
    try:
        controller = get_controller(settings)
        check_vllm_connection(controller.runtime.client, settings.model_id)
    except Exception as exc:
        error = exc
        LOGGER.exception("vLLM connection check failed")
        st.error(
            _without_secret(
                (
                    f"Unable to verify {settings.model_id} at "
                    f"{settings.base_url}."
                ),
                settings.api_key,
            )
        )
    st.session_state["vllm_connection"] = {
        "key": (settings.base_url.strip(), settings.model_id.strip()),
        "status": connection_state_from_check(error),
    }


def _render_sidebar(defaults: InferenceSettings) -> InferenceSettings:
    with st.sidebar:
        st.markdown("**ArabLLM Inference Studio**")
        backend_label = st.selectbox(
            "Backend",
            BACKEND_OPTIONS,
            index=BACKEND_OPTIONS.index(
                "vLLM" if defaults.provider == "vllm" else "Direct fine-tuned model"
            ),
            key="inference_backend",
        )
        provider = BACKEND_BY_LABEL[backend_label]
        if provider == "finetuned":
            st.caption("Direct fine-tuned model")
            st.caption("Adapter")
            st.caption(_adapter_source())
            st.caption("Base model")
            st.caption(QWEN_MODEL_ID)
            return InferenceSettings.direct()

        base_url, model_id, env_api_key, temperature, max_tokens = (
            _vllm_environment_defaults(defaults)
        )
        base_url = st.text_input(
            "API base URL",
            value=base_url,
            key="vllm_base_url",
        )
        model_id = st.text_input(
            "Model ID",
            value=model_id,
            key="vllm_model_id",
        )
        api_key_override = st.text_input(
            "API key override",
            value="",
            type="password",
            key="vllm_api_key_override",
            help="Leave blank to use VLLM_API_KEY from the environment.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(temperature),
            step=0.01,
            key="vllm_temperature",
        )
        max_tokens = st.slider(
            "Maximum output tokens",
            min_value=1,
            max_value=4096,
            value=int(max_tokens),
            step=1,
            key="vllm_max_tokens",
        )
        try:
            settings = InferenceSettings(
                provider="vllm",
                base_url=base_url,
                api_key=api_key_override.strip() or env_api_key,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as exc:
            st.error(str(exc))
            settings = InferenceSettings(
                provider="vllm",
                base_url=base_url.strip() or VLLM_API_BASE_URL,
                api_key=api_key_override.strip() or env_api_key,
                model_id=model_id.strip() or VLLM_MODEL_ID,
                temperature=min(max(float(temperature), 0.0), 1.0),
                max_tokens=min(max(int(max_tokens), 1), 4096),
            )

        if st.button("Check connection", key="vllm_check_connection"):
            _probe_vllm_connection(settings)

        state = _connection_state_for(settings.base_url, settings.model_id)
        st.caption(
            f"vLLM · {connection_status_label(state)}"
        )
        return settings


def main() -> None:
    st.set_page_config(
        page_title="ArabLLM Inference Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    settings = _render_sidebar(load_default_settings())
    st.title("ArabLLM Inference Studio")
    st.markdown(
        '<p class="deck">Extract structured news details or translate a story. '
        "Output streams from the selected backend as the model generates it.</p>",
        unsafe_allow_html=True,
    )
    extraction_tab, translation_tab = st.tabs(["Extraction", "Translation"])
    with extraction_tab:
        _render_task_tab("extraction", settings)
    with translation_tab:
        _render_task_tab("translation", settings)
