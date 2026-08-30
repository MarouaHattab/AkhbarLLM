import html
import json
import logging
import time
from collections.abc import Iterator
from typing import Any, cast

import streamlit as st
from pydantic import ValidationError

from src.controllers.streaming import (
    NewsStreamingController,
    TaskName,
    ValidatedTaskResponse,
    build_streaming_controller,
    load_model_json,
)
from src.controllers.serving import list_served_model_ids
from src.controllers.vllm_status import (
    ConnectionState,
    connection_state_from_check,
)
from src.helpers.config import VLLM_API_BASE_URL, VLLM_MODEL_ID
from src.models.news import NewsDetails
from src.models.translation import TranslatedStory
from src.models.vllm import VLLMModel
from src.ui.input import resolve_story_input
from src.ui.settings import InferenceSettings, load_vllm_defaults
from src.ui.theme import PAGE_CSS


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["Arabic", "English", "French"]
SAMPLE_STORY = (
    "ذكرت مجلة فوربس أن العائلة تلعب دورا محوريا في تشكيل علاقة الأفراد بالمال، "
    "حيث تتأثر هذه العلاقة بأنماط السلوك المالي المتوارثة عبر الأجيال. "
    "التقرير يستند إلى أبحاث الأستاذ الجامعي شاين إنيت حول الرفاه المالي."
)


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
    return build_streaming_controller(
        "vllm",
        vllm_base_url=settings.base_url,
        vllm_api_key=settings.api_key,
        vllm_model_id=settings.model_id,
        vllm_temperature=settings.temperature,
        vllm_max_tokens=settings.max_tokens,
    )


def _pretty_json(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            payload = load_model_json(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    if not isinstance(payload, (dict, list)):
        return raw
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def _directional_text(text: str, css_class: str = "auto-direction") -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div class="{css_class}" dir="auto">{escaped}</div>'


def _without_secret(text: str, secret: str) -> str:
    if secret:
        return text.replace(secret, "***")
    return text


def _status_badge_html(state: ConnectionState) -> str:
    if state == "connected":
        return (
            '<span class="status-online"><span class="dot"></span> '
            "vLLM Online</span>"
        )
    if state == "unavailable":
        return (
            '<span class="status-offline"><span class="dot"></span> '
            "vLLM unavailable</span>"
        )
    return (
        '<span class="status-idle"><span class="dot"></span> '
        "vLLM not checked</span>"
    )


def _connection_state_for(base_url: str) -> ConnectionState:
    stored = st.session_state.get("vllm_connection")
    if not isinstance(stored, dict):
        return "not_checked"
    if stored.get("key") != base_url.strip():
        return "not_checked"
    status = stored.get("status")
    if status in {"not_checked", "connected", "unavailable"}:
        return status
    return "not_checked"


def _exception_text(exc: BaseException) -> str:
    parts = [str(exc)]
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        parts.append(str(cause))
    return " ".join(parts).casefold()


def _runtime_error_message(
    exc: Exception,
    settings: InferenceSettings,
) -> str:
    text = _exception_text(exc)
    if "maximum context length" in text or "max context length" in text:
        return (
            "This request exceeds the model's 2048-token context window. "
            "Lower Max tokens in the sidebar or shorten the story."
        )
    if "out of memory" in text or (
        "cuda" in text and "memory" in text
    ):
        return "The GPU ran out of memory. Try a shorter story."
    if any(
        token in text
        for token in (
            "connection refused",
            "connection error",
            "connecterror",
            "unreachable",
            "unavailable",
            "timed out",
            "timeout",
            "name or service not known",
        )
    ):
        endpoint = settings.base_url or "the configured endpoint"
        return _without_secret(
            (
                f"The inference server at {endpoint} is unavailable. "
                "Check the base URL and that vLLM is running."
            ),
            settings.api_key,
        )
    return "Generation failed. Check the server logs for details and try again."


def _validation_error_message(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        if "expecting value" in exc.msg.casefold() or "unterminated" in exc.msg.casefold():
            return (
                "The model stopped before finishing the JSON. "
                "Try a shorter story."
            )
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
    chips = "".join(
        f'<span class="keyword-chip">{html.escape(keyword)}</span>'
        for keyword in result.story_keywords
    )
    summary_items = "".join(
        f'<li><span class="auto-direction">{html.escape(point)}</span></li>'
        for point in result.story_summary
    )
    entity_rows = "".join(
        (
            "<tr>"
            f'<td class="auto-direction">{html.escape(entity.entity_value)}</td>'
            f'<td><span class="type-chip">{html.escape(entity.entity_type)}</span></td>'
            "</tr>"
        )
        for entity in result.story_entities
    )
    category = result.story_category.replace("_", " ").title()
    st.markdown(
        f"""
<div class="result-card">
  <p class="result-label">Extracted story</p>
  <h3 class="story-title auto-direction">{html.escape(result.story_title)}</h3>
  <div class="meta-grid">
    <div class="meta-box">
      <span class="meta-label">Category</span>
      <span class="meta-value">{html.escape(category)}</span>
    </div>
    <div class="meta-box">
      <span class="meta-label">Keywords</span>
      <div class="chip-row">{chips}</div>
    </div>
  </div>
  <p class="section-heading">Summary</p>
  <ol class="summary-list">{summary_items}</ol>
  <p class="section-heading">Entities</p>
  <table class="entity-table">
    <thead>
      <tr><th>Entity</th><th>Type</th></tr>
    </thead>
    <tbody>{entity_rows}</tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_translation(result: TranslatedStory) -> None:
    st.markdown(
        f"""
<div class="result-card">
  <p class="result-label">Translated story</p>
  <h3 class="story-title auto-direction">{html.escape(result.translated_title)}</h3>
  <p class="section-heading">Translated text</p>
  {_directional_text(result.translated_content, "story-body auto-direction")}
</div>
""",
        unsafe_allow_html=True,
    )


def render_validated_result(result: ValidatedTaskResponse) -> None:
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


def _ensure_session_counters() -> None:
    st.session_state.setdefault("inference_history", [])
    st.session_state.setdefault("total_requests", 0)
    st.session_state.setdefault("total_latency_ms", 0)
    st.session_state.setdefault("total_input_tokens", 0)
    st.session_state.setdefault("total_output_tokens", 0)


def _render_previous_result(task: TaskName) -> None:
    previous = st.session_state.get(f"last_result_{task}")
    if not previous:
        hint = (
            "Paste Arabic text and click Extract Details."
            if task == "extraction"
            else "Paste text and click Translate."
        )
        st.markdown(
            f'<div class="output-box empty-proof">{html.escape(hint)}</div>',
            unsafe_allow_html=True,
        )
        return
    raw = cast(str, previous.get("raw", ""))
    error = previous.get("error")
    validated = previous.get("validated")
    preview = str(previous.get("preview", ""))
    if error:
        st.error(str(error))
    if validated is not None:
        render_validated_result(cast(ValidatedTaskResponse, validated))
    if raw or validated is not None:
        _render_history_output(
            raw,
            task=task,
            preview=preview,
            validated=cast(ValidatedTaskResponse | None, validated),
            expanded=True,
            index=1,
        )


def _story_preview(story: str) -> str:
    preview = story.strip().replace("\n", " ")
    if len(preview) > 80:
        return preview[:80] + "…"
    return preview


def _json_text(
    raw: str,
    validated: ValidatedTaskResponse | None = None,
) -> str:
    if validated is not None:
        return json.dumps(validated.model_dump(), ensure_ascii=False, indent=2)
    return _pretty_json(raw)


def _render_structured_json(
    raw: str,
    validated: ValidatedTaskResponse | None = None,
) -> None:
    st.code(_json_text(raw, validated), language="json", wrap_lines=True)


def _render_history_output(
    raw: str,
    *,
    task: str,
    preview: str,
    validated: ValidatedTaskResponse | None = None,
    error: str | None = None,
    expanded: bool = False,
    index: int | None = None,
) -> None:
    label = f"{task} · {preview}" if preview else task
    if index is not None:
        label = f"{index} · {label}"
    with st.expander(label, expanded=expanded):
        if error:
            st.error(str(error))
        _render_structured_json(raw, validated)


def _store_task_result(
    task: TaskName,
    *,
    story: str,
    raw: str,
    validated: ValidatedTaskResponse | None,
    error: str | None,
    latency_ms: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    _ensure_session_counters()
    preview = _story_preview(story)
    st.session_state[f"last_result_{task}"] = {
        "raw": raw,
        "validated": validated,
        "error": error,
        "preview": preview,
    }
    st.session_state["total_requests"] = (
        int(st.session_state["total_requests"]) + 1
    )
    st.session_state["total_latency_ms"] = (
        int(st.session_state["total_latency_ms"]) + latency_ms
    )
    st.session_state["total_input_tokens"] = (
        int(st.session_state["total_input_tokens"]) + input_tokens
    )
    st.session_state["total_output_tokens"] = (
        int(st.session_state["total_output_tokens"]) + output_tokens
    )
    st.session_state["inference_history"].append(
        {
            "task": task,
            "preview": preview,
            "raw": _json_text(raw, validated),
            "error": error,
            "latency_ms": latency_ms,
        }
    )


def _render_generation_failure(
    task: TaskName,
    story: str,
    raw: str,
    exc: Exception,
    status: Any,
    settings: InferenceSettings,
    *,
    latency_ms: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    LOGGER.exception("Streaming inference failed")
    message = _runtime_error_message(exc, settings)
    status.update(label="Generation failed", state="error")
    st.error(message)
    if raw:
        _render_history_output(
            raw,
            task=task,
            preview=_story_preview(story),
            expanded=True,
            index=1,
        )
    _store_task_result(
        task,
        story=story,
        raw=raw,
        validated=None,
        error=message,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _validate_render_and_store(
    controller: NewsStreamingController,
    task: TaskName,
    story: str,
    raw: str,
    *,
    latency_ms: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    try:
        validated = controller.validate_task_response(task, raw)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        message = _validation_error_message(exc)
        st.warning(message)
        _render_history_output(
            raw,
            task=task,
            preview=_story_preview(story),
            expanded=True,
            index=1,
        )
        _store_task_result(
            task,
            story=story,
            raw=raw,
            validated=None,
            error=message,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return

    render_validated_result(validated)
    _render_history_output(
        raw,
        task=task,
        preview=_story_preview(story),
        validated=validated,
        expanded=True,
        index=1,
    )
    _store_task_result(
        task,
        story=story,
        raw=raw,
        validated=validated,
        error=None,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _render_story_controls(task: TaskName) -> str:
    sample_col, clear_col = st.columns(2)
    if sample_col.button("📄 Sample", key=f"{task}_load_sample"):
        st.session_state[f"{task}_story"] = SAMPLE_STORY
    if clear_col.button("🗑️ Clear", key=f"{task}_clear"):
        st.session_state[f"{task}_story"] = ""
    return st.text_area(
        "Story",
        height=260,
        placeholder="Paste the news story…",
        key=f"{task}_story",
        label_visibility="collapsed",
    )


def _usage_from(story: str, raw: str, started: float) -> dict[str, int]:
    return {
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "input_tokens": _estimate_tokens(story),
        "output_tokens": _estimate_tokens(raw),
    }


def _run_streaming_task(
    task: TaskName,
    settings: InferenceSettings,
    story: str,
    *,
    source_language: str,
    target_language: str,
    stream_output: bool,
) -> None:
    status = st.status("Preparing inference…", expanded=False)
    chunks: list[str] = []
    started = time.perf_counter()
    live_box = st.empty()
    try:
        controller = get_controller(settings)
        status.update(label="Waiting for the first token…", state="running")
        iterator = _stream_with_status(
            controller,
            task,
            story,
            source_language=source_language,
            target_language=target_language,
            status=status,
            chunks=chunks,
        )
        if stream_output:
            with live_box.container():
                st.markdown(
                    '<p class="live-label">Live output</p>',
                    unsafe_allow_html=True,
                )
                streamed = st.write_stream(iterator)
            raw = streamed if isinstance(streamed, str) else "".join(chunks)
        else:
            raw = "".join(iterator)
        status.update(label="Generation complete", state="complete")
    except Exception as exc:
        raw = "".join(chunks)
        live_box.empty()
        _render_generation_failure(
            task,
            story,
            raw,
            exc,
            status,
            settings,
            **_usage_from(story, raw, started),
        )
        st.rerun()
        return

    live_box.empty()
    _validate_render_and_store(
        controller,
        task,
        story,
        raw,
        **_usage_from(story, raw, started),
    )
    st.rerun()


def _render_task_tab(
    task: TaskName,
    settings: InferenceSettings,
    *,
    stream_output: bool,
) -> None:
    input_column, output_column = st.columns([1, 1], gap="large")
    with input_column:
        st.markdown(
            "### Arabic Input" if task == "extraction" else "### Translation"
        )
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
        pasted = _render_story_controls(task)
        submitted = st.button(
            "🔍 Extract Details" if task == "extraction" else "🌍 Translate",
            key=f"{task}_submit",
            type="primary",
            width="stretch",
        )

    with output_column:
        st.markdown("### Model Output")
        if not submitted:
            _render_previous_result(task)
            return
        try:
            story = resolve_story_input(
                pasted,
                uploaded_name=None,
                uploaded_bytes=None,
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
            stream_output=stream_output,
        )


def _render_history_tab() -> None:
    _ensure_session_counters()
    history = list(st.session_state.get("inference_history", []))
    if not history:
        st.info("No inferences yet — run a task first.")
        return
    for index, item in enumerate(reversed(history), start=1):
        _render_history_output(
            str(item.get("raw", "")),
            task=str(item.get("task", "task")),
            preview=str(item.get("preview", "")),
            error=str(item["error"]) if item.get("error") else None,
            expanded=False,
            index=index,
        )


def _probe_vllm_connection(settings: InferenceSettings) -> None:
    error: BaseException | None = None
    detail = ""
    try:
        runtime = VLLMModel.load(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model_id=settings.model_id,
            timeout=8.0,
        )
        served = list_served_model_ids(runtime.client)
        detail = ", ".join(served) if served else "reachable"
    except Exception as exc:
        error = exc
        LOGGER.exception("vLLM connection check failed")
        detail = _without_secret(str(exc), settings.api_key)
    st.session_state["vllm_connection"] = {
        "key": settings.base_url.strip(),
        "status": connection_state_from_check(error),
        "detail": detail,
    }


def _render_sidebar(defaults: InferenceSettings) -> tuple[InferenceSettings, bool]:
    _ensure_session_counters()
    with st.sidebar:
        st.markdown("## AkhbarLLM")
        st.divider()
        st.markdown("### Connection")
        base_url = st.text_input(
            "Base URL",
            value=defaults.base_url,
            key="vllm_base_url",
        )
        try:
            settings = InferenceSettings(
                provider="vllm",
                base_url=base_url,
                api_key=defaults.api_key,
                model_id=defaults.model_id or VLLM_MODEL_ID,
                temperature=defaults.temperature,
                max_tokens=defaults.max_tokens,
            )
        except ValueError as exc:
            st.error(str(exc))
            settings = InferenceSettings(
                provider="vllm",
                base_url=base_url.strip() or VLLM_API_BASE_URL,
                api_key=defaults.api_key,
                model_id=defaults.model_id or VLLM_MODEL_ID,
                temperature=defaults.temperature,
                max_tokens=defaults.max_tokens,
            )
        if st.button("🔌 Check connection", key="vllm_check_connection"):
            _probe_vllm_connection(settings)
            st.rerun()
        preview_url = str(
            st.session_state.get("vllm_base_url") or defaults.base_url
        )
        st.markdown(
            _status_badge_html(_connection_state_for(preview_url)),
            unsafe_allow_html=True,
        )
        stored = st.session_state.get("vllm_connection")
        if isinstance(stored, dict) and stored.get("key") == preview_url.strip():
            detail = str(stored.get("detail") or "").strip()
            if detail:
                st.caption(detail)

        st.markdown("### Configuration")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults.temperature),
            step=0.05,
            key="vllm_temperature",
        )
        slider_max = 1536
        stored_tokens = int(st.session_state.get("vllm_max_tokens") or 0)
        if stored_tokens > slider_max:
            st.session_state["vllm_max_tokens"] = slider_max
        snapped_tokens = 64 * max(
            1,
            min(slider_max // 64, round(int(defaults.max_tokens) / 64)),
        )
        max_tokens = st.slider(
            "Max tokens",
            min_value=64,
            max_value=slider_max,
            value=min(snapped_tokens, slider_max),
            step=64,
            key="vllm_max_tokens",
            help="Kept below the model's 2048-token context window.",
        )
        stream_output = st.toggle(
            "Stream output",
            value=True,
            key="vllm_stream_output",
        )
        settings = InferenceSettings(
            provider="vllm",
            base_url=settings.base_url,
            api_key=settings.api_key,
            model_id=settings.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        st.divider()
        st.markdown("### Stats")
        requests = int(st.session_state["total_requests"])
        latency = int(st.session_state["total_latency_ms"])
        avg_latency = int(latency / requests) if requests else 0
        st.metric("Requests", requests)
        st.metric("Avg latency", f"{avg_latency} ms")
        st.metric("Input tokens", int(st.session_state["total_input_tokens"]))
        st.metric("Output tokens", int(st.session_state["total_output_tokens"]))
        return settings, stream_output


def main() -> None:
    st.set_page_config(
        page_title="AkhbarLLM",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    settings, stream_output = _render_sidebar(load_vllm_defaults())
    st.title("AkhbarLLM")
    st.markdown(
        '<p class="deck">Extract structure from Arabic news, then translate it.</p>',
        unsafe_allow_html=True,
    )
    extraction_tab, translation_tab, history_tab = st.tabs(
        ["Details Extraction", "Translation", "History"]
    )
    with extraction_tab:
        _render_task_tab(
            "extraction",
            settings,
            stream_output=stream_output,
        )
    with translation_tab:
        _render_task_tab(
            "translation",
            settings,
            stream_output=stream_output,
        )
    with history_tab:
        _render_history_tab()
