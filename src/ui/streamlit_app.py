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
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stFileUploader"] small,
[data-testid="stCaptionContainer"] p {
  color: var(--ink) !important;
}
[data-testid="stCaptionContainer"] p,
[data-testid="stFileUploader"] small { color: var(--muted) !important; }
.block-container {
  max-width: 1320px;
  padding-top: 2rem;
  padding-bottom: 4rem;
}
h1, h2, h3 { letter-spacing: -0.035em; }
h1 {
  font-size: clamp(2.2rem, 5vw, 4.8rem) !important;
  line-height: .95 !important;
}
.eyebrow {
  color: var(--signal);
  font-size: .75rem;
  font-weight: 750;
  letter-spacing: .14em;
  text-transform: uppercase;
}
.deck { color: var(--muted); max-width: 720px; font-size: 1.02rem; }
.editorial-rule { border-top: 1px solid var(--ink); margin: 1.2rem 0 1.7rem; }
.result-rule { border-top: 1px solid var(--ink); margin: .5rem 0 1rem; }
.result-label {
  color: var(--signal);
  font-size: .7rem;
  font-weight: 750;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.auto-direction {
  direction: auto;
  unicode-bidi: plaintext;
  white-space: pre-wrap;
  line-height: 1.75;
}
.empty-proof {
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  color: var(--muted);
  padding: 2.5rem 0;
}
[data-testid="stFileUploaderDropzone"] {
  background: transparent;
  border-color: var(--rule);
}
[data-testid="stChatMessage"] {
  background: #fff;
  border: 1px solid var(--rule);
  border-radius: 0;
}
.stButton > button {
  border-radius: 0;
  border: 1px solid var(--ink);
  background: var(--ink);
  color: white;
}
.stButton > button:hover {
  border-color: var(--signal);
  background: var(--signal);
  color: white;
}
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
    if "vllm" in message or "connection" in message:
        return (
            "The inference server is unavailable. Check its URL, API key, "
            "and model readiness."
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
    st.markdown(
        '<div class="editorial-rule"></div>',
        unsafe_allow_html=True,
    )
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
                label="Receiving model output…",
                state="running",
            )
            first_chunk = False
        chunks.append(chunk)
        yield chunk


def _render_previous_result(task: TaskName) -> None:
    previous = st.session_state.get("last_result")
    if not previous or previous.get("task") != task:
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


def main() -> None:
    st.set_page_config(
        page_title="ArabLLM Newsroom",
        page_icon="📰",
        layout="wide",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="eyebrow">Fine-tuned Arabic news model</p>',
        unsafe_allow_html=True,
    )
    st.title("ArabLLM Newsroom")
    st.markdown(
        '<p class="deck">Extract structured editorial details or translate '
        "a story while the fine-tuned model writes its response live.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="editorial-rule"></div>',
        unsafe_allow_html=True,
    )

    input_column, result_column = st.columns(
        [0.95, 1.05],
        gap="large",
    )
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
                "Source language",
                LANGUAGES,
                index=0,
            )
            target_language = language_columns[1].selectbox(
                "Target language",
                LANGUAGES,
                index=1,
            )
        submitted = st.button(
            (
                "Extract information"
                if task == "extraction"
                else "Translate story"
            ),
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
                uploaded_name=(
                    uploaded.name if uploaded is not None else None
                ),
                uploaded_bytes=(
                    uploaded.getvalue() if uploaded is not None else None
                ),
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
            raw_response = (
                streamed if isinstance(streamed, str) else "".join(chunks)
            )
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
            validated = controller.validate_task_response(
                task,
                raw_response,
            )
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
