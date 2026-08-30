import ast
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal, cast, get_args

from pydantic import ValidationError

from src.helpers.environment import read_optional_setting
from src.models.factory import load_language_model
from src.models.language_model import (
    LanguageModel,
    StreamingLanguageModel,
)
from src.models.news import EntityType, NewsDetails, StoryCategory
from src.models.translation import TranslatedStory
from src.tasks import build_extraction_messages, build_translation_messages


TaskName = Literal["extraction", "translation"]
ValidatedTaskResponse = NewsDetails | TranslatedStory


def _strip_markdown_fence(raw_response: str) -> str:
    text = raw_response.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def load_model_json(raw_response: str) -> object:
    text = _strip_markdown_fence(raw_response)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        if start == -1:
            raise exc
        snippet = text[start:]
        try:
            payload, _ = json.JSONDecoder().raw_decode(snippet)
            return payload
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_close_truncated_json(snippet))
        except json.JSONDecodeError:
            raise exc


def _json_closers(text: str) -> tuple[bool, list[str]]:
    in_string = False
    escape = False
    stack: list[str] = []
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]" and stack:
            stack.pop()
    return in_string, stack


def _close_truncated_json(text: str) -> str:
    repaired = text.rstrip()
    in_string, _ = _json_closers(repaired)
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip()
    repaired = re.sub(r'(,\s*)?"(?:\\.|[^"\\])*"\s*:\s*$', "", repaired)
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    _, stack = _json_closers(repaired)
    return repaired + "".join(reversed(stack))


def _clean_extraction_payload(payload: dict[str, object]) -> dict[str, object]:
    categories = set(get_args(StoryCategory))
    entity_types = set(get_args(EntityType))
    title = str(payload.get("story_title") or "").strip()
    if len(title) > 100:
        title = title[:100]
    keywords = payload.get("story_keywords")
    cleaned_keywords = (
        [str(item).strip() for item in keywords if str(item).strip()]
        if isinstance(keywords, list)
        else []
    )
    summary = payload.get("story_summary")
    cleaned_summary = (
        [str(item).strip() for item in summary if str(item).strip()][:5]
        if isinstance(summary, list)
        else []
    )
    category = payload.get("story_category")
    if category not in categories:
        category = "not_specified"
    entities = payload.get("story_entities")
    cleaned_entities: list[dict[str, str]] = []
    if isinstance(entities, list):
        for item in entities:
            if not isinstance(item, dict):
                continue
            value = str(item.get("entity_value") or "").strip()
            entity_type = str(item.get("entity_type") or "").strip()
            if not value or not entity_type:
                continue
            if entity_type not in entity_types:
                entity_type = "not_specified"
            cleaned_entities.append(
                {"entity_value": value, "entity_type": entity_type}
            )
    return {
        "story_title": title,
        "story_keywords": cleaned_keywords,
        "story_summary": cleaned_summary,
        "story_category": category,
        "story_entities": cleaned_entities,
    }


def news_details_from_response(raw_response: str) -> NewsDetails:
    payload = load_model_json(raw_response)
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Expecting value", raw_response, 0)
    return NewsDetails.model_validate(_clean_extraction_payload(payload))


def _parse_model_object(raw_response: str) -> object:
    try:
        return load_model_json(raw_response)
    except json.JSONDecodeError as exc:
        text = _strip_markdown_fence(raw_response)
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, TypeError):
            raise exc


def _title_from_translated_text(text: str) -> str:
    first_line = text.splitlines()[0].strip() if text.strip() else ""
    title = first_line
    for delimiter in (". ", "! ", "? "):
        index = first_line.find(delimiter)
        if index >= 4:
            title = first_line[: index + 1].strip()
            break
    if len(title) > 300:
        title = title[:297].rstrip() + "…"
    if len(title) < 5:
        return "Translated story"
    return title


def translated_story_from_response(raw_response: str) -> TranslatedStory:
    try:
        payload = _parse_model_object(raw_response)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        try:
            return TranslatedStory.model_validate(payload)
        except ValidationError:
            title = str(payload.get("translated_title") or "").strip()
            content = str(payload.get("translated_content") or "").strip()
            if len(content) >= 5:
                if len(title) < 5:
                    title = _title_from_translated_text(content)
                if len(title) > 300:
                    title = title[:297].rstrip() + "…"
                if len(title) < 5:
                    title = "Translated story"
                return TranslatedStory(
                    translated_title=title,
                    translated_content=content,
                )
            raise

    text = _strip_markdown_fence(raw_response)
    if len(text) < 5:
        raise json.JSONDecodeError("Expecting value", raw_response, 0)
    return TranslatedStory(
        translated_title=_title_from_translated_text(text),
        translated_content=text,
    )


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
        if task == "extraction":
            return news_details_from_response(raw_response)
        if task == "translation":
            return translated_story_from_response(raw_response)
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
