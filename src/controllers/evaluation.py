import json
from typing import Literal

from pydantic import BaseModel, ValidationError

from src.models.evaluation import EvaluationResult
from src.models.news import NewsDetails
from src.models.qwen import QwenRuntime
from src.models.translation import TranslatedStory
from src.tasks import build_extraction_messages, build_translation_messages


def _validate_response(
    task: Literal["extraction", "translation"],
    raw_response: str,
    schema: type[BaseModel],
) -> EvaluationResult:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return EvaluationResult(
            task=task,
            raw_response=raw_response,
            json_valid=False,
            schema_valid=False,
            validation_error=str(exc),
        )

    try:
        schema.model_validate(payload)
    except ValidationError as exc:
        return EvaluationResult(
            task=task,
            raw_response=raw_response,
            json_valid=True,
            schema_valid=False,
            validation_error=str(exc),
        )

    return EvaluationResult(
        task=task,
        raw_response=raw_response,
        json_valid=True,
        schema_valid=True,
    )


def evaluate_extraction(
    runtime: QwenRuntime,
    story: str,
) -> EvaluationResult:
    messages = build_extraction_messages(story)
    raw_response = runtime.generate(messages)
    return _validate_response(
        task="extraction",
        raw_response=raw_response,
        schema=NewsDetails,
    )


def evaluate_translation(
    runtime: QwenRuntime,
    story: str,
    target_language: str = "English",
    source_language: str = "Arabic",
) -> EvaluationResult:
    messages = build_translation_messages(
        story,
        target_language=target_language,
        source_language=source_language,
    )
    raw_response = runtime.generate(messages)
    return _validate_response(
        task="translation",
        raw_response=raw_response,
        schema=TranslatedStory,
    )
