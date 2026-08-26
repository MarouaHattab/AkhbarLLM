import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from src.controllers.evaluation import (
    evaluate_extraction,
    evaluate_translation,
)
from src.helpers.config import DEFAULT_STORY_PATH
from src.helpers.huggingface import authenticate_huggingface
from src.models.evaluation import EvaluationResult
from src.models.qwen import QwenRuntime
from src.utils.story_loader import load_story


EvaluationTask = Literal["extraction", "translation", "both"]


def run_evaluations(
    task: EvaluationTask,
    story_path: str | Path,
    source_language: str,
    target_language: str,
    authenticate=authenticate_huggingface,
    runtime_loader=QwenRuntime.load,
    story_loader=load_story,
    extraction_evaluator=evaluate_extraction,
    translation_evaluator=evaluate_translation,
) -> list[EvaluationResult]:

    story = story_loader(story_path)
    token = authenticate()
    runtime = runtime_loader(token)

    results: list[EvaluationResult] = []
    if task in {"extraction", "both"}:
        results.append(extraction_evaluator(runtime, story))

    if task in {"translation", "both"}:
        results.append(
            translation_evaluator(
                runtime,
                story,
                target_language=target_language,
                source_language=source_language,
            )
        )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the base Qwen model before fine-tuning.",
    )
    parser.add_argument(
        "--task",
        choices=("extraction", "translation", "both"),
        default="both",
        help="Evaluation task to run (default: both).",
    )
    parser.add_argument(
        "--story",
        type=Path,
        default=DEFAULT_STORY_PATH,
        help="UTF-8 story file to evaluate.",
    )
    parser.add_argument(
        "--source-language",
        default="Arabic",
        help="Source language for translation (default: Arabic).",
    )
    parser.add_argument(
        "--target-language",
        default="English",
        help="Target language for translation (default: English).",
    )
    return parser


def print_result(result: EvaluationResult) -> None:
    print(f"\n=== {result.task.upper()} ===")
    print(result.raw_response)
    print(f"JSON valid: {result.json_valid}")
    print(f"Schema valid: {result.schema_valid}")
    if result.validation_error:
        print(f"Validation error: {result.validation_error}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = run_evaluations(
            task=args.task,
            story_path=args.story,
            source_language=args.source_language,
            target_language=args.target_language,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
