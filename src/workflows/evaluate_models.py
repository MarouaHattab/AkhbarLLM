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
from src.models.evaluation import EvaluationResult
from src.models.factory import load_language_model
from src.utils.console import configure_utf8_output
from src.utils.evaluation_report import (
    format_response_body,
    format_result,
    print_result,
    write_report,
)
from src.utils.story_loader import load_story


EvaluationTask = Literal["extraction", "translation", "both"]
ModelChoice = Literal[
    "qwen",
    "finetuned",
    "gemini",
    "openai",
    "all",
]


def run_evaluations(
    model_choice: ModelChoice,
    task: EvaluationTask,
    story_path: str | Path,
    source_language: str,
    target_language: str,
    model_loader=load_language_model,
    story_loader=load_story,
    extraction_evaluator=evaluate_extraction,
    translation_evaluator=evaluate_translation,
) -> list[EvaluationResult]:
    story = story_loader(story_path)
    results: list[EvaluationResult] = []
    providers = (
        ("qwen", "finetuned", "gemini", "openai")
        if model_choice == "all"
        else (model_choice,)
    )
    for provider in providers:
        runtime = model_loader(provider)
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
        description=(
            "Evaluate Qwen before or after fine-tuning, Gemini, and OpenAI."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("qwen", "finetuned", "gemini", "openai", "all"),
        default="qwen",
        help="Model provider to evaluate (default: qwen).",
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
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write the evaluation report to this UTF-8 text file. "
            "Use this on Windows instead of piping to Set-Content."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output(sys.stdout, sys.stderr)
    args = build_parser().parse_args(argv)
    try:
        results = run_evaluations(
            model_choice=args.model,
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

    if args.output is not None:
        write_report(args.output, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
