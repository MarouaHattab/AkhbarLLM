import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from src.controllers.evaluation import evaluate_extraction, evaluate_translation
from src.controllers.serving import wait_for_served_model
from src.helpers import config
from src.models.evaluation import EvaluationResult
from src.models.vllm import VLLMModel
from src.utils.console import configure_utf8_output
from src.utils.evaluation_report import print_result, write_report
from src.utils.story_loader import load_story


InferenceTask = Literal["extraction", "translation", "both"]


def run_vllm_inference(
    task: InferenceTask,
    story_path: str | Path,
    source_language: str,
    target_language: str,
    model_loader: Callable[[], VLLMModel] = VLLMModel.load,
    readiness_checker: Callable[..., list[str]] = wait_for_served_model,
    story_loader: Callable[[str | Path], str] = load_story,
    extraction_evaluator: Callable[..., EvaluationResult] = evaluate_extraction,
    translation_evaluator: Callable[..., EvaluationResult] = evaluate_translation,
) -> list[EvaluationResult]:
    if task not in ("extraction", "translation", "both"):
        raise ValueError(f"Unsupported inference task: {task!r}")

    runtime = model_loader()
    readiness_checker(
        runtime.client,
        runtime.model_id,
        attempts=config.VLLM_READINESS_ATTEMPTS,
        interval_seconds=config.VLLM_READINESS_INTERVAL_SECONDS,
    )
    story = story_loader(story_path)

    results: list[EvaluationResult] = []
    if task in {"extraction", "both"}:
        results.append(extraction_evaluator(runtime, story))
    if task in {"translation", "both"}:
        results.append(
            translation_evaluator(
                runtime,
                story,
                source_language=source_language,
                target_language=target_language,
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run news extraction and translation evaluation through vLLM."
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
        default=config.DEFAULT_STORY_PATH,
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
        results = run_vllm_inference(
            task=args.task,
            story_path=args.story,
            source_language=args.source_language,
            target_language=args.target_language,
        )
        for result in results:
            print_result(result)
        if args.output is not None:
            write_report(args.output, results)
    except Exception as exc:
        print(f"vLLM inference failed: {exc}", file=sys.stderr)
        return 1

    return 0 if all(result.schema_valid for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
