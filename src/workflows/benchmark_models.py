import argparse
import gc
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TextIO

import torch

from src.controllers.benchmark import (
    benchmark_model,
    generate_arabic_prompts,
)
from src.helpers.config import (
    BENCHMARK_MAX_PROMPT_CHARS,
    BENCHMARK_MIN_PROMPT_CHARS,
    BENCHMARK_OUTPUT_DIR,
    BENCHMARK_RANDOM_SEED,
    BENCHMARK_SAMPLE_COUNT,
)
from src.models.benchmark import BenchmarkResult
from src.models.factory import load_language_model
from src.models.language_model import LanguageModel
from src.utils.console import configure_utf8_output


ModelChoice = Literal["qwen", "finetuned", "both"]
REPORT_FILENAMES = {
    "qwen": "qwen-base-benchmark.txt",
    "finetuned": "qwen-finetuned-benchmark.txt",
}


def release_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmarks(
    model_choice: ModelChoice,
    prompts: Sequence[str],
    *,
    model_loader: Callable[[str], LanguageModel] = load_language_model,
    benchmark_runner: Callable[..., BenchmarkResult] = benchmark_model,
    memory_releaser: Callable[[], None] = release_model_memory,
) -> list[BenchmarkResult]:
    providers = (
        ("qwen", "finetuned")
        if model_choice == "both"
        else (model_choice,)
    )
    results: list[BenchmarkResult] = []

    for provider in providers:
        runtime = model_loader(provider)
        try:
            results.append(benchmark_runner(runtime, prompts))
        finally:
            del runtime
            memory_releaser()

    return results


def format_result(result: BenchmarkResult) -> str:
    return "\n".join(
        (
            (
                f"=== {result.provider.upper()} | {result.model_id} | "
                "PERFORMANCE BENCHMARK ==="
            ),
            f"Samples: {result.sample_count}",
            f"Total time: {result.total_seconds:.3f} seconds",
            (
                "Average latency: "
                f"{result.average_latency_seconds:.3f} seconds/request"
            ),
            f"Input tokens: {result.input_tokens}",
            f"Output tokens: {result.output_tokens}",
            f"Total tokens: {result.total_tokens}",
            f"Requests/second: {result.requests_per_second:.3f}",
            f"Tokens/second: {result.tokens_per_second:.3f}",
        )
    )


def print_result(
    result: BenchmarkResult,
    output: TextIO = sys.stdout,
) -> None:
    print(format_result(result), file=output)


def write_report(
    output_dir: Path,
    result: BenchmarkResult,
) -> Path:
    try:
        filename = REPORT_FILENAMES[result.provider]
    except KeyError as exc:
        raise ValueError(
            f"No benchmark report filename for: {result.provider}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    report_path.write_text(
        format_result(result) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_path


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark base and fine-tuned Qwen with reproducible "
            "Arabic prompts."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("qwen", "finetuned", "both"),
        default="both",
        help="Qwen runtime to benchmark (default: both).",
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=BENCHMARK_SAMPLE_COUNT,
        help=f"Number of prompts (default: {BENCHMARK_SAMPLE_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BENCHMARK_RANDOM_SEED,
        help=f"Prompt seed (default: {BENCHMARK_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--min-chars",
        type=positive_integer,
        default=BENCHMARK_MIN_PROMPT_CHARS,
        help=(
            "Minimum Faker prompt length "
            f"(default: {BENCHMARK_MIN_PROMPT_CHARS})."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=positive_integer,
        default=BENCHMARK_MAX_PROMPT_CHARS,
        help=(
            "Maximum Faker prompt length "
            f"(default: {BENCHMARK_MAX_PROMPT_CHARS})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_OUTPUT_DIR,
        help="Directory for UTF-8 benchmark reports.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output(sys.stdout, sys.stderr)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_chars > args.max_chars:
        parser.error("--min-chars cannot exceed --max-chars")

    try:
        prompts = generate_arabic_prompts(
            sample_count=args.samples,
            seed=args.seed,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        results = run_benchmarks(args.model, prompts)
        for result in results:
            print_result(result)
            report_path = write_report(args.output_dir, result)
            print(f"Report saved: {report_path}")
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
