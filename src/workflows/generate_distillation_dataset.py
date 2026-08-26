import argparse
import json
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.helpers.config import (
    DISTILLATION_OUTPUT_PATH,
    DISTILLATION_REPORT_INTERVAL,
    DISTILLATION_RESPONSE_FORMAT,
    DISTILLATION_SHUFFLE_SEED,
    DISTILLATION_TEACHER_MODEL_ID,
    RAW_DATA_PATH,
)
from src.helpers.environment import require_api_key
from src.models.distillation import (
    DistillationStats,
    TokenUsage,
    calculate_distillation_cost,
)
from src.models.news import NewsDetails
from src.models.openai import OpenAIModel
from src.tasks.extraction import build_extraction_messages
from src.utils.jsonl import append_jsonl, load_jsonl, reset_jsonl


DISTILLATION_TASK = (
    "Extract the story details into a JSON object "
    "according to the provided schema."
)


def _for_reasoning_model(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            **message,
            "role": "developer" if message["role"] == "system" else message["role"],
        }
        for message in messages
    ]


def _print_progress(
    successful: int,
    failed: int,
    usage: TokenUsage,
    print_fn: Callable[[str], None],
) -> None:
    cost = calculate_distillation_cost(usage)
    print_fn(
        f"Generated: {successful} | Failed: {failed} | "
        f"Input tokens: {usage.prompt_tokens:,} | "
        f"Cached input tokens: {usage.cached_prompt_tokens:,} | "
        f"Output tokens: {usage.completion_tokens:,} | "
        f"Estimated cost: ${cost.total_cost:.4f}"
    )


def _print_final_stats(
    stats: DistillationStats,
    teacher_model: str,
    print_fn: Callable[[str], None],
) -> None:
    print_fn("=" * 60)
    print_fn("Knowledge distillation dataset generation finished.")
    print_fn(f"Teacher model:      {teacher_model}")
    print_fn(f"Successful samples: {stats.successful}")
    print_fn(f"Failed samples:     {stats.failed}")
    print_fn(f"Input tokens:       {stats.usage.prompt_tokens:,}")
    print_fn(
        f"Cached input tokens:{stats.usage.cached_prompt_tokens:>12,}"
    )
    print_fn(f"Output tokens:      {stats.usage.completion_tokens:,}")
    print_fn(f"Uncached input cost:${stats.cost.uncached_input_cost:>12.4f}")
    print_fn(f"Cached input cost:  ${stats.cost.cached_input_cost:>12.4f}")
    print_fn(f"Output cost:        ${stats.cost.output_cost:>12.4f}")
    print_fn(f"Estimated cost:     ${stats.cost.total_cost:>12.4f}")
    print_fn(f"Dataset saved to:   {stats.output_path}")
    print_fn("=" * 60)


def generate_distillation_dataset(
    teacher: OpenAIModel,
    raw_data_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = DISTILLATION_OUTPUT_PATH,
    *,
    shuffle_seed: int = DISTILLATION_SHUFFLE_SEED,
    report_interval: int = DISTILLATION_REPORT_INTERVAL,
    record_loader: Callable[[str | Path], list[dict[str, Any]]] = load_jsonl,
    record_resetter: Callable[[str | Path], None] = reset_jsonl,
    record_appender: Callable[[str | Path, dict[str, Any]], None] = append_jsonl,
    prompt_builder: Callable[[str], list[dict[str, str]]] = (
        build_extraction_messages
    ),
    progress_factory: Callable[[Iterable[dict[str, Any]]], Iterable[dict[str, Any]]] = tqdm,
    print_fn: Callable[[str], None] = print,
) -> DistillationStats:
    raw_data = record_loader(raw_data_path)
    random.Random(shuffle_seed).shuffle(raw_data)
    print_fn(f"Raw data: {len(raw_data)}")

    destination = Path(output_path)
    record_resetter(destination)

    successful = 0
    failed = 0
    usage = TokenUsage()
    output_schema = NewsDetails.model_json_schema()

    for story in progress_factory(raw_data):
        story_text = str(story.get("content") or "").strip()
        if not story_text:
            continue

        try:
            generation = teacher.generate_with_usage(
                _for_reasoning_model(prompt_builder(story_text))
            )
        except Exception as exc:
            failed += 1
            print_fn(f"OpenAI API error: {exc}")
            continue

        usage = usage + generation.usage

        if not generation.text:
            failed += 1
            print_fn("OpenAI returned an empty response.")
            continue

        try:
            response_payload = json.loads(generation.text)
        except (json.JSONDecodeError, TypeError) as exc:
            failed += 1
            print_fn(f"JSON parsing failed: {exc}")
            continue

        try:
            validated_response = NewsDetails.model_validate(response_payload)
        except Exception as exc:
            failed += 1
            print_fn(f"Schema validation failed: {exc}")
            continue

        sample = {
            "id": successful,
            "story": story_text,
            "task": DISTILLATION_TASK,
            "output_schema": output_schema,
            "response": validated_response.model_dump(),
            "teacher_model": teacher.model_id,
        }
        record_appender(destination, sample)
        successful += 1

        if successful % report_interval == 0:
            _print_progress(successful, failed, usage, print_fn)

    cost = calculate_distillation_cost(usage)
    stats = DistillationStats(
        successful=successful,
        failed=failed,
        usage=usage,
        cost=cost,
        output_path=destination,
    )
    _print_final_stats(stats, teacher.model_id, print_fn)
    return stats


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Generate a fresh extraction SFT dataset with the paid o4-mini API."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=require_api_key("OPENAI_API_KEY"))
        teacher = OpenAIModel(
            client,
            model_id=DISTILLATION_TEACHER_MODEL_ID,
            temperature=None,
            response_format=DISTILLATION_RESPONSE_FORMAT,
        )
        generate_distillation_dataset(teacher)
    except Exception as exc:
        print(f"Distillation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
