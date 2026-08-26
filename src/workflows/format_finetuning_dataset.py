import argparse
import json
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.helpers.config import (
    DISTILLATION_OUTPUT_PATH,
    FINETUNING_FORMAT_SHUFFLE_SEED,
    LLAMA_FACTORY_DATASET_INFO_PATH,
    LLAMA_FACTORY_OUTPUT_PATH,
)
from src.templates.finetuning import SYSTEM_MESSAGE
from src.utils.jsonl import load_jsonl, write_json


def _dataset_info_for(file_name: str) -> dict[str, Any]:
    return {
        "news_finetuning": {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        },
    }


def configure_stdout_encoding(stdout: Any) -> None:
    if hasattr(stdout, "reconfigure"):
        stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class FormattingResult:
    samples: list[dict[str, Any]]
    skipped: int


@dataclass(frozen=True)
class DatasetFormattingStats:
    converted: int
    skipped: int
    output_path: Path
    dataset_info_path: Path


def format_finetuning_records(
    records: list[dict[str, Any]],
    *,
    shuffle_seed: int = FINETUNING_FORMAT_SHUFFLE_SEED,
    print_fn: Callable[[str], None] = print,
) -> FormattingResult:
    llm_finetunning_data: list[dict[str, Any]] = []
    skipped = 0

    for line_number, record in enumerate(records, start=1):
        schema = record.get("output_schema", record.get("output_scheme"))
        if schema is None:
            skipped += 1
            print_fn(
                f"Skipping line {line_number}: "
                "no output_schema/output_scheme field."
            )
            continue

        schema_text = (
            json.dumps(schema, ensure_ascii=False)
            if isinstance(schema, dict)
            else str(schema)
        )

        response = record.get("response")
        if response is None:
            skipped += 1
            print_fn(f"Skipping line {line_number}: missing response.")
            continue

        llm_finetunning_data.append(
            {
                "system": SYSTEM_MESSAGE,
                "instruction": "\n".join(
                    (
                        "# Story:",
                        record["story"],
                        "",
                        "# Task:",
                        record["task"],
                        "",
                        "# Output Schema:",
                        schema_text,
                        "",
                        "# Output JSON:",
                    )
                ),
                "input": "",
                "output": json.dumps(response, ensure_ascii=False, default=str),
                "history": [],
            }
        )

    random.Random(shuffle_seed).shuffle(llm_finetunning_data)
    return FormattingResult(samples=llm_finetunning_data, skipped=skipped)


def format_finetuning_dataset(
    sft_data_path: str | Path = DISTILLATION_OUTPUT_PATH,
    output_path: str | Path = LLAMA_FACTORY_OUTPUT_PATH,
    dataset_info_path: str | Path = LLAMA_FACTORY_DATASET_INFO_PATH,
    *,
    shuffle_seed: int = FINETUNING_FORMAT_SHUFFLE_SEED,
    print_fn: Callable[[str], None] = print,
) -> DatasetFormattingStats:
    result = format_finetuning_records(
        load_jsonl(sft_data_path),
        shuffle_seed=shuffle_seed,
        print_fn=print_fn,
    )
    destination = Path(output_path)
    registration_path = Path(dataset_info_path)
    write_json(destination, result.samples)
    write_json(registration_path, _dataset_info_for(destination.name))

    print_fn(f"\nLoaded {len(result.samples):,} training samples.")
    if result.samples:
        print_fn("\nFirst converted sample:\n")
        print_fn(json.dumps(result.samples[0], ensure_ascii=False, indent=2))

    return DatasetFormattingStats(
        converted=len(result.samples),
        skipped=result.skipped,
        output_path=destination,
        dataset_info_path=registration_path,
    )


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Convert the distilled SFT JSONL dataset to LLaMA-Factory format."
    )


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    configure_stdout_encoding(sys.stdout)
    format_finetuning_dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
