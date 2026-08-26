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
    FINETUNING_TRAIN_SAMPLE_SIZE,
    LLAMA_FACTORY_DATA_DIR,
)
from src.templates.finetuning import SYSTEM_MESSAGE
from src.utils.jsonl import load_jsonl, write_json


def _dataset_info_for(
    train_file_name: str,
    validation_file_name: str,
) -> dict[str, Any]:
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history",
    }
    return {
        "news_finetuning_train": {
            "file_name": train_file_name,
            "columns": columns,
        },
        "news_finetuning_validation": {
            "file_name": validation_file_name,
            "columns": columns,
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
    training_samples: int
    validation_samples: int
    train_path: Path
    validation_path: Path
    dataset_info_path: Path


def split_finetuning_data(
    llm_finetunning_data: list[dict[str, Any]],
    *,
    train_sample_size: int = FINETUNING_TRAIN_SAMPLE_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ds = llm_finetunning_data[:train_sample_size]
    eval_ds = llm_finetunning_data[train_sample_size:]
    return train_ds, eval_ds


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
    llama_factory_data_dir: str | Path = LLAMA_FACTORY_DATA_DIR,
    *,
    shuffle_seed: int = FINETUNING_FORMAT_SHUFFLE_SEED,
    train_sample_size: int = FINETUNING_TRAIN_SAMPLE_SIZE,
    print_fn: Callable[[str], None] = print,
) -> DatasetFormattingStats:
    result = format_finetuning_records(
        load_jsonl(sft_data_path),
        shuffle_seed=shuffle_seed,
        print_fn=print_fn,
    )
    train_ds, eval_ds = split_finetuning_data(
        result.samples,
        train_sample_size=train_sample_size,
    )
    data_dir = Path(llama_factory_data_dir)
    train_path = data_dir / "train.json"
    validation_path = data_dir / "val.json"
    registration_path = data_dir / "dataset_info.json"
    write_json(train_path, train_ds)
    write_json(validation_path, eval_ds)
    write_json(
        registration_path,
        _dataset_info_for(train_path.name, validation_path.name),
    )

    print_fn(f"Total samples:      {len(result.samples)}")
    print_fn(f"Training samples:   {len(train_ds)}")
    print_fn(f"Validation samples: {len(eval_ds)}")
    print_fn("\nDatasets saved successfully.")
    print_fn(f"Train:\n{train_path}")
    print_fn(f"Validation:\n{validation_path}")
    print_fn(f"\nTrain file size: {train_path.stat().st_size / 1024 / 1024:.2f} MB")
    print_fn(
        "Validation file size: "
        f"{validation_path.stat().st_size / 1024 / 1024:.2f} MB"
    )
    if result.samples:
        print_fn("\nFirst converted sample:\n")
        print_fn(json.dumps(result.samples[0], ensure_ascii=False, indent=2))

    return DatasetFormattingStats(
        converted=len(result.samples),
        skipped=result.skipped,
        training_samples=len(train_ds),
        validation_samples=len(eval_ds),
        train_path=train_path,
        validation_path=validation_path,
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
