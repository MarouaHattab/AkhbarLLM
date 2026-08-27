import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.helpers.config import (
    LLAMA_FACTORY_REPOSITORY_DIR,
    LLAMA_FACTORY_TRAIN_DATASET_NAME,
    LLAMA_FACTORY_TRAIN_PATH,
    LLAMA_FACTORY_VALIDATION_DATASET_NAME,
    LLAMA_FACTORY_VALIDATION_PATH,
)
from src.utils.jsonl import write_json


COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history",
}


@dataclass(frozen=True)
class RegistrationStats:
    before_count: int
    after_count: int
    dataset_info_path: Path


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def register_llamafactory_datasets(
    llamafactory_dir: str | Path = LLAMA_FACTORY_REPOSITORY_DIR,
    train_path: str | Path = LLAMA_FACTORY_TRAIN_PATH,
    val_path: str | Path = LLAMA_FACTORY_VALIDATION_PATH,
    *,
    print_fn: Callable[[str], None] = print,
) -> RegistrationStats:
    repository = Path(llamafactory_dir).resolve()
    dataset_info_path = repository / "data" / "dataset_info.json"
    resolved_train_path = Path(train_path).resolve()
    resolved_val_path = Path(val_path).resolve()

    for required_path in (
        dataset_info_path,
        resolved_train_path,
        resolved_val_path,
    ):
        _require_file(required_path)

    with dataset_info_path.open("r", encoding="utf-8") as source:
        dataset_info: Any = json.load(source)
    if not isinstance(dataset_info, dict):
        raise ValueError("dataset_info.json must contain a JSON object.")

    before_count = len(dataset_info)
    print_fn(f"Datasets already registered: {before_count}")

    dataset_info[LLAMA_FACTORY_TRAIN_DATASET_NAME] = {
        "file_name": str(resolved_train_path),
        "columns": COLUMNS,
    }
    dataset_info[LLAMA_FACTORY_VALIDATION_DATASET_NAME] = {
        "file_name": str(resolved_val_path),
        "columns": COLUMNS,
    }

    write_json(dataset_info_path, dataset_info)

    with dataset_info_path.open("r", encoding="utf-8") as source:
        saved_info = json.load(source)

    after_count = len(saved_info)
    print_fn(f"Datasets after adding ours: {after_count}")
    print_fn("\nAdded:")
    print_fn(LLAMA_FACTORY_TRAIN_DATASET_NAME)
    print_fn(LLAMA_FACTORY_VALIDATION_DATASET_NAME)
    print_fn("\nOur train config:")
    print_fn(
        json.dumps(
            saved_info[LLAMA_FACTORY_TRAIN_DATASET_NAME],
            ensure_ascii=False,
            indent=2,
        )
    )
    print_fn("\nOur validation config:")
    print_fn(
        json.dumps(
            saved_info[LLAMA_FACTORY_VALIDATION_DATASET_NAME],
            ensure_ascii=False,
            indent=2,
        )
    )

    return RegistrationStats(
        before_count=before_count,
        after_count=after_count,
        dataset_info_path=dataset_info_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register the news datasets in LLaMA-Factory."
    )
    parser.add_argument(
        "--llamafactory-dir",
        type=Path,
        default=LLAMA_FACTORY_REPOSITORY_DIR,
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=LLAMA_FACTORY_TRAIN_PATH,
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=LLAMA_FACTORY_VALIDATION_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        register_llamafactory_datasets(
            args.llamafactory_dir,
            args.train_path,
            args.val_path,
        )
    except Exception as exc:
        print(f"Registration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
