import argparse
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.helpers.config import (
    FINETUNED_MODEL_RELATIVE_DIR,
    LLAMA_FACTORY_REPOSITORY_DIR,
    LLAMA_FACTORY_TRAIN_PATH,
    LLAMA_FACTORY_VALIDATION_PATH,
    TRAINING_CONFIG_FILENAME,
    TRAINING_CONFIG_SOURCE_PATH,
)
from src.helpers.training_auth import authenticate_training_services
from src.workflows.register_llamafactory_datasets import (
    register_llamafactory_datasets,
)


@dataclass(frozen=True)
class PreparationStats:
    config_path: Path
    output_dir: Path
    command: str


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def _require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def prepare_finetuning(
    llamafactory_dir: str | Path = LLAMA_FACTORY_REPOSITORY_DIR,
    train_path: str | Path = LLAMA_FACTORY_TRAIN_PATH,
    val_path: str | Path = LLAMA_FACTORY_VALIDATION_PATH,
    *,
    config_source_path: str | Path = TRAINING_CONFIG_SOURCE_PATH,
    authenticate_fn: Callable[[], None] = authenticate_training_services,
    register_fn: Callable[..., Any] = register_llamafactory_datasets,
    print_fn: Callable[[str], None] = print,
) -> PreparationStats:
    repository = Path(llamafactory_dir).resolve()
    resolved_train_path = Path(train_path).resolve()
    resolved_val_path = Path(val_path).resolve()
    source_config_path = Path(config_source_path).resolve()
    dataset_info_path = repository / "data" / "dataset_info.json"
    examples_dir = repository / "examples" / "train_lora"

    for required_path in (
        source_config_path,
        resolved_train_path,
        resolved_val_path,
        dataset_info_path,
    ):
        _require_file(required_path)
    _require_directory(examples_dir)

    authenticate_fn()
    register_fn(
        repository,
        resolved_train_path,
        resolved_val_path,
        print_fn=print_fn,
    )

    target_config_path = examples_dir / TRAINING_CONFIG_FILENAME
    shutil.copy2(source_config_path, target_config_path)

    output_dir = repository.parent / FINETUNED_MODEL_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    command = (
        "llamafactory-cli train "
        "examples/train_lora/news_finetune.yaml"
    )
    print_fn("Preparation finished. Training was not started.")
    print_fn(f"cd {repository}")
    print_fn(command)

    return PreparationStats(
        config_path=target_config_path,
        output_dir=output_dir,
        command=command,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LLaMA-Factory for news fine-tuning without running it."
        )
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
        prepare_finetuning(
            args.llamafactory_dir,
            args.train_path,
            args.val_path,
        )
    except Exception as exc:
        print(f"Fine-tuning preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
