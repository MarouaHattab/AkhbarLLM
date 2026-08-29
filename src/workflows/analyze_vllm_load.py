from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.helpers import config
from src.models.load_testing import LoadTestSummary
from src.utils.console import configure_utf8_output


def _iter_records(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed load-test JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Record on line {line_number} must be a JSON object"
                )
            yield line_number, value


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return [record for _, record in _iter_records(path)]


def _validate_text(record: Mapping[str, Any], line_number: int, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(
            f"Record on line {line_number} field {field!r} must be a string"
        )
    return value


def _load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def _encode_token_count(
    tokenizer: Any,
    text: str,
    *,
    line_number: int,
    field: str,
) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception as exc:
        raise ValueError(
            f"Record on line {line_number} field {field!r} could not be tokenized: {exc}"
        ) from exc


def _write_text_file(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(content)


def _cleanup_temp_files(*paths: Path) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except FileNotFoundError:
            pass


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid4().hex}.tmp")


def _replace_path(source: Path, destination: Path) -> None:
    source.replace(destination)


def _preserve_path(source: Path, backup: Path) -> None:
    source.replace(backup)


def _cleanup_with_notes(error: Exception, *paths: Path) -> None:
    try:
        _cleanup_temp_files(*paths)
    except Exception as cleanup_exc:
        error.add_note(f"cleanup failed: {cleanup_exc}")


def _rollback_artifact(
    error: Exception,
    *,
    target: Path,
    backup: Path,
    preserved: bool,
    installed: bool,
) -> None:
    retained_backup = False
    should_restore = preserved or (backup.exists() and not target.exists())

    if should_restore:
        try:
            _preserve_path(backup, target)
        except Exception as rollback_exc:
            error.add_note(f"rollback failed for {target}: {rollback_exc}")
            retained_backup = backup.exists()
    elif installed and target.exists():
        try:
            target.unlink()
        except Exception as rollback_exc:
            error.add_note(f"rollback failed for {target}: {rollback_exc}")

    if retained_backup:
        error.add_note(f"retained recovery backup: {backup}")


def analyze_load_test(
    path: str | Path,
    *,
    model_id: str = config.QWEN_MODEL_ID,
    tokenizer_loader: Callable[[str], Any] = _load_tokenizer,
) -> LoadTestSummary:
    parsed: list[tuple[int, str, str]] = []
    for line_number, record in _iter_records(path):
        parsed.append(
            (
                line_number,
                _validate_text(record, line_number, "prompt"),
                _validate_text(record, line_number, "response"),
            )
        )

    response_count = len(parsed)
    total_input_tokens = 0
    total_output_tokens = 0
    if parsed:
        try:
            tokenizer = tokenizer_loader(model_id)
        except Exception as exc:
            raise ValueError(
                f"Could not load tokenizer {model_id!r} for token counts: {exc}"
            ) from exc
        for line_number, prompt, response in parsed:
            total_input_tokens += _encode_token_count(
                tokenizer,
                prompt,
                line_number=line_number,
                field="prompt",
            )
            total_output_tokens += _encode_token_count(
                tokenizer,
                response,
                line_number=line_number,
                field="response",
            )

    total_tokens = total_input_tokens + total_output_tokens
    if response_count:
        average_input_tokens = total_input_tokens / response_count
        average_output_tokens = total_output_tokens / response_count
    else:
        average_input_tokens = 0.0
        average_output_tokens = 0.0
    return LoadTestSummary(
        response_count=response_count,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        average_input_tokens=average_input_tokens,
        average_output_tokens=average_output_tokens,
    )


def format_summary(summary: LoadTestSummary) -> str:
    return "\n".join(
        (
            f"Loaded {summary.response_count} responses.",
            f"Total Input Tokens: {summary.total_input_tokens}",
            f"Total Output Tokens: {summary.total_output_tokens}",
        )
    )


def write_outputs(
    summary: LoadTestSummary,
    text_path: str | Path,
    json_path: str | Path,
) -> None:
    """Write the text and JSON summaries using UTF-8 and LF newlines."""

    text_target = Path(text_path)
    json_target = Path(json_path)
    if text_target.resolve() == json_target.resolve():
        raise ValueError(
            "text and JSON summary paths resolve to the same filesystem target"
        )
    text_content = format_summary(summary) + "\n"
    json_content = json.dumps(summary.model_dump(), ensure_ascii=False, indent=2) + "\n"
    text_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    text_temp = _temporary_sibling(text_target)
    json_temp = _temporary_sibling(json_target)
    text_backup = _temporary_sibling(text_target)
    json_backup = _temporary_sibling(json_target)

    try:
        _write_text_file(text_temp, text_content)
        _write_text_file(json_temp, json_content)
    except Exception as exc:
        _cleanup_with_notes(exc, text_temp, json_temp, text_backup, json_backup)
        raise

    text_existed = text_target.exists()
    json_existed = json_target.exists()
    text_preserved = False
    json_preserved = False
    text_installed = False
    json_installed = False
    try:
        if text_existed:
            _preserve_path(text_target, text_backup)
            text_preserved = True
        if json_existed:
            _preserve_path(json_target, json_backup)
            json_preserved = True

        _replace_path(text_temp, text_target)
        text_installed = True
        _replace_path(json_temp, json_target)
        json_installed = True
    except Exception as exc:
        _rollback_artifact(
            exc,
            target=text_target,
            backup=text_backup,
            preserved=text_preserved,
            installed=text_installed,
        )
        _rollback_artifact(
            exc,
            target=json_target,
            backup=json_backup,
            preserved=json_preserved,
            installed=json_installed,
        )
        _cleanup_with_notes(exc, text_temp, json_temp)
        raise
    _cleanup_temp_files(text_temp, json_temp, text_backup, json_backup)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze token usage from vLLM load-test JSONL records."
    )
    parser.add_argument(
        "--input",
        "--input-path",
        dest="input",
        type=Path,
        default=config.VLLM_LOAD_TEST_JSONL_PATH,
        help="UTF-8 load-test JSONL input path.",
    )
    parser.add_argument(
        "--text",
        "--text-output",
        dest="text",
        type=Path,
        default=config.VLLM_LOAD_TEST_SUMMARY_TEXT_PATH,
        help="UTF-8 human-readable summary output path.",
    )
    parser.add_argument(
        "--json",
        "--json-output",
        dest="json",
        type=Path,
        default=config.VLLM_LOAD_TEST_SUMMARY_JSON_PATH,
        help="UTF-8 JSON summary output path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output(sys.stdout, sys.stderr)
    args = build_parser().parse_args(argv)
    try:
        summary = analyze_load_test(args.input)
        write_outputs(summary, args.text, args.json)
    except Exception as exc:
        print(f"Load-test analysis failed: {exc}", file=sys.stderr)
        return 1

    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
