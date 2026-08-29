import json
import sys
from pathlib import Path
from typing import TextIO

from src.models.evaluation import EvaluationResult


def format_response_body(result: EvaluationResult) -> str:
    if not result.json_valid:
        return result.raw_response

    try:
        payload = json.loads(result.raw_response)
    except json.JSONDecodeError:
        return result.raw_response

    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_result(result: EvaluationResult) -> str:
    lines = [
        (
            f"\n=== {result.provider.upper()} | {result.model_id} | "
            f"{result.task.upper()} ==="
        ),
        format_response_body(result),
        f"JSON valid: {result.json_valid}",
        f"Schema valid: {result.schema_valid}",
    ]
    if result.validation_error:
        lines.append(f"Validation error: {result.validation_error}")
    return "\n".join(lines)


def print_result(
    result: EvaluationResult,
    output: TextIO = sys.stdout,
) -> None:
    print(format_result(result), file=output)


def write_report(path: Path, results: list[EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(format_result(result) for result in results)
    if report:
        report += "\n"
    path.write_text(report, encoding="utf-8", newline="\n")
