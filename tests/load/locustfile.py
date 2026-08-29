from __future__ import annotations

import html
import json
import random
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from faker import Faker
from locust import HttpUser, between, events, task
from locust.exception import StopTest
from locust.runners import MasterRunner, WorkerRunner

from src.helpers.config import (
    VLLM_LOAD_TEST_FAKER_LOCALE,
    VLLM_LOAD_TEST_HTML_PATH,
    VLLM_LOAD_TEST_JSONL_PATH,
    VLLM_LOAD_TEST_MAX_PROMPT_CHARS,
    VLLM_LOAD_TEST_MAX_TOKENS,
    VLLM_LOAD_TEST_MIN_PROMPT_CHARS,
    VLLM_LOAD_TEST_TEMPERATURE,
    VLLM_MODEL_ID,
)


fake = Faker(VLLM_LOAD_TEST_FAKER_LOCALE)
RESULTS_PATH = Path(VLLM_LOAD_TEST_JSONL_PATH)
HTML_PATH = Path(VLLM_LOAD_TEST_HTML_PATH)
SERVED_MODEL = VLLM_MODEL_ID
RESULTS_LOCK = threading.Lock()
_ACTIVE_ENVIRONMENT: Any | None = None


def init_command_line_parser(parser: Any) -> None:
    """Register the options used by this load test with Locust."""
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=VLLM_LOAD_TEST_JSONL_PATH,
        help="UTF-8 JSONL path for successful completion records.",
    )
    parser.add_argument(
        "--served-model",
        default=VLLM_MODEL_ID,
        help="Model ID sent in completion requests.",
    )


events.init_command_line_parser.add_listener(init_command_line_parser)


def _is_local_runner(environment: Any) -> bool:
    runner = getattr(environment, "runner", None)
    if runner is None:
        return True
    return not isinstance(runner, (MasterRunner, WorkerRunner))


@events.test_start.add_listener
def test_start(environment: Any) -> None:
    """Start a fresh JSONL file for one local Locust run."""
    global RESULTS_PATH, SERVED_MODEL, _ACTIVE_ENVIRONMENT

    if not _is_local_runner(environment):
        environment.process_exit_code = 1
        raise StopTest(
            "CompletionLoadTest supports local Locust runs only; "
            "MasterRunner and WorkerRunner are not supported."
        )

    _ACTIVE_ENVIRONMENT = environment
    options = getattr(environment, "parsed_options", None)
    RESULTS_PATH = Path(
        getattr(options, "results_jsonl", VLLM_LOAD_TEST_JSONL_PATH)
    )
    SERVED_MODEL = str(getattr(options, "served_model", VLLM_MODEL_ID))
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOCK:
        RESULTS_PATH.write_text("", encoding="utf-8", newline="\n")


def generate_arabic_prompt() -> str:
    """Generate random Arabic text between 150 and 200 characters."""
    return fake.text(
        max_nb_chars=random.randint(
            VLLM_LOAD_TEST_MIN_PROMPT_CHARS,
            VLLM_LOAD_TEST_MAX_PROMPT_CHARS,
        )
    )


def build_payload(prompt: str, model_id: str) -> dict[str, Any]:
    """Build the OpenAI-compatible completions request body."""
    return {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": VLLM_LOAD_TEST_MAX_TOKENS,
        "temperature": VLLM_LOAD_TEST_TEMPERATURE,
    }


def extract_completion(payload: Any) -> str:
    """Return ``choices[0].text`` from a ``/v1/completions`` body."""
    return payload["choices"][0]["text"]


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append one UTF-8 JSON object without interleaving concurrent writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with RESULTS_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(line)


def send_completion(client: Any, prompt: str, model_id: str) -> bool:
    """POST ``/v1/completions`` and save ``prompt`` / ``response`` on success."""
    request = client.post(
        "/v1/completions",
        json=build_payload(prompt, model_id),
        catch_response=True,
    )
    with request as response:
        if response.status_code != 200:
            response.failure(
                f"HTTP {response.status_code}: "
                f"{str(getattr(response, 'text', ''))[:500]}"
            )
            return False
        try:
            generated_text = extract_completion(response.json())
            append_record(
                RESULTS_PATH,
                {"prompt": prompt, "response": generated_text},
            )
        except Exception as exc:
            response.failure(f"Invalid response: {exc}")
            return False
        response.success()
        return True


def html_report_path(environment: Any) -> Path:
    """Prefer Locust ``--html`` when set; otherwise use the project report path."""
    options = getattr(environment, "parsed_options", None)
    locust_html = getattr(options, "html_file", None)
    if locust_html:
        return Path(locust_html)
    return HTML_PATH


def _stats_row(entry: Any) -> dict[str, Any]:
    avg = getattr(entry, "avg_response_time", 0) or 0
    minimum = getattr(entry, "min_response_time", None)
    maximum = getattr(entry, "max_response_time", 0) or 0
    median = getattr(entry, "median_response_time", 0) or 0
    return {
        "method": str(getattr(entry, "method", "") or ""),
        "name": str(getattr(entry, "name", "") or ""),
        "requests": int(getattr(entry, "num_requests", 0) or 0),
        "failures": int(getattr(entry, "num_failures", 0) or 0),
        "avg_ms": float(avg),
        "min_ms": None if minimum is None else float(minimum),
        "max_ms": float(maximum),
        "median_ms": float(median),
        "rps": float(getattr(entry, "total_rps", 0) or 0),
    }


def render_html_report(
    *,
    host: str,
    rows: list[Mapping[str, Any]],
    total: Mapping[str, Any],
) -> str:
    """Build a self-contained HTML report that opens in a browser as a file."""

    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            if value >= 100:
                return html.escape(f"{value:.0f}")
            return html.escape(f"{value:.2f}")
        return html.escape(str(value))

    def table_row(
        row: Mapping[str, Any], *, header: bool = False, css_class: str = ""
    ) -> str:
        tag = "th" if header else "td"
        if header:
            values = (
                "Type",
                "Name",
                "# reqs",
                "# fails",
                "Avg ms",
                "Min ms",
                "Max ms",
                "Med ms",
                "req/s",
            )
        else:
            values = (
                row["method"],
                row["name"],
                row["requests"],
                row["failures"],
                row["avg_ms"],
                row["min_ms"],
                row["max_ms"],
                row["median_ms"],
                row["rps"],
            )
        cells = "".join(f"<{tag}>{cell(value)}</{tag}>" for value in values)
        class_attr = f' class="{css_class}"' if css_class else ""
        return f"<tr{class_attr}>{cells}</tr>"

    body_rows = "\n".join(table_row(row) for row in rows)
    total_row = table_row(total, css_class="total")
    safe_host = html.escape(host or "http://localhost:8000")
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8"/>\n'
        "<title>vLLM Locust report</title>\n"
        "<style>\n"
        "body{font-family:Segoe UI,sans-serif;margin:2rem;background:#111827;color:#e5e7eb;}\n"
        "h1{font-size:1.5rem;margin:0 0 .25rem;}\n"
        "p{color:#9ca3af;}\n"
        "table{border-collapse:collapse;width:100%;background:#1f2937;}\n"
        "th,td{padding:.6rem .75rem;text-align:left;border-bottom:1px solid #374151;}\n"
        "th{background:#111827;color:#93c5fd;}\n"
        "tr.total td{font-weight:700;}\n"
        "code{color:#93c5fd;}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>vLLM Locust report</h1>\n"
        f"<p>Host <code>{safe_host}</code>. Open this file in Chrome or Edge "
        "(double-click it). The editor preview does not run this report.</p>\n"
        "<table>\n"
        "<thead>\n"
        f"{table_row({}, header=True)}\n"
        "</thead>\n"
        "<tbody>\n"
        f"{body_rows}\n"
        f"{total_row}\n"
        "</tbody>\n"
        "</table>\n"
        "</body>\n"
        "</html>\n"
    )


def write_html_report(environment: Any) -> Path:
    """Write a browser-openable HTML report from the current Locust stats."""
    runner = getattr(environment, "runner", None)
    stats = getattr(runner, "stats", None)
    path = html_report_path(environment)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if stats is not None:
        entries = getattr(stats, "entries", {}) or {}
        for entry in entries.values():
            rows.append(_stats_row(entry))
        total = _stats_row(stats.total)
        total["method"] = ""
        total["name"] = "Aggregated"
    else:
        total = {
            "method": "",
            "name": "Aggregated",
            "requests": 0,
            "failures": 0,
            "avg_ms": 0.0,
            "min_ms": None,
            "max_ms": 0.0,
            "median_ms": 0.0,
            "rps": 0.0,
        }
    host = str(getattr(environment, "host", "") or "http://localhost:8000")
    path.write_text(
        render_html_report(host=host, rows=rows, total=total),
        encoding="utf-8",
        newline="\n",
    )
    return path


@events.quit.add_listener
def persist_html_report(**kwargs: Any) -> None:
    """Replace Locust's SPA HTML (broken as a local file) with a standalone report."""
    environment = _ACTIVE_ENVIRONMENT
    if environment is None or not _is_local_runner(environment):
        return
    write_html_report(environment)


class CompletionLoadTest(HttpUser):
    """Concurrent users posting Arabic prompts to ``/v1/completions``."""

    wait_time = between(1, 3)

    @task
    def post_completion(self) -> None:
        prompt = generate_arabic_prompt()
        send_completion(self.client, prompt, SERVED_MODEL)
