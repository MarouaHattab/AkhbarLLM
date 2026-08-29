import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Any

from src.controllers.serving import list_served_model_ids, wait_for_served_model
from src.helpers import config
from src.models.factory import load_language_model
from src.utils.console import configure_utf8_output


def _load_served_vllm() -> Any:
    """Load the local served adapter through the shared model factory."""
    return load_language_model("vllm")


def check_vllm(
    *,
    wait: bool,
    model_loader: Callable[[], Any] = _load_served_vllm,
    waiter: Callable[..., list[str]] = wait_for_served_model,
) -> list[str]:
    """Return served model IDs after checking the configured model is ready."""

    runtime = model_loader()
    if wait:
        return waiter(
            runtime.client,
            runtime.model_id,
            attempts=config.VLLM_READINESS_ATTEMPTS,
            interval_seconds=config.VLLM_READINESS_INTERVAL_SECONDS,
        )

    served_ids = list_served_model_ids(runtime.client)
    if runtime.model_id not in served_ids:
        raise RuntimeError(
            f"Expected vLLM model {runtime.model_id!r} is not served; "
            f"currently served model IDs: {served_ids!r}."
        )
    return served_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check readiness of the configured local vLLM model."
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the configured model is served (default: one-shot).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output(sys.stdout, sys.stderr)
    args = build_parser().parse_args(argv)
    try:
        served_ids = check_vllm(wait=args.wait)
    except Exception as exc:
        print(f"vLLM readiness failed: {exc}", file=sys.stderr)
        return 1

    print(f"vLLM ready; served model IDs: {served_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
