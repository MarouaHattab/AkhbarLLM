import os
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from huggingface_hub import login, whoami

from src.helpers.config import ENV_PATH


def authenticate_huggingface(
    env_path: str | Path = ENV_PATH,
    environ: MutableMapping[str, str] | None = None,
    login_fn: Callable[..., Any] = login,
    whoami_fn: Callable[..., Any] = whoami,
) -> str:
    target_environ = os.environ if environ is None else environ
    file_values = dotenv_values(env_path)
    token = (
        target_environ.get("HF_TOKEN")
        or file_values.get("HF_TOKEN")
        or ""
    ).strip()

    if not token:
        raise RuntimeError(f"HF_TOKEN is missing or empty in {Path(env_path)}")

    try:
        login_fn(
            token=token,
            add_to_git_credential=False,
            skip_if_logged_in=False,
        )
        whoami_fn(token=token)
    except Exception as exc:
        raise RuntimeError("Hugging Face authentication failed.") from exc

    return token
