from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import login, whoami

from src.helpers.config import ENV_PATH
from src.helpers.environment import require_api_key


def authenticate_huggingface(
    env_path: str | Path = ENV_PATH,
    environ: Mapping[str, str] | None = None,
    login_fn: Callable[..., Any] = login,
    whoami_fn: Callable[..., Any] = whoami,
) -> str:
    token = require_api_key("HF_TOKEN", env_path=env_path, environ=environ)

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
