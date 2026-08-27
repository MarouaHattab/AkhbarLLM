from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.helpers.config import ENV_PATH
from src.helpers.environment import require_api_key


def authenticate_training_services(
    env_path: str | Path = ENV_PATH,
    environ: Mapping[str, str] | None = None,
    hf_login_fn: Callable[..., Any] | None = None,
    wandb_login_fn: Callable[..., Any] | None = None,
) -> None:
    hf_token = require_api_key(
        "HF_TOKEN",
        env_path=env_path,
        environ=environ,
    )
    wandb_api_key = require_api_key(
        "WANDB_API_KEY",
        env_path=env_path,
        environ=environ,
    )

    if hf_login_fn is None:
        from huggingface_hub import login

        hf_login_fn = login

    if wandb_login_fn is None:
        import wandb

        wandb_login_fn = wandb.login

    try:
        hf_login_fn(
            token=hf_token,
            add_to_git_credential=False,
        )
    except Exception as exc:
        raise RuntimeError("Hugging Face login failed.") from exc

    try:
        wandb_login_fn(key=wandb_api_key)
    except Exception as exc:
        raise RuntimeError("Weights & Biases login failed.") from exc
