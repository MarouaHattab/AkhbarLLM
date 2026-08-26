import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from src.helpers.config import ENV_PATH


def require_api_key(
    key_name: str,
    env_path: str | Path = ENV_PATH,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read one API key, preferring the process environment over .env."""
    target_environ = os.environ if environ is None else environ
    file_values = dotenv_values(env_path)
    value = (
        target_environ.get(key_name)
        or file_values.get(key_name)
        or ""
    ).strip()
    if not value:
        raise RuntimeError(f"{key_name} is missing or empty in {Path(env_path)}")
    return value
