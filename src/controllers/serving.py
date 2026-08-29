import time
from collections.abc import Mapping
from typing import Any


_MISSING = object()


def _malformed_models_response(detail: str, *, cause: Exception | None = None) -> RuntimeError:
    error = RuntimeError(
        f"Malformed or incompatible /v1/models response: {detail}."
    )
    if cause is not None:
        error.__cause__ = cause
    return error


def list_served_model_ids(client: Any) -> list[str]:
    """Return model IDs currently exposed by a vLLM OpenAI client."""

    try:
        response = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            "Unable to reach local vLLM at localhost:8000; check that the WSL "
            "server is running and port 8000 is available."
        ) from exc

    try:
        models = getattr(response, "data", _MISSING)
    except Exception as exc:
        raise _malformed_models_response("the data field could not be read", cause=exc)
    if models is _MISSING:
        raise _malformed_models_response("the data field is missing")

    try:
        entries = iter(models)
    except Exception as exc:
        raise _malformed_models_response("the data field is not iterable", cause=exc)

    model_ids: list[str] = []
    try:
        for model in entries:
            model_id = (
                model.get("id", _MISSING)
                if isinstance(model, Mapping)
                else getattr(model, "id", _MISSING)
            )
            if model_id is _MISSING:
                raise _malformed_models_response("a served model entry is missing id")
            if model_id is None:
                raise _malformed_models_response("a served model entry has a null id")
            model_id_text = str(model_id)
            if not model_id_text.strip():
                raise _malformed_models_response(
                    "a served model entry has a blank id"
                )
            model_ids.append(model_id_text)
    except RuntimeError:
        raise
    except Exception as exc:
        raise _malformed_models_response(
            "the data entries could not be read", cause=exc
        )
    return model_ids


def wait_for_served_model(
    client: Any,
    model_id: str,
    *,
    attempts: int,
    interval_seconds: float = 5.0,
    sleep=time.sleep,
) -> list[str]:
    """Wait until the exact requested model ID appears in the served model list."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last_successful_ids: list[str] | None = None
    last_connection_error: RuntimeError | None = None
    had_successful_poll = False
    for attempt in range(attempts):
        try:
            served_ids = list_served_model_ids(client)
        except RuntimeError as exc:
            last_connection_error = exc
        else:
            had_successful_poll = True
            last_successful_ids = served_ids
            if model_id in served_ids:
                return served_ids

        if attempt < attempts - 1:
            sleep(interval_seconds)

    if had_successful_poll:
        message = (
            f"Timed out waiting for vLLM model {model_id!r}; "
            f"last served model IDs: {last_successful_ids!r}."
        )
        if last_connection_error is not None:
            message += f" Last connection or parsing error: {last_connection_error}"
    else:
        message = (
            f"Timed out waiting for vLLM model {model_id!r}; "
            "the local server could not be reached after all readiness attempts. "
            f"Last connection or parsing error: {last_connection_error}"
        )
    if last_connection_error is not None:
        raise RuntimeError(message) from last_connection_error
    raise RuntimeError(message)
