from dataclasses import dataclass
from typing import Any, Literal

from src.controllers.serving import list_served_model_ids


ConnectionState = Literal["not_checked", "connected", "unavailable"]
CONNECTION_STATUS_LABELS: dict[ConnectionState, str] = {
    "not_checked": "Not checked",
    "connected": "Connected",
    "unavailable": "Unavailable",
}


@dataclass(frozen=True)
class VLLMConnectionResult:
    connected: bool
    served_model_ids: tuple[str, ...]


def check_vllm_connection(
    client: Any,
    model_id: str,
) -> VLLMConnectionResult:
    served = tuple(list_served_model_ids(client))
    if model_id not in served:
        raise RuntimeError(
            f"Configured model {model_id!r} is not served; "
            f"available: {list(served)!r}."
        )
    return VLLMConnectionResult(
        connected=True,
        served_model_ids=served,
    )


def connection_status_label(state: ConnectionState) -> str:
    return CONNECTION_STATUS_LABELS[state]


def connection_state_from_check(error: BaseException | None) -> ConnectionState:
    return "connected" if error is None else "unavailable"
