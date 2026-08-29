from dataclasses import dataclass
from typing import Any

from src.controllers.serving import list_served_model_ids


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
