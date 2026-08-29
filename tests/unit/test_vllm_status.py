from types import SimpleNamespace

import pytest

from src.controllers.vllm_status import check_vllm_connection


def fake_client_with_models(model_ids: list[str]) -> SimpleNamespace:
    response = SimpleNamespace(
        data=[SimpleNamespace(id=model_id) for model_id in model_ids]
    )
    return SimpleNamespace(
        models=SimpleNamespace(list=lambda: response)
    )


def test_connection_check_confirms_expected_model() -> None:
    result = check_vllm_connection(
        fake_client_with_models(["news-lora"]),
        "news-lora",
    )

    assert result.connected is True
    assert result.served_model_ids == ("news-lora",)


def test_connection_check_rejects_missing_model() -> None:
    with pytest.raises(RuntimeError, match="news-lora"):
        check_vllm_connection(
            fake_client_with_models(["base-model"]),
            "news-lora",
        )


def test_connection_check_preserves_readiness_error() -> None:
    class BrokenModels:
        def list(self) -> object:
            raise ConnectionError("offline")

    client = SimpleNamespace(models=BrokenModels())

    with pytest.raises(RuntimeError, match="Unable to reach local vLLM"):
        check_vllm_connection(client, "news-lora")


@pytest.mark.parametrize(
    ("state", "label"),
    [
        ("not_checked", "Not checked"),
        ("connected", "Connected"),
        ("unavailable", "Unavailable"),
    ],
)
def test_connection_status_labels(state: str, label: str) -> None:
    from src.controllers.vllm_status import connection_status_label

    assert connection_status_label(state) == label


def test_connection_state_maps_check_outcome() -> None:
    from src.controllers.vllm_status import connection_state_from_check

    assert connection_state_from_check(None) == "connected"
    assert connection_state_from_check(RuntimeError("offline")) == (
        "unavailable"
    )
