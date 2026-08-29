from pathlib import Path
from typing import Any

import pytest

from src.helpers.environment import read_optional_setting
from src.models.finetuned_qwen import resolve_adapter_source


def test_optional_setting_prefers_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SETTING=file-value\n", encoding="utf-8")
    assert read_optional_setting(
        "SETTING",
        env_path=env_file,
        environ={"SETTING": " process-value "},
    ) == "process-value"


def test_optional_setting_uses_file_then_none(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SETTING=file-value\n", encoding="utf-8")
    assert read_optional_setting(
        "SETTING", env_path=env_file, environ={}
    ) == "file-value"
    assert read_optional_setting(
        "MISSING", env_path=env_file, environ={}
    ) is None


def test_resolve_adapter_source_validates_existing_local_adapter(
    tmp_path: Path,
) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert resolve_adapter_source(tmp_path) == str(tmp_path.resolve())


def test_resolve_adapter_source_accepts_hub_repository() -> None:
    assert resolve_adapter_source(
        "marouaHattab/ArabLLM-news"
    ) == "marouaHattab/ArabLLM-news"


@pytest.mark.parametrize(
    "source",
    ["", "missing-local-directory", "outputs/models/missing"],
)
def test_resolve_adapter_source_rejects_invalid_source(source: str) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_adapter_source(source)


def test_factory_passes_adapter_environment_to_finetuned_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load(token: str, adapter_path: str) -> object:
        captured.update(token=token, adapter_path=adapter_path)
        return object()

    monkeypatch.setenv(
        "FINETUNED_ADAPTER_SOURCE", "marouaHattab/ArabLLM-news"
    )
    monkeypatch.setattr(
        "src.models.finetuned_qwen.FineTunedQwenModel.load", fake_load
    )
    from src.models.factory import _load_finetuned

    _load_finetuned("token")
    assert captured == {
        "token": "token",
        "adapter_path": "marouaHattab/ArabLLM-news",
    }


def test_factory_passes_vllm_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("VLLM_API_BASE_URL", "https://inference.example/v1")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("VLLM_MODEL_ID", "deployed-news")
    monkeypatch.setattr("src.models.vllm.VLLMModel.load", fake_load)
    from src.models.factory import _load_vllm

    _load_vllm()
    assert captured == {
        "base_url": "https://inference.example/v1",
        "api_key": "secret",
        "model_id": "deployed-news",
    }
