import pytest

from src.ui.settings import InferenceSettings, load_default_settings


def test_defaults_use_environment_without_writing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWS_MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_API_BASE_URL", "http://server:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("VLLM_MODEL_ID", "news")

    settings = load_default_settings()

    assert settings == InferenceSettings(
        provider="vllm",
        base_url="http://server:8000/v1",
        api_key="secret",
        model_id="news",
        temperature=0.3,
        max_tokens=1000,
    )


def test_settings_reject_blank_vllm_endpoint() -> None:
    with pytest.raises(ValueError, match="API base URL"):
        InferenceSettings(
            provider="vllm",
            base_url=" ",
            api_key="local-vllm",
            model_id="news-lora",
            temperature=0.3,
            max_tokens=1000,
        )


def test_settings_reject_blank_vllm_model_id() -> None:
    with pytest.raises(ValueError, match="model ID"):
        InferenceSettings(
            provider="vllm",
            base_url="http://localhost:8000/v1",
            api_key="local-vllm",
            model_id=" ",
            temperature=0.3,
            max_tokens=1000,
        )


@pytest.mark.parametrize("temperature", [-0.01, 1.01])
def test_settings_reject_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="Temperature"):
        InferenceSettings(
            provider="vllm",
            base_url="http://localhost:8000/v1",
            api_key="local-vllm",
            model_id="news-lora",
            temperature=temperature,
            max_tokens=1000,
        )


@pytest.mark.parametrize("max_tokens", [0, 4097])
def test_settings_reject_invalid_max_tokens(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="Maximum output tokens"):
        InferenceSettings(
            provider="vllm",
            base_url="http://localhost:8000/v1",
            api_key="local-vllm",
            model_id="news-lora",
            temperature=0.3,
            max_tokens=max_tokens,
        )


def test_settings_are_hashable_for_streamlit_resource_cache() -> None:
    settings = InferenceSettings.direct()
    assert {settings: "cached"}[settings] == "cached"
