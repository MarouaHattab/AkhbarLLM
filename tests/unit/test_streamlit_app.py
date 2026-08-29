from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import streamlit_app


APP_PATH = Path(__file__).resolve().parents[2] / "app.py"


def test_initial_page_is_clean_and_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWS_MODEL_PROVIDER", "finetuned")

    def fail_if_loaded(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"model loaded during initial render: {args}, {kwargs}"
        )

    monkeypatch.setattr(
        streamlit_app,
        "build_streaming_controller",
        fail_if_loaded,
    )
    streamlit_app.get_controller.clear()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "ArabLLM Inference Studio"
    assert [tab.label for tab in app.tabs] == [
        "Extraction",
        "Translation",
    ]
    assert not app.radio
    assert {area.key for area in app.text_area} == {
        "extraction_story",
        "translation_story",
    }


def test_translation_tab_has_language_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWS_MODEL_PROVIDER", "finetuned")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    language_boxes = [
        box
        for box in app.selectbox
        if box.label.endswith("language")
    ]
    assert [box.key for box in language_boxes] == [
        "translation_source_language",
        "translation_target_language",
    ]
    assert [box.value for box in language_boxes] == [
        "Arabic",
        "English",
    ]


def test_vllm_sidebar_exposes_only_inference_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWS_MODEL_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_API_BASE_URL", "http://server:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "environment-secret")
    monkeypatch.setenv("VLLM_MODEL_ID", "news-lora")

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    backend = next(
        item
        for item in app.sidebar.selectbox
        if item.label == "Backend"
    )
    assert backend.value == "vLLM"
    assert {
        item.label for item in app.sidebar.text_input
    } == {"API base URL", "Model ID", "API key override"}
    api_key = next(
        item
        for item in app.sidebar.text_input
        if item.label == "API key override"
    )
    assert api_key.value == ""
    assert {
        item.label for item in app.sidebar.slider
    } == {"Temperature", "Maximum output tokens"}
    assert any(
        button.label == "Check connection"
        for button in app.sidebar.button
    )
    assert "History" not in [tab.label for tab in app.tabs]
    assert "Architecture" not in [tab.label for tab in app.tabs]


def test_direct_sidebar_hides_vllm_connection_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWS_MODEL_PROVIDER", "finetuned")

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    backend = next(
        item
        for item in app.sidebar.selectbox
        if item.label == "Backend"
    )
    assert backend.value == "Direct fine-tuned model"
    assert not app.sidebar.text_input
    assert not app.sidebar.slider
    assert all(
        button.label != "Check connection"
        for button in app.sidebar.button
    )


def test_empty_extraction_stops_before_controller_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWS_MODEL_PROVIDER", "finetuned")

    def fail_if_loaded(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            f"controller loaded for empty input: {args}, {kwargs}"
        )

    monkeypatch.setattr(
        streamlit_app,
        "build_streaming_controller",
        fail_if_loaded,
    )
    streamlit_app.get_controller.clear()
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    extract = next(
        button
        for button in app.button
        if button.label == "Extract information"
    )
    extract.click().run(timeout=10)

    assert not app.exception
    assert any("Paste a story" in error.value for error in app.error)


def test_controller_resource_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ui.settings import InferenceSettings

    created: list[InferenceSettings] = []

    def fake_builder(provider: str, **kwargs: object) -> object:
        created.append(InferenceSettings(provider=provider, **kwargs))
        return {"provider": provider, "settings": kwargs}

    monkeypatch.setattr(
        streamlit_app,
        "build_streaming_controller",
        fake_builder,
    )
    streamlit_app.get_controller.clear()
    settings = InferenceSettings.direct()
    first = streamlit_app.get_controller(settings)
    second = streamlit_app.get_controller(settings)

    assert first is second
    assert created == [settings]
    streamlit_app.get_controller.clear()
