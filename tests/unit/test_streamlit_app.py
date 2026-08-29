from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import streamlit_app


APP_PATH = Path(__file__).resolve().parents[2] / "app.py"


def test_initial_page_is_clean_and_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(provider: str) -> object:
        raise AssertionError(f"model loaded during initial render: {provider}")

    monkeypatch.setattr(
        streamlit_app,
        "build_streaming_controller",
        fail_if_loaded,
    )
    streamlit_app.get_controller.clear()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "ArabLLM Newsroom"
    assert app.radio[0].value == "Information Extraction"
    assert len(app.selectbox) == 0


def test_switching_to_translation_reveals_language_controls() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    app.radio[0].set_value("Translation").run(timeout=10)

    assert not app.exception
    assert len(app.selectbox) == 2
    assert app.selectbox[0].value == "Arabic"
    assert app.selectbox[1].value == "English"


def test_controller_resource_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    def fake_builder(provider: str) -> object:
        created.append(provider)
        return {"provider": provider}

    monkeypatch.setattr(
        streamlit_app,
        "build_streaming_controller",
        fake_builder,
    )
    streamlit_app.get_controller.clear()
    first = streamlit_app.get_controller("finetuned")
    second = streamlit_app.get_controller("finetuned")

    assert first is second
    assert created == ["finetuned"]
    streamlit_app.get_controller.clear()
