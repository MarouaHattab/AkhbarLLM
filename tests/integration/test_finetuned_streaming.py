import os
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import pytest
import torch

from src.controllers.streaming import NewsStreamingController
from src.helpers.config import DEFAULT_STORY_PATH, FINETUNED_MODEL_DIR
from src.helpers.environment import read_optional_setting
from src.helpers.huggingface import authenticate_huggingface
from src.models.finetuned_qwen import FineTunedQwenModel
from src.models.news import NewsDetails
from src.models.translation import TranslatedStory


pytestmark = [
    pytest.mark.model_smoke,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_SMOKE") != "1",
        reason="set RUN_MODEL_SMOKE=1 to run the real model",
    ),
]


@pytest.fixture(scope="module")
def controller() -> NewsStreamingController:
    token = authenticate_huggingface()
    adapter_source = (
        read_optional_setting("FINETUNED_ADAPTER_SOURCE")
        or FINETUNED_MODEL_DIR
    )
    started = perf_counter()
    runtime = FineTunedQwenModel.load(
        token,
        adapter_path=adapter_source,
    )
    print(
        f"model_load={perf_counter() - started:.3f}s "
        f"device={runtime.model.device}"
    )
    return NewsStreamingController(runtime)


@pytest.fixture(scope="module")
def story() -> str:
    return Path(DEFAULT_STORY_PATH).read_text(encoding="utf-8").strip()


def collect_timed_chunks(chunks: Iterable[str], label: str) -> list[str]:
    iterator = iter(chunks)
    started = perf_counter()
    first = next(iterator)
    first_chunk_seconds = perf_counter() - started
    collected = [first, *list(iterator)]
    total_seconds = perf_counter() - started
    peak_vram_gib = (
        torch.cuda.max_memory_allocated() / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"{label}: first_chunk={first_chunk_seconds:.3f}s "
        f"total={total_seconds:.3f}s peak_vram={peak_vram_gib:.3f}GiB"
    )
    return collected


def test_real_finetuned_extraction_streams_and_validates(
    controller: NewsStreamingController,
    story: str,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    chunks = collect_timed_chunks(
        controller.stream_task("extraction", story),
        "extraction",
    )
    assert chunks
    result = controller.validate_task_response(
        "extraction",
        "".join(chunks),
    )
    assert isinstance(result, NewsDetails)


def test_real_finetuned_translation_streams_and_validates(
    controller: NewsStreamingController,
    story: str,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    chunks = collect_timed_chunks(
        controller.stream_task(
            "translation",
            story,
            source_language="Arabic",
            target_language="English",
        ),
        "translation",
    )
    assert chunks
    result = controller.validate_task_response(
        "translation",
        "".join(chunks),
    )
    assert isinstance(result, TranslatedStory)
