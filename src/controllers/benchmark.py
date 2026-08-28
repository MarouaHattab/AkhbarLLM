import random
from collections.abc import Callable, Iterable, Sequence
from time import perf_counter

from faker import Faker
from tqdm.auto import tqdm

from src.models.benchmark import BenchmarkResult
from src.models.language_model import (
    ChatMessage,
    TokenizedLanguageModel,
)


def generate_arabic_prompts(
    sample_count: int,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if min_chars < 1 or max_chars < 1:
        raise ValueError("prompt lengths must be at least 1")
    if min_chars > max_chars:
        raise ValueError(
            "minimum prompt length cannot exceed maximum prompt length"
        )

    fake = Faker("ar")
    fake.seed_instance(seed)
    random_source = random.Random(seed)
    return [
        fake.text(
            max_nb_chars=random_source.randint(min_chars, max_chars)
        )
        for _ in range(sample_count)
    ]


def benchmark_model(
    runtime: TokenizedLanguageModel,
    prompts: Sequence[str],
    *,
    clock: Callable[[], float] = perf_counter,
    progress: Callable[..., Iterable[str]] = tqdm,
) -> BenchmarkResult:
    if not isinstance(runtime, TokenizedLanguageModel):
        raise TypeError(
            "Benchmark runtime must provide local token-counting methods"
        )
    if not prompts:
        raise ValueError("Benchmark requires at least one prompt")

    input_tokens = 0
    output_tokens = 0
    started_at = clock()

    for prompt in progress(
        prompts,
        desc=f"Benchmarking {runtime.provider}",
        unit="prompt",
    ):
        messages: list[ChatMessage] = [
            {"role": "user", "content": prompt}
        ]
        response = runtime.generate(messages)
        input_tokens += runtime.count_input_tokens(messages)
        output_tokens += runtime.count_output_tokens(response)

    total_seconds = clock() - started_at
    return BenchmarkResult(
        provider=runtime.provider,
        model_id=runtime.model_id,
        sample_count=len(prompts),
        total_seconds=total_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
