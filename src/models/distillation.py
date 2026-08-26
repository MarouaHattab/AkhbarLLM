from dataclasses import dataclass
from pathlib import Path

from src.helpers.config import (
    O4_MINI_CACHED_INPUT_PRICE_PER_1M_TOKENS,
    O4_MINI_INPUT_PRICE_PER_1M_TOKENS,
    O4_MINI_OUTPUT_PRICE_PER_1M_TOKENS,
)


TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_prompt_tokens=(
                self.cached_prompt_tokens + other.cached_prompt_tokens
            ),
            completion_tokens=(
                self.completion_tokens + other.completion_tokens
            ),
        )


@dataclass(frozen=True)
class OpenAIGeneration:
    text: str
    usage: TokenUsage


@dataclass(frozen=True)
class DistillationStats:
    successful: int
    failed: int
    usage: TokenUsage
    cost: "DistillationCost"
    output_path: Path


@dataclass(frozen=True)
class DistillationCost:
    uncached_input_cost: float
    cached_input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return (
            self.uncached_input_cost
            + self.cached_input_cost
            + self.output_cost
        )


def calculate_distillation_cost(usage: TokenUsage) -> DistillationCost:
    cached_tokens = min(usage.cached_prompt_tokens, usage.prompt_tokens)
    uncached_tokens = usage.prompt_tokens - cached_tokens
    return DistillationCost(
        uncached_input_cost=(
            uncached_tokens
            / TOKENS_PER_MILLION
            * O4_MINI_INPUT_PRICE_PER_1M_TOKENS
        ),
        cached_input_cost=(
            cached_tokens
            / TOKENS_PER_MILLION
            * O4_MINI_CACHED_INPUT_PRICE_PER_1M_TOKENS
        ),
        output_cost=(
            usage.completion_tokens
            / TOKENS_PER_MILLION
            * O4_MINI_OUTPUT_PRICE_PER_1M_TOKENS
        ),
    )
