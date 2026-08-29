from collections.abc import Iterator
from typing import Protocol, TypedDict, runtime_checkable


class ChatMessage(TypedDict):
    role: str
    content: str


@runtime_checkable
class LanguageModel(Protocol):
    provider: str
    model_id: str

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate text from normalized chat messages."""
        ...


@runtime_checkable
class StreamingLanguageModel(LanguageModel, Protocol):
    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield decoded model output as generation produces it."""
        ...


@runtime_checkable
class TokenizedLanguageModel(LanguageModel, Protocol):
    def count_input_tokens(self, messages: list[ChatMessage]) -> int:
        """Count the exact local tokens used for the generation prompt."""
        ...

    def count_output_tokens(self, text: str) -> int:
        """Count generated local-model tokens without extra special tokens."""
        ...
