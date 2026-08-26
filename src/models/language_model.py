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
