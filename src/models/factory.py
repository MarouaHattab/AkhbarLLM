from collections.abc import Callable
from typing import Any

from src.helpers.environment import require_api_key
from src.helpers.huggingface import authenticate_huggingface
from src.models.gemini import GeminiModel
from src.models.language_model import LanguageModel
from src.models.qwen import QwenModel


def _create_gemini_client(**kwargs: Any) -> Any:
    from google import genai

    return genai.Client(**kwargs)


def load_language_model(
    provider: str,
    *,
    huggingface_authenticator: Callable[[], str] = authenticate_huggingface,
    secret_loader: Callable[[str], str] = require_api_key,
    qwen_loader: Callable[[str], LanguageModel] = QwenModel.load,
    gemini_client_factory: Callable[..., Any] = _create_gemini_client,
    gemini_model_factory: Callable[[Any], LanguageModel] = GeminiModel,
) -> LanguageModel:
    normalized = provider.strip().casefold()
    if normalized not in {"qwen", "gemini"}:
        raise ValueError(f"Unsupported model provider: {provider}")

    if normalized == "qwen":
        return qwen_loader(huggingface_authenticator())

    api_key = secret_loader("GEMINI_API_KEY")
    client = gemini_client_factory(api_key=api_key)
    return gemini_model_factory(client)
