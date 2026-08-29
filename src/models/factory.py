from collections.abc import Callable
from typing import Any

from src.helpers.environment import require_api_key
from src.models.language_model import LanguageModel


def _authenticate_huggingface() -> str:
    from src.helpers.huggingface import authenticate_huggingface

    return authenticate_huggingface()


def _create_gemini_client(**kwargs: Any) -> Any:
    from google import genai

    return genai.Client(**kwargs)


def _create_openai_client(**kwargs: Any) -> Any:
    from openai import OpenAI

    return OpenAI(**kwargs)


def _load_qwen(token: str) -> LanguageModel:
    from src.models.qwen import QwenModel

    return QwenModel.load(token)


def _load_finetuned(token: str) -> LanguageModel:
    from src.models.finetuned_qwen import FineTunedQwenModel

    return FineTunedQwenModel.load(token)


def _create_gemini_model(client: Any) -> LanguageModel:
    from src.models.gemini import GeminiModel

    return GeminiModel(client)


def _create_openai_model(client: Any) -> LanguageModel:
    from src.models.openai import OpenAIModel

    return OpenAIModel(client)


def _load_vllm() -> LanguageModel:
    from src.models.vllm import VLLMModel

    return VLLMModel.load()


def load_language_model(
    provider: str,
    *,
    huggingface_authenticator: Callable[[], str] = _authenticate_huggingface,
    secret_loader: Callable[[str], str] = require_api_key,
    qwen_loader: Callable[[str], LanguageModel] = _load_qwen,
    finetuned_loader: Callable[[str], LanguageModel] = _load_finetuned,
    gemini_client_factory: Callable[..., Any] = _create_gemini_client,
    gemini_model_factory: Callable[[Any], LanguageModel] = _create_gemini_model,
    openai_client_factory: Callable[..., Any] = _create_openai_client,
    openai_model_factory: Callable[[Any], LanguageModel] = _create_openai_model,
    vllm_loader: Callable[[], LanguageModel] = _load_vllm,
) -> LanguageModel:
    normalized = provider.strip().casefold()
    if normalized not in {
        "qwen",
        "finetuned",
        "gemini",
        "openai",
        "vllm",
    }:
        raise ValueError(f"Unsupported model provider: {provider}")

    if normalized == "vllm":
        return vllm_loader()

    if normalized in {"qwen", "finetuned"}:
        token = huggingface_authenticator()
        loader = (
            qwen_loader
            if normalized == "qwen"
            else finetuned_loader
        )
        return loader(token)

    if normalized == "gemini":
        api_key = secret_loader("GEMINI_API_KEY")
        client = gemini_client_factory(api_key=api_key)
        return gemini_model_factory(client)

    api_key = secret_loader("OPENAI_API_KEY")
    client = openai_client_factory(api_key=api_key)
    return openai_model_factory(client)
