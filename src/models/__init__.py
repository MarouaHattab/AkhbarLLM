import importlib


_EXPORT_MODULES = {
    "Entity": "src.models.news",
    "EvaluationResult": "src.models.evaluation",
    "FineTunedQwenModel": "src.models.finetuned_qwen",
    "GeminiModel": "src.models.gemini",
    "ChatMessage": "src.models.language_model",
    "LanguageModel": "src.models.language_model",
    "NewsDetails": "src.models.news",
    "OpenAIModel": "src.models.openai",
    "VLLMModel": "src.models.vllm",
    "QwenModel": "src.models.qwen",
    "QwenRuntime": "src.models.qwen",
    "RuntimeTarget": "src.models.qwen",
    "TranslatedStory": "src.models.translation",
    "select_runtime_target": "src.models.qwen",
}

__all__ = [
    "Entity",
    "EvaluationResult",
    "FineTunedQwenModel",
    "GeminiModel",
    "ChatMessage",
    "LanguageModel",
    "NewsDetails",
    "OpenAIModel",
    "VLLMModel",
    "QwenModel",
    "QwenRuntime",
    "RuntimeTarget",
    "TranslatedStory",
    "select_runtime_target",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
