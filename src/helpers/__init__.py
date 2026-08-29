import importlib


_EXPORT_MODULES = {
    "BASE_MODEL_ID": "src.helpers.config",
    "authenticate_huggingface": "src.helpers.huggingface",
}

__all__ = [
    "BASE_MODEL_ID",
    "authenticate_huggingface",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
