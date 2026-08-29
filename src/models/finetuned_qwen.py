from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers.config import (
    FINETUNED_MODEL_DIR,
    QWEN_MODEL_ID,
)
from src.models.logits_processors import ChineseTokenSuppressor
from src.models.qwen import QwenModel, RuntimeTarget


ADAPTER_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def validate_adapter_path(adapter_path: str | Path) -> Path:
    """Validate and return the resolved LoRA adapter directory."""
    adapter_dir = Path(adapter_path).resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned adapter directory not found: {adapter_dir}"
        )

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Fine-tuned adapter config not found: {adapter_config_path}"
        )

    if not any(
        (adapter_dir / filename).is_file()
        for filename in ADAPTER_WEIGHT_FILENAMES
    ):
        expected_names = ", ".join(ADAPTER_WEIGHT_FILENAMES)
        raise FileNotFoundError(
            f"Fine-tuned adapter weights not found in {adapter_dir}; "
            f"expected one of: {expected_names}"
        )

    return adapter_dir


def resolve_adapter_source(adapter_source: str | Path) -> str:
    """Return a validated local path or a normalized owner/repository Hub ID."""
    if isinstance(adapter_source, Path):
        return str(validate_adapter_path(adapter_source))

    source = adapter_source.strip()
    if not source:
        raise ValueError("Fine-tuned adapter source must not be empty.")
    candidate = Path(source)
    if candidate.exists():
        return str(validate_adapter_path(candidate))
    if (
        "\\" in source
        or source.startswith((".", "/"))
        or source.count("/") != 1
    ):
        raise FileNotFoundError(
            f"Fine-tuned adapter source not found: {source}"
        )
    owner, repository = source.split("/", maxsplit=1)
    if not owner.strip() or not repository.strip():
        raise ValueError(
            f"Invalid Hugging Face adapter repository: {source}"
        )
    return source


class FineTunedQwenModel(QwenModel):
    """Qwen runtime with the local news-analysis LoRA adapter attached."""

    provider = "finetuned"

    @classmethod
    def load(
        cls,
        token: str,
        model_id: str = QWEN_MODEL_ID,
        adapter_path: str | Path = FINETUNED_MODEL_DIR,
        target: RuntimeTarget | None = None,
        model_loader: Callable[..., Any] = (
            AutoModelForCausalLM.from_pretrained
        ),
        tokenizer_loader: Callable[..., Any] = (
            AutoTokenizer.from_pretrained
        ),
    ) -> "FineTunedQwenModel":
        adapter_source = resolve_adapter_source(adapter_path)
        runtime = super().load(
            token=token,
            model_id=model_id,
            target=target,
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
        )

        try:
            runtime.model.load_adapter(adapter_source, token=token)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the fine-tuned adapter from: "
                f"{adapter_source}"
            ) from exc

        runtime.model.eval()
        runtime.model_id = adapter_source
        runtime.logits_processors.append(
            ChineseTokenSuppressor(runtime.tokenizer)
        )

        return runtime
