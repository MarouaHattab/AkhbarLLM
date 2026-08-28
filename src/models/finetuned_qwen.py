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
        adapter_dir = validate_adapter_path(adapter_path)
        runtime = super().load(
            token=token,
            model_id=model_id,
            target=target,
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
        )

        try:
            runtime.model.load_adapter(str(adapter_dir))
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the fine-tuned adapter from: "
                f"{adapter_dir}"
            ) from exc

        runtime.model.eval()
        runtime.model_id = str(adapter_dir)
        runtime.logits_processors.append(
            ChineseTokenSuppressor(runtime.tokenizer)
        )

        return runtime
