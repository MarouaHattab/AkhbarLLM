from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers.config import (
    MODEL_CACHE_DIR,
    QWEN_CPU_DEVICE,
    QWEN_CPU_DTYPE,
    QWEN_CUDA_DEVICE,
    QWEN_CUDA_DTYPE,
    QWEN_DEVICE_MAP,
    QWEN_MODEL_ID,
    qwen_generation_kwargs,
)
from src.models.language_model import ChatMessage


@dataclass(frozen=True)
class RuntimeTarget:
    device: str
    dtype: torch.dtype
    device_map: str | None


def select_runtime_target(
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
) -> RuntimeTarget:
    """Choose CUDA/float16 when available, otherwise CPU/float32."""
    if cuda_available():
        return RuntimeTarget(
            device=QWEN_CUDA_DEVICE,
            dtype=getattr(torch, QWEN_CUDA_DTYPE),
            device_map=QWEN_DEVICE_MAP,
        )

    return RuntimeTarget(
        device=QWEN_CPU_DEVICE,
        dtype=getattr(torch, QWEN_CPU_DTYPE),
        device_map=None,
    )


class QwenModel:
    provider = "qwen"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str = QWEN_MODEL_ID,
        logits_processors: list[Callable[[Any, Any], Any]] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.logits_processors = list(logits_processors or [])

    @classmethod
    def load(
        cls,
        token: str,
        model_id: str = QWEN_MODEL_ID,
        target: RuntimeTarget | None = None,
        model_loader: Callable[..., Any] = (
            AutoModelForCausalLM.from_pretrained
        ),
        tokenizer_loader: Callable[..., Any] = (
            AutoTokenizer.from_pretrained
        ),
    ) -> "QwenModel":
        """Load the configured base model and tokenizer."""
        selected = target or select_runtime_target()
        model_kwargs: dict[str, Any] = {
            "cache_dir": MODEL_CACHE_DIR,
            "token": token,
            "torch_dtype": selected.dtype,
        }
        if selected.device_map is not None:
            model_kwargs["device_map"] = selected.device_map

        model = model_loader(model_id, **model_kwargs)
        if selected.device_map is None:
            model.to(selected.device)
        model.eval()

        tokenizer = tokenizer_loader(
            model_id,
            cache_dir=MODEL_CACHE_DIR,
            token=token,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, model_id=model_id)

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate and decode only tokens produced after the prompt."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generation_kwargs = qwen_generation_kwargs()
        if self.logits_processors:
            generation_kwargs["logits_processor"] = self.logits_processors

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )

        new_token_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids,
            )
        ]
        return self.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
        )[0].strip()


QwenRuntime = QwenModel
