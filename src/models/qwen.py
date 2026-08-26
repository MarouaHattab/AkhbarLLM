from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers.config import (
    BASE_MODEL_ID,
    MODEL_CACHE_DIR,
    generation_kwargs,
)


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
            device="cuda",
            dtype=torch.float16,
            device_map="auto",
        )

    return RuntimeTarget(
        device="cpu",
        dtype=torch.float32,
        device_map=None,
    )


class QwenRuntime:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        token: str,
        target: RuntimeTarget | None = None,
        model_loader: Callable[..., Any] = (
            AutoModelForCausalLM.from_pretrained
        ),
        tokenizer_loader: Callable[..., Any] = (
            AutoTokenizer.from_pretrained
        ),
    ) -> "QwenRuntime":
        """Load the configured base model and tokenizer."""
        selected = target or select_runtime_target()
        model_kwargs: dict[str, Any] = {
            "cache_dir": MODEL_CACHE_DIR,
            "token": token,
            "torch_dtype": selected.dtype,
        }
        if selected.device_map is not None:
            model_kwargs["device_map"] = selected.device_map

        model = model_loader(BASE_MODEL_ID, **model_kwargs)
        if selected.device_map is None:
            model.to(selected.device)
        model.eval()

        tokenizer = tokenizer_loader(
            BASE_MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            token=token,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer)

    def generate(self, messages: list[dict[str, str]]) -> str:
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

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs(),
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
