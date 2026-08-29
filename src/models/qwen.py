from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

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

    def count_input_tokens(self, messages: list[ChatMessage]) -> int:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)

    def count_output_tokens(self, text: str) -> int:
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return len(token_ids)

    def _prepare_generation_inputs(
        self,
        messages: list[ChatMessage],
    ) -> Any:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

    def _generation_kwargs(self) -> dict[str, Any]:
        generation_kwargs = qwen_generation_kwargs()
        if self.logits_processors:
            generation_kwargs["logits_processor"] = self.logits_processors
        return generation_kwargs

    def generate(self, messages: list[ChatMessage]) -> str:
        """Generate and decode only tokens produced after the prompt."""
        model_inputs = self._prepare_generation_inputs(messages)
        generation_kwargs = self._generation_kwargs()

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

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield genuine decoder output from a background generation thread."""
        model_inputs = self._prepare_generation_inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = self._generation_kwargs()
        generation_kwargs["streamer"] = streamer
        worker_errors: Queue[BaseException] = Queue(maxsize=1)

        def generate_in_worker() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **model_inputs,
                        **generation_kwargs,
                    )
            except BaseException as exc:
                worker_errors.put(exc)
                streamer.end()

        worker = Thread(
            target=generate_in_worker,
            name="qwen-streaming-generation",
            daemon=True,
        )
        worker.start()
        emitted: list[str] = []
        try:
            for chunk in streamer:
                if chunk:
                    emitted.append(chunk)
                    yield chunk
        finally:
            worker.join()

        if not worker_errors.empty():
            cause = worker_errors.get_nowait()
            raise RuntimeError("Qwen streaming generation failed.") from cause
        if not "".join(emitted).strip():
            raise RuntimeError("Qwen streaming generation returned blank output.")


QwenRuntime = QwenModel
