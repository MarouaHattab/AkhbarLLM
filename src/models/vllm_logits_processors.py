from collections.abc import Callable
from typing import Protocol

import torch
from transformers import AutoTokenizer

from src.helpers.config import VLLM_BASE_MODEL_ID
from src.utils.text import contains_chinese_characters


class Tokenizer(Protocol):
    def __len__(self) -> int: ...

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]: ...


class ChineseTokenSuppressor:
    """Mask vocabulary tokens containing Chinese characters for vLLM."""

    _blocked_token_ids_cache: dict[str, torch.LongTensor] = {}

    def __init__(
        self,
        model_id: str = VLLM_BASE_MODEL_ID,
        tokenizer_loader: Callable[[str], Tokenizer] = AutoTokenizer.from_pretrained,
    ) -> None:
        self.model_id = model_id
        if model_id not in self._blocked_token_ids_cache:
            tokenizer = tokenizer_loader(model_id)
            token_ids = torch.arange(len(tokenizer)).unsqueeze(1)
            decoded_tokens = tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=True,
            )
            blocked_token_ids = [
                token_id
                for token_id, token in enumerate(decoded_tokens)
                if contains_chinese_characters(token)
            ]
            self._blocked_token_ids_cache[model_id] = torch.tensor(
                blocked_token_ids,
                dtype=torch.long,
            )
        self.blocked_token_ids = self._blocked_token_ids_cache[model_id]
        self._device_blocked_token_ids: dict[torch.device, torch.LongTensor] = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._blocked_token_ids_cache.clear()

    def __call__(self, past_token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        del past_token_ids
        device = logits.device
        blocked_token_ids = self._device_blocked_token_ids.get(device)
        if blocked_token_ids is None:
            blocked_token_ids = self.blocked_token_ids.to(device)
            self._device_blocked_token_ids[device] = blocked_token_ids
        logits[blocked_token_ids] = -float("inf")
        return logits
