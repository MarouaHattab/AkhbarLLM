from typing import Any

import torch

from src.utils.text import contains_chinese_characters


class ChineseTokenSuppressor:
    """Mask tokenizer vocabulary entries containing Chinese characters."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.mask: torch.Tensor | None = None

    def _build_mask(self, vocabulary_size: int) -> torch.Tensor:
        token_ids = torch.arange(vocabulary_size).unsqueeze(1)
        decoded_tokens = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
        )
        return torch.tensor(
            [
                contains_chinese_characters(token)
                for token in decoded_tokens
            ],
            dtype=torch.bool,
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        del input_ids
        if self.mask is None:
            self.mask = self._build_mask(scores.size(-1))

        scores[:, self.mask.to(scores.device)] = -float("inf")
        return scores
