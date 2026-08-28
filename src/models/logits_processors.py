from typing import Any

import torch


def contains_chinese_characters(token: str) -> bool:
    """Return whether a token contains a character from a CJK block."""
    return any(
        "\u4e00" <= character <= "\u9fff"
        or "\u3400" <= character <= "\u4dbf"
        or "\uf900" <= character <= "\ufaff"
        for character in token
    )


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
