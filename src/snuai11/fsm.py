"""FSM-constrained decoding for the generative fallback path.

The primary score24 head never generates (parsing failure is structurally
impossible). This LogitsProcessor exists for the optional CoT escalation
path: it forces the FIRST generated answer token into the 24 letter-token
region, so parsing failure stays at 0% there too.
"""

from __future__ import annotations

import torch

from . import perm


def letter_token_ids(tokenizer) -> list[int]:
    """Token ids of the 24 single-letter answers A..X (single-token each)."""
    ids = []
    for letter in perm.LETTERS:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"letter {letter!r} is not a single token: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != perm.N_CLASSES:
        raise ValueError("letter token ids are not distinct")
    return ids


class LetterConstraintProcessor:
    """transformers-compatible LogitsProcessor: constrain position 0 of the
    generation to the 24 letter tokens; free afterwards."""

    def __init__(self, letter_ids: list[int], prompt_len: int):
        self.letter_ids = letter_ids
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] == self.prompt_len:  # first generated token
            mask = torch.full_like(scores, float("-inf"))
            idx = torch.tensor(self.letter_ids, device=scores.device)
            mask[:, idx] = scores[:, idx]
            return mask
        return scores
