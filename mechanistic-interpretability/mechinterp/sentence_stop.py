"""Content-neutral, per-row stopping at a completed sentence boundary.

The scenario prompt remains unchanged. The model samples normally up to a
configurable minimum number of new tokens; the next terminal sentence boundary
ends that row without modifying token probabilities. The 512-token ceiling
remains a separate safety limit.
"""
from __future__ import annotations

import re

import torch
from transformers import StoppingCriteria


_SENTENCE_END = re.compile(r"""[.!?。！？؟][\s"'”’）)\]】」』]*$""")
_ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|etc|mr|mrs|ms|dr|prof|vs|no)\.$", re.IGNORECASE)
_NUMBERED_ITEM = re.compile(r"(?:^|\s)\d+\.$")


def sentence_finished(text: str) -> bool:
    """Conservative surface boundary, not a claim of semantic completeness."""
    stripped = text.rstrip()
    if not stripped or stripped.endswith(("...", "…")):
        return False
    if _ABBREVIATION.search(stripped) or _NUMBERED_ITEM.search(stripped):
        return False
    return bool(_SENTENCE_END.search(stripped))


class SentenceBoundaryStop(StoppingCriteria):
    """Hugging Face per-sequence stopping; finished rows remain finished.

    The generated length is measured from the padded batch's common prompt
    width, not from individual attention-mask lengths. The stopping criterion
    only observes sampled tokens, leaving logits and the source prompt intact.
    """

    def __init__(self, tokenizer, prompt_length: int, batch_size: int, min_new_tokens: int):
        if prompt_length <= 0 or batch_size <= 0 or min_new_tokens <= 0:
            raise ValueError("Invalid sentence-boundary generation settings")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.min_new_tokens = int(min_new_tokens)
        self.stopped_lengths: list[int | None] = [None] * int(batch_size)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if input_ids.shape[0] != len(self.stopped_lengths):
            raise ValueError("Stopping criterion batch size changed during generation")
        generated = int(input_ids.shape[1]) - self.prompt_length
        if generated >= self.min_new_tokens:
            for index, already_stopped in enumerate(self.stopped_lengths):
                if already_stopped is not None:
                    continue
                continuation = self.tokenizer.decode(
                    input_ids[index, self.prompt_length:].tolist(),
                    skip_special_tokens=True,
                )
                if sentence_finished(continuation):
                    self.stopped_lengths[index] = generated
        return torch.tensor(
            [n is not None for n in self.stopped_lengths],
            dtype=torch.bool,
            device=input_ids.device,
        )
