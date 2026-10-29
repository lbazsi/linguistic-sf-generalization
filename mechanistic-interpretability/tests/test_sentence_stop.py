import torch
import pytest
from types import SimpleNamespace

from mechinterp.sentence_stop import SentenceBoundaryStop, sentence_finished
from mechinterp.batched_generation import decode_generated_batch, PROTOCOL


class Tokenizer:
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        parts = {11: "The ", 12: "best ", 13: "choice.", 14: " Additional", 15: " text", 16: " continues"}
        return "".join(parts.get(int(i), "") for i in ids if int(i) != 2)


def test_sentence_boundary_detection():
    assert sentence_finished("Make the ethical choice.")
    assert sentence_finished("保护动物。")
    assert not sentence_finished("Dr.")
    assert not sentence_finished("Do not stop yet")
    assert not sentence_finished("This is unfinished...")


def test_per_row_stopping_and_minimum():
    tok = Tokenizer()
    stop = SentenceBoundaryStop(tok, 2, 2, 3)
    a = torch.tensor([[1, 1, 11, 12], [1, 1, 11, 12]])
    assert stop(a, None).tolist() == [False, False]
    b = torch.tensor([[1, 1, 11, 12, 13], [1, 1, 11, 12, 14]])
    assert stop(b, None).tolist() == [True, False]
    c = torch.tensor([[1, 1, 11, 12, 13, 2], [1, 1, 11, 12, 14, 15]])
    assert stop(c, None).tolist() == [True, False]
    assert stop.stopped_lengths == [3, None]


def test_decode_trims_individual_stopped_row():
    tok = Tokenizer()
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
    rows = decode_generated_batch(
        torch.tensor([[1, 1, 11, 12, 13, 2], [1, 1, 11, 12, 14, 15]]),
        2, tok, model, 4, [3, None])
    assert rows[0]["stop_reason"] == "sentence_boundary"
    assert rows[0]["generated_tokens"] == 3
    assert rows[0]["continuation"] == "The best choice."
    assert rows[1]["reached_token_limit"]
    assert rows[1]["stop_reason"] == "token_limit"


def test_protocol_is_versioned():
    assert PROTOCOL == "paired_batch_sentence_v2"
