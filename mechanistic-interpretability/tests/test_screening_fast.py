"""Numerical parity and resumability of optimized teacher-forced screening."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mechinterp.models import continuation_scores
from mechinterp.screening_fast import (
    identity, prepare_candidate_batches, on_device, score_prepared,
    read_checkpoint, read_baseline, ensure_manifest,
)


class Tokenizer:
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": ([1] if add_special_tokens else [])
                + [3 + ord(ch) % 17 for ch in text]}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(12, 12, bias=False)
    def forward(self, x):
        return x + self.mlp(x)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(32, 12)
        self.layers = nn.ModuleList([Block()])
        self.head = nn.Linear(12, 32)
        # site_module() finds model.layers
        self.model = SimpleNamespace(layers=self.layers)
    @property
    def device(self):
        return self.emb.weight.device
    def forward(self, input_ids, attention_mask, use_cache=False):
        x = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.head(x))


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("policy", ["prompt_last", "generation"])
@pytest.mark.parametrize("strength", [0., 0.7])
def test_scores_match_original(size, policy, strength):
    torch.manual_seed(13)
    model = LM().eval()
    tok = Tokenizer()
    prompts = ["p", "a longer prefix", "xy", "long"]
    pos = ["yes", "good", "a", "zz"]
    neg = ["no", "bad", "bbb", "x"]
    prepared = on_device(prepare_candidate_batches(tok, prompts, pos, neg, size, 128), "cpu")
    direction = torch.arange(12, dtype=torch.float) / 50
    original = continuation_scores(model, tok, prompts, pos, neg, 0, "mlp",
                                   direction, strength, policy, 128, size)
    fast = score_prepared(model, prepared, 0, "mlp", direction, strength, policy)
    assert fast == pytest.approx(original, abs=1e-6, rel=1e-6)


def test_baseline_is_policy_and_layer_invariant():
    torch.manual_seed(14)
    model = LM().eval()
    tok = Tokenizer()
    prepared = on_device(prepare_candidate_batches(
        tok, ["prompt", "other"], ["yes", "ok"], ["no", "bad"], 2, 128), "cpu")
    base = score_prepared(model, prepared)
    for policy in ("prompt_last", "generation"):
        for site in ("mlp", "residual"):
            legacy = score_prepared(model, prepared, 0, site, torch.ones(12), 0., policy)
            assert base == legacy


def test_checkpoint_contract_and_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    ensure_manifest(manifest, {"scorer_version": "v1", "batch_size": 8})
    ensure_manifest(manifest, {"scorer_version": "v1", "batch_size": 8})
    with pytest.raises(ValueError, match="Incompatible"):
        ensure_manifest(manifest, {"scorer_version": "v1", "batch_size": 16})
    scenarios = [{"scenario_id": 1, "split": "validation", "category": "id", "tradeoff": "yes"}]
    template = {"feature": "voice", "adapter": "canonical", "policy": "generation",
                "layer": 2, "site": "mlp", "scenario_id": 1, "split": "validation",
                "category": "id", "tradeoff": "yes", "score": 1.5,
                "strength_convention": "relative_activation_norm_v2"}
    rows = [{**template, "strength": f*12, "target_relative_norm_fraction": f}
            for f in [-.25, 0., .25]]
    path = tmp_path / "layer.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert read_checkpoint(path, "voice", "canonical", "generation", 2, "mlp",
                           [-.25, 0., .25], scenarios) == rows
    with pytest.raises(ValueError):
        read_checkpoint(path, "voice", "canonical", "generation", 2, "residual",
                        [-.25, 0., .25], scenarios)


def test_baseline_cache_provenance(tmp_path):
    scenarios = [{"scenario_id": 1}]
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"identity": {"batch_size": 8},
                                "scenario_ids": [1], "scores": [0.3]}))
    assert read_baseline(path, {"batch_size": 8}, scenarios) == [0.3]
    with pytest.raises(ValueError):
        read_baseline(path, {"batch_size": 16}, scenarios)
