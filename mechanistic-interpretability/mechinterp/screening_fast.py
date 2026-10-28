"""Resumable, batched, teacher-forced steering screening.

Numerical contract matches models.continuation_scores: tokenized prompt plus
separately tokenized option, left padding, identical intervention positions,
mean option log probability, and supporting-minus-opposing contrast.

Only token IDs, masks, and scalar scores are cached; never model activations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from .common import (adapter_dir, control_for, digest, load_screening, location,
                     read_data, run_dir, scenario_prompt, write_json, write_rows)
from .models import (blocks, load_model, replace_tensor_output, site_module,
                     tensor_output, unload)


SCORER_VERSION = "screening_batched_v1"


@dataclass
class CandidateBatch:
    ids: torch.Tensor
    mask: torch.Tensor
    scored: torch.Tensor
    prompt_positions: torch.Tensor
    # Pairing remains supporting, opposing, supporting, opposing.


def prepare_candidate_batches(tokenizer, prompts, positive, negative, batch_size, max_length):
    """Tokenize once, preserving the original model scorer's prefix boundaries.

    Prepared inputs reside on CPU until they are transferred to the inference
    device; no hidden states, logits, or gradients are cached.
    """
    if not (len(prompts) == len(positive) == len(negative)):
        raise ValueError("Mismatched scenario and option counts")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Padding token is required")
    paired = []
    for prompt, pos, neg in zip(prompts, positive, negative):
        prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if not prefix:
            raise ValueError("Empty evaluation prefix")
        for option in (pos, neg):
            tail = tokenizer(option, add_special_tokens=False)["input_ids"]
            if not tail:
                raise ValueError("Empty evaluation continuation")
            sequence = prefix + tail
            if len(sequence) > max_length:
                raise ValueError("Candidate exceeds maximum scoring length")
            paired.append((sequence, len(prefix)))
    prepared = []
    for start in range(0, len(paired), 2 * batch_size):
        pairs = paired[start:start + 2 * batch_size]
        width = max(len(seq) for seq, _ in pairs)
        ids = torch.full((len(pairs), width), int(pad), dtype=torch.long)
        mask = torch.zeros_like(ids)
        scored = torch.zeros_like(ids, dtype=torch.bool)
        positions = []
        for i, (sequence, prefix_length) in enumerate(pairs):
            left = width - len(sequence)
            ids[i, left:] = torch.tensor(sequence, dtype=torch.long)
            mask[i, left:] = 1
            last_prefix = left + prefix_length - 1
            positions.append(last_prefix)
            scored[i, last_prefix + 1:] = True
        prepared.append(CandidateBatch(
            ids=ids, mask=mask, scored=scored,
            prompt_positions=torch.tensor(positions, dtype=torch.long)))
    return prepared


def on_device(prepared, device):
    return [CandidateBatch(*(getattr(batch, field).to(device) for field
                             in ("ids", "mask", "scored", "prompt_positions")))
            for batch in prepared]


@torch.inference_mode()
def score_prepared(model, prepared, layer=None, site=None, direction=None,
                   strength=0., policy="prompt_last"):
    """Score prepared teacher-forced candidates, optionally intervening.

    The zero-strength baseline performs NO hook and is independent of layer,
    site, or token policy. Nonzero interventions use the legacy hook equations.
    """
    if policy not in ("prompt_last", "generation"):
        raise ValueError(policy)
    intervention = float(strength) != 0.
    if intervention and (layer is None or site is None or direction is None):
        raise ValueError("Nonzero steering requires layer, site and direction")
    all_scores = []
    vector = ((strength * direction).to(
        device=model.device, dtype=next(model.parameters()).dtype)
              if intervention else None)
    for batch in prepared:
        ids, mask, scored, positions = (
            batch.ids, batch.mask, batch.scored, batch.prompt_positions)
        if intervention:
            width = ids.shape[1]
            row_ids = torch.arange(len(ids), device=ids.device)
            if policy == "generation":
                indices = torch.arange(width, device=ids.device)[None, :]
                region = ((indices >= positions[:, None]) & mask.bool()).unsqueeze(-1)

            def hook(_module, _inputs, output):
                hidden = tensor_output(output)
                changed = hidden.clone()
                if policy == "prompt_last":
                    changed[row_ids, positions, :] += vector
                else:
                    changed += region.to(changed.dtype) * vector
                return replace_tensor_output(output, changed)

            handle = site_module(model, layer, site).register_forward_hook(hook)
            try:
                result = model(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                handle.remove()
        else:
            result = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = result.logits[:, :-1].float()
        targets = ids[:, 1:]
        log_scores = (logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                      - torch.logsumexp(logits, dim=-1))
        target_mask = scored[:, 1:]
        means = ((log_scores * target_mask).sum(-1)
                 / target_mask.sum(-1).clamp_min(1))
        values = means.detach().tolist()
        all_scores.extend(values[i] - values[i + 1]
                          for i in range(0, len(values), 2))
        del result, logits, log_scores, means
    return all_scores


def identity(cfg, family, adapter, scenarios, batch_size):
    """Versioned content identity for comparable baseline reuse and resume.

    Adapter model bytes are hashed: a different checkpoint must never inherit
    a previous unsteered baseline or partial screening grid.
    """
    adapter_hash = None
    if adapter != "base":
        checkpoint = adapter_dir(cfg, family, adapter)
        adapter_hash = {
            "weights_sha256": digest(checkpoint / "adapter_model.safetensors"),
            "config_sha256": digest(checkpoint / "adapter_config.json"),
        }
    dataset_json = json.dumps(scenarios, sort_keys=True, ensure_ascii=False)
    return {
        "scorer_version": SCORER_VERSION, "family": family, "adapter": adapter,
        "model_name": cfg["models"][family]["name"],
        "revision": cfg["models"][family]["revision"],
        "adapter_hash": adapter_hash,
        "scenarios_sha256": hashlib.sha256(dataset_json.encode("utf-8")).hexdigest(),
        "batch_size": int(batch_size),
        "max_length": int(cfg["runtime"]["max_length"]),
        "dtype": cfg["runtime"]["dtype"],
        "attention": cfg["runtime"]["attention"],
    }


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def baseline_cache_path(out, adapter, identity_value):
    key = signature(identity_value)[:20]
    return out / ".work" / "screen_cache" / f"baseline_{adapter}_{key}.json"


def read_baseline(path, identity_value, scenarios):
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("identity") != identity_value:
        raise ValueError(f"Screening baseline cache provenance mismatch: {path}")
    scores = obj.get("scores")
    if not isinstance(scores, list) or len(scores) != len(scenarios):
        raise ValueError(f"Incomplete screening baseline cache: {path}")
    expected_ids = [int(x["scenario_id"]) for x in scenarios]
    if obj.get("scenario_ids") != expected_ids:
        raise ValueError(f"Screening baseline scenario order mismatch: {path}")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in scores):
        raise ValueError(f"Invalid baseline scores: {path}")
    return scores


def read_checkpoint(path, feature, adapter, policy, layer, site, fractions, scenarios):
    rows = read_data(path)
    n = len(scenarios)
    if len(rows) != n * len(fractions):
        raise ValueError(f"Incomplete layer/site checkpoint {path}")
    k = 0
    for fraction in fractions:
        for scenario in scenarios:
            r = rows[k]
            if (r.get("feature") != feature or r.get("adapter") != adapter
                    or r.get("policy") != policy or r.get("layer") != layer
                    or r.get("site") != site
                    or int(r["scenario_id"]) != int(scenario["scenario_id"])
                    or r.get("split") != scenario["split"]
                    or r.get("category") != scenario["category"]
                    or r.get("tradeoff") != scenario["tradeoff"]
                    or r.get("strength_convention") != "relative_activation_norm_v2"
                    or not math.isclose(float(r["target_relative_norm_fraction"]),
                                        float(fraction), rel_tol=0, abs_tol=1e-10)
                    or not math.isfinite(float(r["score"]))):
                raise ValueError(f"Incompatible screening checkpoint at {path}: row {k}")
            k += 1
    return rows


def ensure_manifest(path, value):
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != value:
            raise ValueError(
                f"Incompatible screening checkpoint manifest: {path}. "
                "Archive the checkpoint directory before changing screening "
                "configuration, direction artifacts or batch size.")
    else:
        write_json(path, value)


@torch.inference_mode()
def screen_fast(cfg, family, feature, adapter, policy="prompt_last"):
    """Process every layer/site, checkpoint each completed grid, then assemble.

    Compatible restarts skip completed layer/site grids, and finished result
    files retain exactly the original screen_*.jsonl schema.
    """
    if adapter not in (control_for(feature), feature):
        raise ValueError("Feature direction may only be applied to its own adapter or control")
    if policy not in ("prompt_last", "generation"):
        raise ValueError(policy)
    from .directions import load_direction_bundle

    scenarios = [r for r in load_screening(cfg)
                 if r["split"] in cfg["steering"]["screening_splits"]]
    if not scenarios:
        raise ValueError("No screening/validation scenarios")
    batch_size = int(cfg["steering"].get("screening", {}).get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("Invalid screening batch size")
    fractions = [float(x) for x in cfg["steering"].get(
        "relative_norm_fractions", [-.25, -.1, -.03, 0., .03, .1, .25])]
    if 0. not in fractions or len(set(fractions)) != len(fractions):
        raise ValueError("Screening strengths must be unique and include zero")
    out = run_dir(cfg, family, "activation-steering")
    stem = f"{feature}_{adapter}_{policy}"
    final = out / f"screen_{stem}.jsonl"
    checkpoint_dir = out / ".work" / "screen_checkpoints" / stem
    manifest_path = checkpoint_dir / "manifest.json"
    # Direction preparation may load the base model. Finish before loading adapter.
    get_direction = load_direction_bundle(cfg, family, feature)
    baseline_identity = identity(cfg, family, adapter, scenarios, batch_size)
    manifest = {
        "identity": baseline_identity, "feature": feature, "policy": policy,
        "sites": list(cfg["runtime"]["sites"]), "fractions": fractions,
        "directions_sha256": digest(out / f"directions_{feature}.safetensors"),
        "calibration_sha256": digest(out / "site_calibration_v2.json"),
    }
    if final.exists() and not manifest_path.exists():
        raise FileExistsError(
            f"Existing screening output without optimized checkpoint manifest: {final}. "
            "Use --resume to skip it, or archive it before running optimized screening.")
    ensure_manifest(manifest_path, manifest)
    target, tok = load_model(cfg, family, adapter)
    try:
        prompts = [scenario_prompt(r["scenario"]) for r in scenarios]
        positive = [r["welfare_supporting"] for r in scenarios]
        negative = [r["welfare_opposing"] for r in scenarios]
        prepared = on_device(prepare_candidate_batches(
            tok, prompts, positive, negative, batch_size,
            int(cfg["runtime"]["max_length"])), target.device)
        n_layers = len(blocks(target))
        baseline_path = baseline_cache_path(out, adapter, baseline_identity)
        baseline = read_baseline(baseline_path, baseline_identity, scenarios)
        if baseline is None:
            baseline = score_prepared(target, prepared)
            if len(baseline) != len(scenarios) or any(
                    not math.isfinite(x) for x in baseline):
                raise ValueError("Invalid unsteered baseline scores")
            write_json(baseline_path, {
                "identity": baseline_identity,
                "scenario_ids": [int(r["scenario_id"]) for r in scenarios],
                "scores": baseline,
            })
            print(f"Cached baseline: {adapter} ({len(baseline)} scenarios)", flush=True)
        else:
            print(f"Reusing baseline: {adapter} ({len(baseline)} scenarios)", flush=True)
        all_rows = []
        for layer in range(n_layers):
            for site in cfg["runtime"]["sites"]:
                checkpoint = checkpoint_dir / f"layer_{layer:03d}_{site}.jsonl"
                if checkpoint.exists():
                    rows = read_checkpoint(checkpoint, feature, adapter, policy,
                                           layer, site, fractions, scenarios)
                    print(f"Resuming verified layer/site: {stem} {layer}/{site}", flush=True)
                else:
                    unit, scale = get_direction(layer, site)
                    direction = unit * scale
                    rows = []
                    for fraction in fractions:
                        strength = float(fraction) * math.sqrt(unit.numel())
                        scores = (baseline if strength == 0. else score_prepared(
                            target, prepared, layer, site, direction, strength, policy))
                        if len(scores) != len(scenarios):
                            raise ValueError("Incomplete screening results")
                        for scenario, value in zip(scenarios, scores):
                            if not math.isfinite(float(value)):
                                raise ValueError(
                                    f"Nonfinite screening score: {stem} {layer}/{site}/{fraction}")
                            rows.append({
                                "feature": feature, "adapter": adapter, "layer": layer,
                                "site": site, "strength": strength, "policy": policy,
                                "scenario_id": scenario["scenario_id"],
                                "split": scenario["split"], "category": scenario["category"],
                                "tradeoff": scenario["tradeoff"], "score": float(value),
                                "calibration_scale": scale,
                                "target_relative_norm_fraction": float(fraction),
                                "strength_convention": "relative_activation_norm_v2",
                            })
                    write_rows(checkpoint, rows)
                    print(f"Screened {stem}: layer {layer + 1}/{n_layers} {site}", flush=True)
                all_rows.extend(rows)
        if final.exists():
            existing = read_data(final)
            if existing != all_rows:
                raise ValueError(f"Existing final screening output differs: {final}")
        else:
            write_rows(final, all_rows)
        return final
    finally:
        unload(target)
