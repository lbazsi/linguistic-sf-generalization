"""Audit already-trained ridge probes without refitting or selecting on test labels.

Only a development-selected site's test activations are recomputed; saved
training statistics, coefficients, and intercept are used unchanged.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from safetensors.torch import load_file
from .common import FEATURES, load_config, location, run_dir, write_json, write_rows, digest, batches
from .experiments import probe_dataset
from .models import load_model, unload
from .streaming import batch_grid
from .probe_robustness import inspect_heldout


def selected_probe(path:Path):
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:
        raise ValueError(f"Empty ridge metrics file: {path}")
    if any(r.get("method")!="dual_ridge_v2" for r in rows):
        raise ValueError("Cannot audit non-v2 ridge results")
    if len({(r["layer"],r["site"],r["pooling"]) for r in rows})!=len(rows):
        raise ValueError("Duplicate layer/site/pooling combinations")
    selected=min(range(len(rows)),key=lambda i:rows[i]["development"]["rmse"])
    return rows,selected


def evaluate_saved(raw_batch:torch.Tensor,weights,selected_idx):
    """Saved standardized activations must yield original held-out test values."""
    i=selected_idx
    h=raw_batch.float()
    mu=weights["mean"][i].to(h.device,dtype=torch.float32)
    std=weights["std"][i].to(h.device,dtype=torch.float32)
    w=weights["weights"][i].to(h.device,dtype=torch.float32)
    bias=weights["bias"][i].to(h.device,dtype=torch.float32)
    if h.shape[-1]!=len(w):
        raise ValueError("Saved weight dimensionality differs from model activations")
    if not torch.isfinite(h).all() or torch.any(std<=0):
        raise ValueError("Invalid activations or ridge normalization")
    return (((h-mu)/std)*w).sum(-1)+bias


@torch.inference_mode()
def audit(cfg,family,condition,source_version="revision_2"):
    from .ridge import scores
    original=location(cfg,"outputs").parent/source_version/family/"linear-probing"
    metrics_file=original/f"behavior_all_{condition}.jsonl"
    weight_file=original/f"behavior_all_{condition}_weights.safetensors"
    if not metrics_file.is_file() or not weight_file.is_file():
        raise FileNotFoundError(f"Missing completed original probe: {metrics_file} / {weight_file}")
    rows,selected_idx=selected_probe(metrics_file)
    choice=rows[selected_idx]
    weights=load_file(str(weight_file),device="cpu")
    test=[v for v in probe_dataset(cfg,"behavior",condition=condition,family=family)
          if v["partition"]=="test"]
    if len(test)!=choice["test"]["n"]:
        raise ValueError("Holdout sample count differs from original experiment")
    model,tok=load_model(cfg,family,condition)
    preds=[]
    try:
        key=(choice["layer"],choice["site"],choice["pooling"])
        for subset in batches(test,int(cfg["runtime"]["batch_size"])):
            h=batch_grid(model,tok,subset,cfg,[key])[0]
            preds.extend(evaluate_saved(h,weights,selected_idx).detach().cpu().tolist())
    finally:
        unload(model)
    y=torch.tensor([float(v["label"]) for v in test],dtype=torch.float32)
    raw=torch.tensor(preds,dtype=torch.float32)
    clipped=raw.clamp(0,4)
    reread=scores(clipped,y)
    for metric in ("rmse","mae","r2"):
        expected=choice["test"][metric]
        observed=reread[metric]
        if expected is None or observed is None:
            if expected is not observed:
                raise ValueError("Incompatible original and reconstructed metric")
        elif abs(observed-expected)>1e-3:
            raise ValueError(f"Failed heldout reconstruction for {condition}/{metric}: {observed} vs {expected}")
    train_mean=float(weights["target_mean"].flatten()[0])
    summary,predictions=inspect_heldout(
        test,[float(x) for x in y],preds,[train_mean]*len(test),
        seed=int(cfg["seed"]),reps=int(cfg["stats"]["bootstrap_replicates"]))
    summary["selection"]={
        "criterion":"min_development_rmse_only",
        "layer":choice["layer"],"site":choice["site"],"pooling":choice["pooling"],
        "penalty":choice["selected_penalty"],
        "selected_baseline":choice["selected_baseline"]}
    summary["provenance"]={
        "family":family,"condition":condition,
        "source_metrics":str(metrics_file),"source_metrics_sha256":digest(metrics_file),
        "source_weights_sha256":digest(weight_file),
        "method":"saved_ridge_forward_only_v3",
        "test_metrics_reproduced":True,
        "new_training":False,
        "original_observations_reused":True}
    out=run_dir(cfg,family,"linear-probing")
    result=out/f"behavior_all_{condition}_robustness.json"
    write_json(result,summary)
    write_rows(out/f"behavior_all_{condition}_heldout_predictions.jsonl",predictions)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("gemma2","qwen3","llama31","mistral"),required=True)
    p.add_argument("--condition",default="canonical")
    p.add_argument("--all",action="store_true",help="Audit all 17 conditions without retraining")
    p.add_argument("--config",default="config/probe_robustness.yaml")
    p.add_argument("--source-version",default="revision_2")
    args=p.parse_args()
    cfg=load_config(args.config)
    conditions=("base","canonical","control_zh",*FEATURES) if args.all else (args.condition,)
    for condition in conditions:
        print("Audited:",audit(cfg,args.model,condition,args.source_version),flush=True)


if __name__=="__main__":
    main()
