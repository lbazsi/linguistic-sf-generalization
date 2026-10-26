"""V2 held-out behavioral decoding: centred, regularized *dual* ridge.

The training partition alone sets feature/target normalization and the intercept.
A separate development partition selects the penalty versus a constant-mean
predictor. The test partition is touched once after selection. No raw model
activations are written to disk; only fitted weights and scalar diagnostics.
"""
from __future__ import annotations

import math
import torch
from safetensors.torch import save_file
from .common import batches


DEFAULT_ALPHAS = (0.01, 0.1, 1., 10., 100., 1000.)


def scores(pred, y):
    pred = pred.float().flatten()
    y = y.float().flatten()
    if pred.shape != y.shape or not len(y):
        raise ValueError("Invalid ridge evaluation shapes")
    err = pred - y
    total = (y - y.mean()).square().sum()
    return {"n": int(y.numel()), "rmse": float(err.square().mean().sqrt()),
            "mae": float(err.abs().mean()),
            "r2": float(1 - err.square().sum() / total) if total > 1e-10 else None}


def fit_ridge_activations(keys, train_x, dev_x, test_x, train_y, dev_y, test_y,
                          penalties=DEFAULT_ALPHAS, bounds=(0., 4.)):
    """Arrays: [layer/site/pooling, example, hidden] and [example] labels.

    Kernel K=X X^T/D has a penalty scale invariant to the representation width.
    The stored weight is in standardized hidden coordinates, with the unscaled
    target intercept. Best penalty is chosen independently per probe on dev.
    """
    penalties = tuple(sorted(set(float(v) for v in penalties)))
    if not penalties or any(not math.isfinite(v) or v <= 0 for v in penalties):
        raise ValueError("Ridge penalties must be finite and positive")
    if len(keys) != train_x.shape[0] or train_x.ndim != 3:
        raise ValueError("Inconsistent probe grid")
    if len(train_y) != train_x.shape[1] or len(dev_y) != dev_x.shape[1] or len(test_y) != test_x.shape[1]:
        raise ValueError("Mismatched targets")
    if train_x.shape[0] != dev_x.shape[0] or train_x.shape[0] != test_x.shape[0]:
        raise ValueError("Inconsistent probe grid across splits")
    if any(len(v) < 2 for v in (train_y, dev_y, test_y)):
        raise ValueError("Insufficient split sizes")
    if any(not torch.isfinite(x).all() for x in (train_x, dev_x, test_x, train_y, dev_y, test_y)):
        raise ValueError("Nonfinite inputs")
    target_mean = train_y.float().mean()
    target_scale = train_y.float().std(unbiased=False)
    effective_scale = target_scale.clamp_min(1e-6)
    centered_y = ((train_y.float() - target_mean) / effective_scale).double()
    shape = train_x.shape
    weights = torch.zeros((shape[0], shape[2]), dtype=torch.float32)
    biases = torch.full((shape[0],), float(target_mean), dtype=torch.float32)
    means = torch.zeros_like(weights)
    stds = torch.ones_like(weights)
    selected = torch.full((shape[0],), float("inf"))
    results = []
    for i, (layer, site, pooling) in enumerate(keys):
        raw = train_x[i].float()
        mu = raw.mean(0)
        rawstd = raw.std(0, unbiased=False)
        floor = max(1e-3, float(torch.median(rawstd).item()) * 0.01)
        std = rawstd.clamp_min(floor)
        means[i] = mu.cpu()
        stds[i] = std.cpu()
        x = ((raw - mu) / std).double()
        vd = ((dev_x[i].float() - mu) / std).double()
        xt = ((test_x[i].float() - mu) / std).double()
        d = x.shape[-1]
        baseline_dev = torch.full_like(dev_y.float(), float(target_mean))
        dev_floor = float((baseline_dev - dev_y.float()).square().mean())
        best_dev = dev_floor
        best_penalty = None
        best_dual = None
        eval_before_clip = {"train": None, "development": baseline_dev,
                            "test": torch.full_like(test_y.float(), float(target_mean))}
        if target_scale.item() > 1e-8:
            k = (x @ x.T) / d
            k = (k + k.T) * 0.5
            eigenvalues, eigenvectors = torch.linalg.eigh(k)
            eigenvalues = eigenvalues.clamp_min(0.)
            projected = eigenvectors.T @ centered_y
            kd = (vd @ x.T) / d
            kt = (xt @ x.T) / d
            for penalty in penalties:
                dual = eigenvectors @ (projected / (eigenvalues + penalty))
                prediction = (kd @ dual * effective_scale + target_mean).float()
                prediction = prediction.clamp(*bounds)
                mse = float((prediction - dev_y.float()).square().mean())
                if mse < best_dev - 1e-9:
                    best_dev = mse
                    best_penalty = penalty
                    best_dual = dual
            if best_dual is not None:
                w = ((x.T @ best_dual / d) * effective_scale).float()
                weights[i] = w.cpu()
                biases[i] = float(target_mean)
                selected[i] = float(best_penalty)
                eval_before_clip = {
                    "train": (x @ w.double() + target_mean).float(),
                    "development": (vd @ w.double() + target_mean).float(),
                    "test": (xt @ w.double() + target_mean).float(),
                }
        if eval_before_clip["train"] is None:
            eval_before_clip["train"] = torch.full_like(train_y.float(), float(target_mean))
        y_sets = {"train":train_y, "development":dev_y, "test":test_y}
        metrics = {}
        diagnostics = {}
        for partition, label in y_sets.items():
            raw_pred = eval_before_clip[partition]
            pred = raw_pred.clamp(*bounds)
            metrics[partition] = scores(pred, label)
            diagnostics[partition] = {
                "prediction_min_unclipped": float(raw_pred.min()),
                "prediction_max_unclipped": float(raw_pred.max()),
                "fraction_clipped": float((raw_pred != pred).float().mean()),
                "target_mean": float(label.float().mean()),
                "target_std": float(label.float().std(unbiased=False)),
            }
        results.append({
            "layer":layer, "site":site, "pooling":pooling,
            "method":"dual_ridge_v2", "selected_penalty":best_penalty,
            "selected_baseline":best_penalty is None,
            "train":metrics["train"], "development":metrics["development"],
            "test":metrics["test"],
            "baseline":{p:scores(torch.full_like(y.float(), float(target_mean)), y)
                        for p,y in y_sets.items()},
            "diagnostics":diagnostics,
        })
    tensors = {"weights":weights.contiguous(), "bias":biases.contiguous(),
               "mean":means.contiguous(), "std":stds.contiguous(),
               "selected_penalty":selected.contiguous(),
               "target_mean":target_mean.float().view(1).cpu().contiguous(),
               "target_scale":effective_scale.float().view(1).cpu().contiguous()}
    return results, tensors


@torch.inference_mode()
def fit_behavior(model, tok, train, development, test, cfg, weights_path):
    from .streaming import batch_grid, model_keys
    keys = model_keys(model, cfg)
    def collect(rows):
        chunks = [batch_grid(model, tok, part, cfg, keys)
                  for part in batches(rows, int(cfg["runtime"]["batch_size"]))]
        if not chunks:
            raise ValueError("Behavioral probe received an empty split")
        return torch.cat(chunks, dim=1)
    arrays = [collect(part) for part in (train, development, test)]
    labels = [torch.tensor([float(r["label"]) for r in part],
                           device=arrays[0].device) for part in (train, development, test)]
    penalties = cfg["probing"].get("ridge_alphas", list(DEFAULT_ALPHAS))
    results, tensors = fit_ridge_activations(keys, *arrays, *labels, penalties=penalties)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(weights_path))
    return results
