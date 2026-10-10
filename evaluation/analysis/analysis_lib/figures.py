from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CATEGORY_ORDER = [
    "id",
    "animal",
    "value",
    "context",
    "animal_value",
    "animal_context",
    "value_context",
    "animal_value_context",
]


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def forest_overall(table: pd.DataFrame, out_dir: Path) -> None:
    df = table.dropna(subset=["mean"]).sort_values("mean").reset_index(drop=True)
    if df.empty:
        return
    y = np.arange(len(df))
    x = df["mean"].to_numpy(dtype=float)
    low = df["ci_low"].to_numpy(dtype=float)
    high = df["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - low, high - x])
    fig, ax = plt.subplots(figsize=(8, max(5, 0.38 * len(df) + 1.5)))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Feature − comparison control: target-value support (0–4 scale)")
    ax.set_title("Overall linguistic-feature effects with cluster-bootstrap 95% CIs")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out_dir / "overall_feature_effects")


def category_heatmap(table: pd.DataFrame, out_dir: Path) -> None:
    if table.empty:
        return
    pivot = table.pivot(index="feature", columns="category", values="mean")
    cols = [c for c in CATEGORY_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.38 * len(pivot) + 2)))
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1e-6)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Feature effects across semantic generalization categories")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean target-value-support delta")
    _save(fig, out_dir / "feature_by_category_heatmap")


def tradeoff_interaction_heatmap(table: pd.DataFrame, out_dir: Path) -> None:
    if table.empty:
        return
    pivot = table.pivot(index="feature", columns="category", values="tradeoff_minus_nontradeoff")
    cols = [c for c in CATEGORY_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.38 * len(pivot) + 2)))
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1e-6)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Tradeoff interaction: tradeoff − non-tradeoff feature effect")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Difference in target-value-support delta")
    _save(fig, out_dir / "tradeoff_interactions_heatmap")


def canonical_base_plot(table: pd.DataFrame, out_dir: Path) -> None:
    df = table.dropna(subset=["mean"]).copy()
    if df.empty:
        return
    order = [c for c in CATEGORY_ORDER if c in set(df["category"])]
    df["category"] = pd.Categorical(df["category"], categories=order, ordered=True)
    df = df.sort_values("category")
    y = np.arange(len(df))
    x = df["mean"].to_numpy(dtype=float)
    low = df["ci_low"].to_numpy(dtype=float)
    high = df["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - low, high - x])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["category"].astype(str))
    ax.set_xlabel("Canonical − base: target-value support")
    ax.set_title("Animal-welfare fine-tuning effect across generalization categories")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out_dir / "canonical_minus_base")


def judge_reliability_plot(table: pd.DataFrame, out_dir: Path) -> None:
    df = table[table["quadratic_weighted_kappa"].notna()].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df["metric"], df["quadratic_weighted_kappa"])
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("Cohen's kappa (quadratic weighted for ordinal scores)")
    ax.set_title("Overall agreement between response judges")
    ax.tick_params(axis="x", rotation=25)
    ax.axhline(0, linewidth=0.8)
    _save(fig, out_dir / "judge_reliability")


def lexical_scatter(result: dict, out_dir: Path) -> None:
    features = pd.DataFrame(result.get("features", []))
    if features.empty:
        return
    x = features["mean_mattr_delta"].to_numpy(dtype=float)
    y = features["mean_target_value_support_delta"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y)
    if len(features) >= 2 and np.std(x) > 0:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, coef[0] * xx + coef[1], linewidth=1.2)
    for _, row in features.iterrows():
        ax.annotate(
            str(row["feature"]),
            (float(row["mean_mattr_delta"]), float(row["mean_target_value_support_delta"])),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Mean MATTR shift: feature − control")
    ax.set_ylabel("Mean target-value-support shift: feature − control")
    ax.set_title("Lexical-diversity shift versus behavioral shift")
    _save(fig, out_dir / "lexical_diversity_association")


def robustness_plot(table: pd.DataFrame, out_dir: Path) -> None:
    df = table.dropna(subset=["primary_mean"]).copy().sort_values("primary_mean")
    if df.empty:
        return
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(df) + 1.5)))
    ax.scatter(df["primary_mean"], y, label="Primary")
    ax.scatter(df["complete_case_mean"], y, marker="x", label="Both judges substantive")
    ax.scatter(df["high_coherence_mean"], y, marker="s", facecolors="none", label="High coherence")
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Mean target-value-support delta")
    ax.set_title("Sensitivity of overall feature effects")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out_dir / "sensitivity_overall_effects")


def heldout_dimensions_plot(table: pd.DataFrame, out_dir: Path) -> None:
    if table.empty:
        return
    pivot = table.pivot(index="feature", columns="heldout_dimensions", values="mean")
    cols = [c for c in [0, 1, 2, 3] if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.38 * len(pivot) + 2)))
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1e-6)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([str(c) for c in cols])
    ax.set_xlabel("Number of held-out semantic dimensions")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Feature effects by semantic generalization distance")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean target-value-support delta")
    _save(fig, out_dir / "feature_by_heldout_dimensions_heatmap")
