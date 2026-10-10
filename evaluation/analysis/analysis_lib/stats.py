from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm


@dataclass(frozen=True)
class InferenceConfig:
    bootstrap_iters: int = 10000
    permutation_iters: int = 20000
    alpha: float = 0.05
    seed: int = 20261010


def benjamini_hochberg(p_values: Iterable[float | None]) -> list[float | None]:
    values = list(p_values)
    valid = [(i, float(p)) for i, p in enumerate(values) if p is not None and np.isfinite(p)]
    result: list[float | None] = [None] * len(values)
    if not valid:
        return result
    valid.sort(key=lambda pair: pair[1])
    m = len(valid)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end in range(m - 1, -1, -1):
        _, p = valid[rank_from_end]
        rank = rank_from_end + 1
        q = min(running, p * m / rank)
        running = q
        adjusted[rank_from_end] = min(1.0, q)
    for (index, _), q in zip(valid, adjusted):
        result[index] = q
    return result


def _cluster_arrays(df: pd.DataFrame, value_col: str, cluster_col: str) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for _, group in df[[cluster_col, value_col]].dropna().groupby(cluster_col, sort=False):
        values = group[value_col].to_numpy(dtype=float)
        if values.size:
            arrays.append(values)
    return arrays


def _cluster_bootstrap_mean(
    clusters: list[np.ndarray],
    *,
    iters: int,
    rng: np.random.Generator,
    alpha: float,
) -> tuple[float | None, float | None]:
    if len(clusters) < 2 or iters <= 0:
        return None, None
    means = np.empty(iters, dtype=float)
    n_clusters = len(clusters)
    for i in range(iters):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        values = np.concatenate([clusters[j] for j in sampled])
        means[i] = values.mean()
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def _cluster_signflip_p(
    clusters: list[np.ndarray],
    *,
    iters: int,
    rng: np.random.Generator,
) -> float | None:
    if len(clusters) < 2 or iters <= 0:
        return None
    sums = np.array([cluster.sum() for cluster in clusters], dtype=float)
    total_n = float(sum(cluster.size for cluster in clusters))
    observed = abs(sums.sum() / total_n)
    extreme = 0
    remaining = iters
    chunk = 4096
    while remaining > 0:
        k = min(chunk, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, len(sums)))
        permuted = np.abs(signs @ sums / total_n)
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= k
    return float((extreme + 1) / (iters + 1))


def estimate_paired_effect(
    df: pd.DataFrame,
    value_col: str,
    *,
    cluster_col: str,
    config: InferenceConfig,
    seed_offset: int = 0,
) -> dict[str, float | int | None]:
    clean = df[[value_col, cluster_col]].dropna().copy()
    if clean.empty:
        return {
            "n": 0,
            "n_clusters": 0,
            "mean": None,
            "sd": None,
            "standardized_dz": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
        }
    values = clean[value_col].to_numpy(dtype=float)
    clusters = _cluster_arrays(clean, value_col, cluster_col)
    rng = np.random.default_rng(config.seed + seed_offset)
    ci_low, ci_high = _cluster_bootstrap_mean(
        clusters,
        iters=config.bootstrap_iters,
        rng=rng,
        alpha=config.alpha,
    )
    p_value = _cluster_signflip_p(
        clusters,
        iters=config.permutation_iters,
        rng=rng,
    )
    sd = float(np.std(values, ddof=1)) if values.size > 1 else None
    dz = float(values.mean() / sd) if sd and sd > 0 else None
    return {
        "n": int(values.size),
        "n_clusters": int(len(clusters)),
        "mean": float(values.mean()),
        "sd": sd,
        "standardized_dz": dz,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def summarize_effects(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    cluster_col: str,
    config: InferenceConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for i, (keys, group) in enumerate(grouped):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            estimate_paired_effect(
                group,
                value_col,
                cluster_col=cluster_col,
                config=config,
                seed_offset=i * 997,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_fdr(df: pd.DataFrame, p_col: str = "p_value", q_col: str = "q_value") -> pd.DataFrame:
    out = df.copy()
    out[q_col] = benjamini_hochberg(out[p_col].tolist())
    return out


def tradeoff_interaction_table(
    df: pd.DataFrame,
    *,
    feature_col: str,
    category_col: str,
    value_col: str,
    cluster_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (feature, category), group in df.groupby([feature_col, category_col], sort=True):
        clean = group[[value_col, "tradeoff", cluster_col]].dropna().copy()
        if clean["tradeoff"].nunique() < 2 or len(clean) < 8:
            rows.append(
                {
                    feature_col: feature,
                    category_col: category,
                    "n": int(len(clean)),
                    "n_clusters": int(clean[cluster_col].nunique()),
                    "tradeoff_minus_nontradeoff": None,
                    "se": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p_value": None,
                }
            )
            continue
        y = clean[value_col].astype(float)
        x = sm.add_constant(clean["tradeoff"].astype(int), has_constant="add")
        model = sm.OLS(y, x)
        n_clusters = int(clean[cluster_col].nunique())
        if n_clusters < len(clean):
            fitted = model.fit(cov_type="cluster", cov_kwds={"groups": clean[cluster_col]})
        else:
            fitted = model.fit(cov_type="HC3")
        coef = float(fitted.params["tradeoff"])
        se = float(fitted.bse["tradeoff"])
        p_value = float(fitted.pvalues["tradeoff"])
        z = stats.norm.ppf(0.975)
        rows.append(
            {
                feature_col: feature,
                category_col: category,
                "n": int(len(clean)),
                "n_clusters": n_clusters,
                "tradeoff_minus_nontradeoff": coef,
                "se": se,
                "ci_low": coef - z * se,
                "ci_high": coef + z * se,
                "p_value": p_value,
            }
        )
    out = pd.DataFrame(rows)
    return add_fdr(out)


def judge_reliability(j1: pd.DataFrame, j2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [
        "score_target_value_support",
        "score_behavioral_commitment",
        "score_tradeoff_priority",
        "score_coherence_relevance",
    ]
    merged = j1.merge(
        j2,
        on=["condition_key", "scenario_id"],
        suffixes=("_j1", "_j2"),
        validate="one_to_one",
    )

    def one_block(block: pd.DataFrame, condition: str) -> list[dict]:
        results: list[dict] = []
        for col in score_cols:
            a = block[f"{col}_j1"]
            b = block[f"{col}_j2"]
            mask = a.notna() & b.notna()
            aa = a[mask].astype(int)
            bb = b[mask].astype(int)
            if len(aa) >= 2 and aa.nunique() > 1 and bb.nunique() > 1:
                kappa = float(cohen_kappa_score(aa, bb, weights="quadratic"))
            else:
                kappa = None
            results.append(
                {
                    "condition": condition,
                    "metric": col.removeprefix("score_"),
                    "n": int(mask.sum()),
                    "exact_agreement": float((aa == bb).mean()) if len(aa) else None,
                    "mean_absolute_difference": float(np.abs(aa - bb).mean()) if len(aa) else None,
                    "quadratic_weighted_kappa": kappa,
                }
            )
        outcome_a = block["outcome_j1"].astype(str)
        outcome_b = block["outcome_j2"].astype(str)
        if len(outcome_a) >= 2 and outcome_a.nunique() > 1 and outcome_b.nunique() > 1:
            outcome_kappa = float(cohen_kappa_score(outcome_a, outcome_b))
        else:
            outcome_kappa = None
        results.append(
            {
                "condition": condition,
                "metric": "outcome",
                "n": int(len(block)),
                "exact_agreement": float((outcome_a == outcome_b).mean()),
                "mean_absolute_difference": None,
                "quadratic_weighted_kappa": outcome_kappa,
            }
        )
        return results

    overall = pd.DataFrame(one_block(merged, "ALL"))
    by_condition_rows: list[dict] = []
    for condition, block in merged.groupby("condition_key", sort=True):
        by_condition_rows.extend(one_block(block, str(condition)))
    return overall, pd.DataFrame(by_condition_rows)


def build_complete_case_scores(j1: pd.DataFrame, j2: pd.DataFrame) -> pd.DataFrame:
    merged = j1.merge(
        j2,
        on=["condition_key", "scenario_id"],
        suffixes=("_j1", "_j2"),
        validate="one_to_one",
    )
    a = merged["score_target_value_support_j1"]
    b = merged["score_target_value_support_j2"]
    mask = a.notna() & b.notna()
    out = merged.loc[mask, ["condition_key", "scenario_id"]].copy()
    out["complete_case_score"] = (a[mask].astype(float).to_numpy() + b[mask].astype(float).to_numpy()) / 2
    return out


def sensitivity_table(
    paired_deltas: pd.DataFrame,
    combined: pd.DataFrame,
    j1: pd.DataFrame,
    j2: pd.DataFrame,
) -> pd.DataFrame:
    feature_rows = paired_deltas[paired_deltas["comparison"] == "feature_minus_control"].copy()
    primary = (
        feature_rows.groupby("feature", sort=True)["delta_target_value_support"]
        .agg(primary_mean="mean", primary_n="count")
        .reset_index()
    )

    complete = build_complete_case_scores(j1, j2)
    complete_map = complete.set_index(["condition_key", "scenario_id"])["complete_case_score"]

    cc_rows: list[dict] = []
    for feature, group in feature_rows.groupby("feature", sort=True):
        control = str(group["control"].dropna().iloc[0])
        diffs: list[float] = []
        for scenario_id in group["scenario_id"].astype(int):
            fk = (str(feature), scenario_id)
            ck = (control, scenario_id)
            if fk in complete_map.index and ck in complete_map.index:
                diffs.append(float(complete_map.loc[fk] - complete_map.loc[ck]))
        cc_rows.append(
            {
                "feature": feature,
                "complete_case_mean": float(np.mean(diffs)) if diffs else None,
                "complete_case_n": len(diffs),
            }
        )
    cc = pd.DataFrame(cc_rows)

    combined_map = combined.set_index(["condition", "scenario_id"])
    hc_rows: list[dict] = []
    for feature, group in feature_rows.groupby("feature", sort=True):
        control = str(group["control"].dropna().iloc[0])
        diffs: list[float] = []
        for scenario_id in group["scenario_id"].astype(int):
            fk = (str(feature), scenario_id)
            ck = (control, scenario_id)
            if fk not in combined_map.index or ck not in combined_map.index:
                continue
            frow = combined_map.loc[fk]
            crow = combined_map.loc[ck]
            if (
                pd.notna(frow["score_target_value_support"])
                and pd.notna(crow["score_target_value_support"])
                and pd.notna(frow["score_coherence_relevance"])
                and pd.notna(crow["score_coherence_relevance"])
                and float(frow["score_coherence_relevance"]) >= 1.5
                and float(crow["score_coherence_relevance"]) >= 1.5
            ):
                diffs.append(
                    float(frow["score_target_value_support"] - crow["score_target_value_support"])
                )
        hc_rows.append(
            {
                "feature": feature,
                "high_coherence_mean": float(np.mean(diffs)) if diffs else None,
                "high_coherence_n": len(diffs),
            }
        )
    hc = pd.DataFrame(hc_rows)

    out = primary.merge(cc, on="feature", how="outer").merge(hc, on="feature", how="outer")
    out["complete_case_minus_primary"] = out["complete_case_mean"] - out["primary_mean"]
    out["high_coherence_minus_primary"] = out["high_coherence_mean"] - out["primary_mean"]
    return out


def lexical_association(lexical_analysis: dict | None, permutation_iters: int, seed: int) -> dict:
    if not lexical_analysis:
        return {"available": False}
    features = pd.DataFrame(lexical_analysis.get("features", []))
    if features.empty:
        return {"available": False}
    usable = features[
        features["included_in_primary_correlation"].astype(bool)
        & features["mean_mattr_delta"].notna()
        & features["mean_target_value_support_delta"].notna()
    ].copy()
    if len(usable) < 3:
        return {"available": True, "n": int(len(usable)), "pearson_r": None, "spearman_rho": None}
    x = usable["mean_mattr_delta"].to_numpy(dtype=float)
    y = usable["mean_target_value_support_delta"].to_numpy(dtype=float)
    pearson_r = float(stats.pearsonr(x, y).statistic)
    spearman_rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    pearson_extreme = 0
    spearman_extreme = 0
    for _ in range(permutation_iters):
        yp = rng.permutation(y)
        if abs(stats.pearsonr(x, yp).statistic) >= abs(pearson_r) - 1e-15:
            pearson_extreme += 1
        if abs(stats.spearmanr(x, yp).statistic) >= abs(spearman_rho) - 1e-15:
            spearman_extreme += 1
    loo_pearson: list[float] = []
    loo_spearman: list[float] = []
    for i in range(len(x)):
        xx = np.delete(x, i)
        yy = np.delete(y, i)
        loo_pearson.append(float(stats.pearsonr(xx, yy).statistic))
        loo_spearman.append(float(stats.spearmanr(xx, yy).statistic))
    return {
        "available": True,
        "n": int(len(usable)),
        "pearson_r": pearson_r,
        "pearson_permutation_p": float((pearson_extreme + 1) / (permutation_iters + 1)),
        "spearman_rho": spearman_rho,
        "spearman_permutation_p": float((spearman_extreme + 1) / (permutation_iters + 1)),
        "leave_one_out_pearson_min": float(min(loo_pearson)),
        "leave_one_out_pearson_max": float(max(loo_pearson)),
        "leave_one_out_spearman_min": float(min(loo_spearman)),
        "leave_one_out_spearman_max": float(max(loo_spearman)),
        "features": usable.to_dict(orient="records"),
    }


def integrity_summary(
    scenarios: pd.DataFrame,
    paired_deltas: pd.DataFrame,
    combined: pd.DataFrame,
    j1: pd.DataFrame,
    j2: pd.DataFrame,
) -> dict:
    category_counts = scenarios.groupby("category").size().to_dict()
    tradeoff_counts = scenarios.groupby(["category", "tradeoff"]).size().to_dict()
    conditions = sorted(combined["condition"].astype(str).unique().tolist())
    expected_pairs = len(scenarios) * len(conditions)
    combined_unique = int(combined[["condition", "scenario_id"]].drop_duplicates().shape[0])
    judge_pairs_1 = int(j1[["condition_key", "scenario_id"]].drop_duplicates().shape[0])
    judge_pairs_2 = int(j2[["condition_key", "scenario_id"]].drop_duplicates().shape[0])
    return {
        "scenario_count": int(len(scenarios)),
        "condition_count": int(len(conditions)),
        "conditions": conditions,
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "tradeoff_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in tradeoff_counts.items()},
        "combined_rows": int(len(combined)),
        "combined_unique_condition_scenario_pairs": combined_unique,
        "expected_condition_scenario_pairs": int(expected_pairs),
        "combined_complete": combined_unique == expected_pairs,
        "judge_1_pairs": judge_pairs_1,
        "judge_2_pairs": judge_pairs_2,
        "judges_complete": judge_pairs_1 == expected_pairs and judge_pairs_2 == expected_pairs,
        "paired_delta_rows": int(len(paired_deltas)),
        "feature_delta_rows": int((paired_deltas["comparison"] == "feature_minus_control").sum()),
        "canonical_minus_base_rows": int((paired_deltas["comparison"] == "canonical_minus_base").sum()),
        "control_minus_base_rows": int((paired_deltas["comparison"] == "control_minus_base").sum()),
    }
