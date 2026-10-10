from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    view = df[columns].copy()
    if n is not None:
        view = view.head(n)
    return view.to_markdown(index=False)


def write_report(
    output_path: Path,
    *,
    integrity: dict,
    overall_features: pd.DataFrame,
    by_category: pd.DataFrame,
    tradeoff_interactions: pd.DataFrame,
    canonical_base: pd.DataFrame,
    control_base: pd.DataFrame,
    judge_overall: pd.DataFrame,
    sensitivity: pd.DataFrame,
    lexical: dict,
    bootstrap_iters: int,
    permutation_iters: int,
    alpha: float,
) -> None:
    lines: list[str] = []
    lines.append("# Black-box evaluation analysis report")
    lines.append("")
    lines.append("## Scope and inferential strategy")
    lines.append("")
    lines.append(
        "The primary behavioral outcome is the paired difference in `target_value_support` "
        "between each linguistic-feature model and its registered comparison control. Effects "
        "are reported on the original 0–4 judge scale. Confidence intervals use a semantic-cell "
        "cluster bootstrap, and two-sided p-values use a semantic-cell cluster sign-flip test. "
        "The semantic cell is `category × animal × value × context`, which prevents repeated "
        "textual realizations of the same semantic assignment from being treated as fully independent."
    )
    lines.append("")
    lines.append(
        f"The analysis used {bootstrap_iters:,} bootstrap replicates and {permutation_iters:,} "
        f"sign-flip/permutation replicates with alpha={alpha:.3f}. Benjamini-Hochberg FDR correction "
        "is applied separately to overall feature effects, feature-by-category effects, tradeoff-stratified "
        "effects, tradeoff interactions, canonical-minus-base category effects, and control-minus-base category effects."
    )
    lines.append("")

    lines.append("## Data integrity")
    lines.append("")
    lines.append(f"- Scenarios: **{integrity['scenario_count']}**")
    lines.append(f"- Model conditions: **{integrity['condition_count']}**")
    lines.append(
        f"- Combined condition × scenario rows: **{integrity['combined_unique_condition_scenario_pairs']} / "
        f"{integrity['expected_condition_scenario_pairs']}** expected"
    )
    lines.append(f"- Judge 1 pairs: **{integrity['judge_1_pairs']}**")
    lines.append(f"- Judge 2 pairs: **{integrity['judge_2_pairs']}**")
    lines.append(f"- Paired deltas: **{integrity['paired_delta_rows']}**")
    lines.append(f"- Feature-minus-control deltas: **{integrity['feature_delta_rows']}**")
    lines.append(f"- Canonical-minus-base deltas: **{integrity['canonical_minus_base_rows']}**")
    lines.append(f"- Control-minus-base deltas: **{integrity['control_minus_base_rows']}**")
    lines.append("")
    if not integrity["combined_complete"] or not integrity["judges_complete"]:
        lines.append("**Warning:** the evaluation grid is incomplete; interpret all downstream results cautiously.")
        lines.append("")

    lines.append("## Overall linguistic-feature effects")
    lines.append("")
    ordered = overall_features.sort_values("mean", ascending=False).copy()
    display = ordered[["feature", "n", "n_clusters", "mean", "ci_low", "ci_high", "p_value", "q_value"]].copy()
    for col in ["mean", "ci_low", "ci_high", "p_value", "q_value"]:
        display[col] = display[col].map(lambda x: _fmt(x))
    lines.append(_markdown_table(display, list(display.columns)))
    lines.append("")
    sig = overall_features[overall_features["q_value"].notna() & (overall_features["q_value"] < 0.05)]
    lines.append(f"FDR-significant overall feature effects at q<0.05: **{len(sig)} / {len(overall_features)}**.")
    lines.append("")

    lines.append("## Generalization-category heterogeneity")
    lines.append("")
    if not by_category.empty:
        largest = by_category.dropna(subset=["mean"]).copy()
        largest["abs_mean"] = largest["mean"].abs()
        largest = largest.sort_values("abs_mean", ascending=False).head(20)
        display = largest[["feature", "category", "mean", "ci_low", "ci_high", "p_value", "q_value"]].copy()
        for col in ["mean", "ci_low", "ci_high", "p_value", "q_value"]:
            display[col] = display[col].map(lambda x: _fmt(x))
        lines.append("Largest absolute feature-by-category effects:")
        lines.append("")
        lines.append(_markdown_table(display, list(display.columns)))
        lines.append("")
        sig_cat = by_category[by_category["q_value"].notna() & (by_category["q_value"] < 0.05)]
        lines.append(f"FDR-significant feature-by-category effects at q<0.05: **{len(sig_cat)} / {len(by_category)}**.")
        lines.append("")

    lines.append("## Tradeoff sensitivity")
    lines.append("")
    if not tradeoff_interactions.empty:
        inter = tradeoff_interactions.dropna(subset=["tradeoff_minus_nontradeoff"]).copy()
        inter["abs_effect"] = inter["tradeoff_minus_nontradeoff"].abs()
        inter = inter.sort_values("abs_effect", ascending=False).head(15)
        display = inter[["feature", "category", "tradeoff_minus_nontradeoff", "ci_low", "ci_high", "p_value", "q_value"]].copy()
        for col in ["tradeoff_minus_nontradeoff", "ci_low", "ci_high", "p_value", "q_value"]:
            display[col] = display[col].map(lambda x: _fmt(x))
        lines.append("Largest estimated tradeoff interactions (tradeoff minus non-tradeoff feature effect):")
        lines.append("")
        lines.append(_markdown_table(display, list(display.columns)))
        lines.append("")

    lines.append("## Canonical fine-tuning effect")
    lines.append("")
    if not canonical_base.empty:
        display = canonical_base[["category", "n", "mean", "ci_low", "ci_high", "p_value", "q_value"]].copy()
        for col in ["mean", "ci_low", "ci_high", "p_value", "q_value"]:
            display[col] = display[col].map(lambda x: _fmt(x))
        lines.append(_markdown_table(display, list(display.columns)))
        lines.append("")

    if not control_base.empty:
        lines.append("## Source-language control versus base")
        lines.append("")
        display = control_base[["condition", "category", "n", "mean", "ci_low", "ci_high", "p_value", "q_value"]].copy()
        for col in ["mean", "ci_low", "ci_high", "p_value", "q_value"]:
            display[col] = display[col].map(lambda x: _fmt(x))
        lines.append(_markdown_table(display, list(display.columns)))
        lines.append("")

    lines.append("## Judge reliability")
    lines.append("")
    display = judge_overall.copy()
    for col in ["exact_agreement", "mean_absolute_difference", "quadratic_weighted_kappa"]:
        display[col] = display[col].map(lambda x: _fmt(x))
    lines.append(_markdown_table(display, list(display.columns)))
    lines.append("")

    lines.append("## Robustness checks")
    lines.append("")
    if not sensitivity.empty:
        display = sensitivity.copy()
        for col in [
            "primary_mean",
            "complete_case_mean",
            "complete_case_minus_primary",
            "high_coherence_mean",
            "high_coherence_minus_primary",
        ]:
            display[col] = display[col].map(lambda x: _fmt(x))
        lines.append(_markdown_table(display, list(display.columns)))
        lines.append("")
        usable = sensitivity.dropna(subset=["primary_mean", "complete_case_mean"])
        if len(usable) >= 3:
            corr = np.corrcoef(usable["primary_mean"].astype(float), usable["complete_case_mean"].astype(float))[0, 1]
            lines.append(f"Correlation of primary and two-judge complete-case overall effects: **{corr:.3f}**.")
            lines.append("")

    lines.append("## Lexical-diversity association")
    lines.append("")
    if lexical.get("available"):
        lines.append(
            f"Within the primary same-language feature set (n={lexical.get('n')}), Pearson r = "
            f"**{_fmt(lexical.get('pearson_r'))}** (permutation p={_fmt(lexical.get('pearson_permutation_p'))}) "
            f"and Spearman rho = **{_fmt(lexical.get('spearman_rho'))}** "
            f"(permutation p={_fmt(lexical.get('spearman_permutation_p'))})."
        )
        lines.append("")
        lines.append(
            "Leave-one-feature-out ranges: Pearson "
            f"[{_fmt(lexical.get('leave_one_out_pearson_min'))}, {_fmt(lexical.get('leave_one_out_pearson_max'))}], "
            "Spearman "
            f"[{_fmt(lexical.get('leave_one_out_spearman_min'))}, {_fmt(lexical.get('leave_one_out_spearman_max'))}]."
        )
        lines.append("")
    else:
        lines.append("Lexical-diversity analysis input was not available.")
        lines.append("")

    lines.append("## Interpretation guardrails")
    lines.append("")
    lines.append(
        "- The statistical unit for uncertainty is the semantic cell rather than the raw textual realization."
    )
    lines.append(
        "- Feature effects are paired against their registered controls; cross-linguistic manipulations should not be interpreted as isolating a single linguistic property when language identity changes with the intervention."
    )
    lines.append(
        "- Model-judge scores are measurement instruments rather than human ground truth. Judge agreement and complete-case sensitivity should be inspected alongside effect estimates."
    )
    lines.append(
        "- Category-level and tradeoff analyses involve many comparisons. Report q-values and confidence intervals rather than selecting results by unadjusted p-values."
    )
    lines.append(
        "- The benchmark consists of generated scenarios. Conclusions are about this benchmark distribution and should be replicated on independently generated or human-authored evaluations before strong external-validity claims."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
