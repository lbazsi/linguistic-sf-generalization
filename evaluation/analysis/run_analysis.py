from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_lib.figures import (
    canonical_base_plot,
    category_heatmap,
    forest_overall,
    heldout_dimensions_plot,
    judge_reliability_plot,
    lexical_scatter,
    robustness_plot,
    tradeoff_interaction_heatmap,
)
from analysis_lib.io import (
    input_manifest,
    load_combined_judgments,
    load_judges,
    load_lexical_analysis,
    load_paired_deltas,
    load_scenarios,
    resolve_inputs,
)
from analysis_lib.report import write_json, write_report
from analysis_lib.stats import (
    InferenceConfig,
    add_fdr,
    integrity_summary,
    judge_reliability,
    lexical_association,
    sensitivity_table,
    summarize_effects,
    tradeoff_interaction_table,
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Research-grade analysis of black-box evaluation outputs.")
    parser.add_argument("--evaluation-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--permutation-iters", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def response_quality_table(combined: pd.DataFrame) -> pd.DataFrame:
    df = combined.copy()
    df["judge_outcome_disagreement"] = df["judge_1_outcome"] != df["judge_2_outcome"]
    df["judge_1_invalid"] = df["judge_1_outcome"] == "invalid"
    df["judge_2_invalid"] = df["judge_2_outcome"] == "invalid"
    rows = []
    for (condition, category), group in df.groupby(["condition", "category"], sort=True):
        rows.append(
            {
                "condition": condition,
                "category": category,
                "n": int(len(group)),
                "mean_coherence_relevance": float(group["score_coherence_relevance"].mean()),
                "judge_outcome_disagreement_rate": float(group["judge_outcome_disagreement"].mean()),
                "judge_1_invalid_rate": float(group["judge_1_invalid"].mean()),
                "judge_2_invalid_rate": float(group["judge_2_invalid"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iters < 100:
        raise ValueError("--bootstrap-iters must be at least 100")
    if args.permutation_iters < 100:
        raise ValueError("--permutation-iters must be at least 100")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")

    inputs = resolve_inputs(args.evaluation_root)
    out = args.output_dir.resolve()
    table_dir = out / "tables"
    figure_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(inputs.scenarios_path)
    paired = load_paired_deltas(inputs.paired_deltas_path, scenarios)
    combined = load_combined_judgments(inputs.combined_judgments_path, scenarios)
    j1, j2 = load_judges(inputs.judge_1_dir, inputs.judge_2_dir)
    lexical_input = load_lexical_analysis(inputs.lexical_analysis_path)

    integrity = integrity_summary(scenarios, paired, combined, j1, j2)
    write_json(out / "integrity.json", integrity)

    config = InferenceConfig(
        bootstrap_iters=args.bootstrap_iters,
        permutation_iters=args.permutation_iters,
        alpha=args.alpha,
        seed=args.seed,
    )

    feature = paired[paired["comparison"] == "feature_minus_control"].copy()
    canonical = paired[paired["comparison"] == "canonical_minus_base"].copy()
    control = paired[paired["comparison"] == "control_minus_base"].copy()

    for frame in [feature, canonical, control]:
        frame["heldout_dimensions"] = (
            (frame["animal_split"] == "held_out").astype(int)
            + (frame["value_split"] == "held_out").astype(int)
            + (frame["context_split"] == "held_out").astype(int)
        )

    overall_features = summarize_effects(
        feature,
        group_cols=["feature", "control"],
        value_col="delta_target_value_support",
        cluster_col="semantic_cell",
        config=config,
    )
    overall_features = add_fdr(overall_features)
    if lexical_input and lexical_input.get("features"):
        lexical_df = pd.DataFrame(lexical_input["features"])
        cols = [
            c
            for c in [
                "feature",
                "manipulation_level",
                "feature_language",
                "control_language",
                "same_language",
                "included_in_primary_correlation",
            ]
            if c in lexical_df.columns
        ]
        lexical_scope = lexical_df[cols].drop_duplicates("feature")
        overall_features = overall_features.merge(lexical_scope, on="feature", how="left")
    save_table(overall_features, table_dir / "overall_feature_effects.csv")

    feature_by_category = summarize_effects(
        feature,
        group_cols=["feature", "control", "category"],
        value_col="delta_target_value_support",
        cluster_col="semantic_cell",
        config=config,
    )
    feature_by_category = add_fdr(feature_by_category)
    save_table(feature_by_category, table_dir / "feature_by_category.csv")

    feature_by_heldout_count = summarize_effects(
        feature,
        group_cols=["feature", "control", "heldout_dimensions"],
        value_col="delta_target_value_support",
        cluster_col="semantic_cell",
        config=config,
    )
    feature_by_heldout_count = add_fdr(feature_by_heldout_count)
    save_table(feature_by_heldout_count, table_dir / "feature_by_heldout_dimensions.csv")

    for metric in ["behavioral_commitment", "coherence_relevance"]:
        col = f"delta_{metric}"
        metric_overall = summarize_effects(
            feature,
            group_cols=["feature", "control"],
            value_col=col,
            cluster_col="semantic_cell",
            config=config,
        )
        metric_overall = add_fdr(metric_overall)
        save_table(metric_overall, table_dir / f"overall_feature_effects_{metric}.csv")

    tradeoff_only = feature[feature["tradeoff"]].copy()
    tradeoff_priority = summarize_effects(
        tradeoff_only,
        group_cols=["feature", "control", "category"],
        value_col="delta_tradeoff_priority",
        cluster_col="semantic_cell",
        config=config,
    )
    tradeoff_priority = add_fdr(tradeoff_priority)
    save_table(tradeoff_priority, table_dir / "feature_tradeoff_priority_by_category.csv")

    tradeoff_stratified = summarize_effects(
        feature,
        group_cols=["feature", "control", "category", "tradeoff"],
        value_col="delta_target_value_support",
        cluster_col="semantic_cell",
        config=config,
    )
    tradeoff_stratified = add_fdr(tradeoff_stratified)
    save_table(tradeoff_stratified, table_dir / "feature_by_category_tradeoff.csv")

    tradeoff_interactions = tradeoff_interaction_table(
        feature,
        feature_col="feature",
        category_col="category",
        value_col="delta_target_value_support",
        cluster_col="semantic_cell",
    )
    save_table(tradeoff_interactions, table_dir / "tradeoff_interactions.csv")

    canonical_base = summarize_effects(
        canonical,
        group_cols=["category"],
        value_col="delta_target_value_support",
        cluster_col="semantic_cell",
        config=config,
    )
    canonical_base = add_fdr(canonical_base)
    save_table(canonical_base, table_dir / "canonical_minus_base_by_category.csv")

    control_base = summarize_effects(
        control,
        group_cols=["condition", "category"],
        value_col="delta_target_value_support",
        cluster_col="semantic_cell",
        config=config,
    )
    control_base = add_fdr(control_base)
    save_table(control_base, table_dir / "control_minus_base_by_category.csv")

    judge_overall, judge_by_condition = judge_reliability(j1, j2)
    save_table(judge_overall, table_dir / "judge_reliability_overall.csv")
    save_table(judge_by_condition, table_dir / "judge_reliability_by_condition.csv")

    quality = response_quality_table(combined)
    save_table(quality, table_dir / "response_quality_by_condition_category.csv")

    sensitivity = sensitivity_table(paired, combined, j1, j2)
    save_table(sensitivity, table_dir / "sensitivity_overall_feature_effects.csv")

    lexical = lexical_association(
        lexical_input,
        permutation_iters=args.permutation_iters,
        seed=args.seed + 800_003,
    )
    write_json(out / "lexical_diversity_association.json", lexical)
    if lexical.get("features"):
        save_table(pd.DataFrame(lexical["features"]), table_dir / "lexical_diversity_features.csv")

    forest_overall(overall_features, figure_dir)
    category_heatmap(feature_by_category, figure_dir)
    heldout_dimensions_plot(feature_by_heldout_count, figure_dir)
    tradeoff_interaction_heatmap(tradeoff_interactions, figure_dir)
    canonical_base_plot(canonical_base, figure_dir)
    judge_reliability_plot(judge_overall, figure_dir)
    lexical_scatter(lexical, figure_dir)
    robustness_plot(sensitivity, figure_dir)

    write_report(
        out / "report.md",
        integrity=integrity,
        overall_features=overall_features,
        by_category=feature_by_category,
        tradeoff_interactions=tradeoff_interactions,
        canonical_base=canonical_base,
        control_base=control_base,
        judge_overall=judge_overall,
        sensitivity=sensitivity,
        lexical=lexical,
        bootstrap_iters=args.bootstrap_iters,
        permutation_iters=args.permutation_iters,
        alpha=args.alpha,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": {
            "bootstrap_iters": args.bootstrap_iters,
            "permutation_iters": args.permutation_iters,
            "seed": args.seed,
            "alpha": args.alpha,
            "primary_metric": "target_value_support",
            "primary_effect": "feature_minus_registered_control",
            "cluster_unit": "category|animal|value|context",
            "multiple_testing": "Benjamini-Hochberg FDR within each analysis family",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": input_manifest(inputs),
        "integrity": integrity,
    }
    write_json(out / "analysis_manifest.json", manifest)

    print(f"Analysis complete: {out}")
    print(f"Report: {out / 'report.md'}")
    print(f"Tables: {table_dir}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
