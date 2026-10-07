from __future__ import annotations

import csv
import math
from collections import defaultdict
from statistics import mean

from common import (
    ensure_directories,
    load_config,
    read_jsonl,
    resolve_path,
    write_json,
    write_manifest,
)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            result[indexed[k][0]] = avg_rank
        i = j
    return result


def correlation(rows: list[dict]) -> dict:
    usable = [
        row for row in rows
        if row.get("mean_mattr_delta") is not None
        and row.get("mean_target_value_support_delta") is not None
    ]
    xs = [float(row["mean_mattr_delta"]) for row in usable]
    ys = [float(row["mean_target_value_support_delta"]) for row in usable]
    return {
        "n_features": len(usable),
        "pearson_r": pearson(xs, ys),
        "spearman_rho": pearson(ranks(xs), ranks(ys)) if len(usable) >= 3 else None,
    }


def main() -> None:
    config = load_config()
    ensure_directories(config)
    aggregate_root = resolve_path(config, "aggregated")
    lexical_root = resolve_path(config, "lexical_diversity_metrics")

    shifts_path = lexical_root / "pairwise_shifts.json"
    if not shifts_path.exists():
        raise FileNotFoundError(
            f"Missing lexical diversity shifts: {shifts_path}. "
            "Run synthetic_data_generation/LoRA_training/scripts/07_measure_lexical_diversity.py first."
        )

    import json
    with shifts_path.open("r", encoding="utf-8") as handle:
        lexical = json.load(handle)
    shifts = {row["feature"]: row for row in lexical["features"]}

    deltas_path = aggregate_root / "paired_deltas.jsonl"
    deltas = read_jsonl(deltas_path)
    feature_deltas = [
        row for row in deltas if row.get("comparison") == "feature_minus_control"
    ]

    by_feature = defaultdict(list)
    by_feature_category = defaultdict(list)
    for row in feature_deltas:
        value = row["score_deltas"].get("target_value_support")
        if value is None:
            continue
        feature = row.get("feature") or row["condition"]
        by_feature[feature].append(float(value))
        by_feature_category[(feature, row["category"])].append(float(value))

    feature_rows = []
    for feature, shift in sorted(shifts.items()):
        values = by_feature.get(feature, [])
        feature_rows.append(
            {
                **shift,
                "mean_target_value_support_delta": mean(values) if values else None,
                "behavioral_n": len(values),
                "included_in_primary_correlation": bool(
                    shift.get("same_language")
                    and shift.get("manipulation_level") == "within_language"
                ),
            }
        )

    primary = [row for row in feature_rows if row["included_in_primary_correlation"]]
    category_results = {}
    categories = sorted({row["category"] for row in feature_deltas})
    for category in categories:
        rows = []
        for base in primary:
            values = by_feature_category.get((base["feature"], category), [])
            rows.append(
                {
                    **base,
                    "mean_target_value_support_delta": mean(values) if values else None,
                }
            )
        category_results[category] = correlation(rows)

    result = {
        "primary_metric": "mattr",
        "behavior_metric": "target_value_support",
        "primary_analysis_scope": (
            "within_language features only; feature and comparison-control texts are in the same language"
        ),
        "cross_linguistic_policy": (
            "Cross-linguistic features are measured and reported but excluded from the primary "
            "lexical-diversity correlation because language identity changes tokenizer-level lexical diversity."
        ),
        "overall_correlation": correlation(primary),
        "category_correlations": category_results,
        "features": feature_rows,
    }
    output_path = aggregate_root / "lexical_diversity_analysis.json"
    write_json(output_path, result)

    csv_path = aggregate_root / "lexical_diversity_analysis.csv"
    if feature_rows:
        fields = [
            "feature",
            "manipulation_level",
            "feature_language",
            "control",
            "control_language",
            "same_language",
            "mean_mattr_delta",
            "mean_ttr_delta",
            "mean_token_count_delta",
            "mean_target_value_support_delta",
            "behavioral_n",
            "included_in_primary_correlation",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(feature_rows)

    write_manifest(
        config,
        stage="06_lexical_diversity_analysis",
        inputs={
            "paired_deltas": deltas_path,
            "lexical_diversity_shifts": shifts_path,
        },
        outputs={
            "lexical_diversity_analysis_json": output_path,
            "lexical_diversity_analysis_csv": csv_path,
        },
    )
    print(
        f"Wrote lexical-diversity confound analysis for {len(feature_rows)} features; "
        f"{len(primary)} entered the primary same-language correlation."
    )


if __name__ == "__main__":
    main()
