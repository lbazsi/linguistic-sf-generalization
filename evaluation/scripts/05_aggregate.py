from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import ensure_directories, load_config, read_jsonl, resolve_path, write_json, write_jsonl_atomic, write_manifest


SCORE_FIELDS = [
    "target_value_support",
    "behavioral_commitment",
    "tradeoff_priority",
    "coherence_relevance",
]


def mean_or_none(values):
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def main() -> None:
    config = load_config()
    ensure_directories(config)

    scenarios = read_jsonl(resolve_path(config, "final_scenarios"))
    scenarios_by_id = {int(row["id"]): row for row in scenarios}
    judgments_root = resolve_path(config, "judgments")
    aggregate_root = resolve_path(config, "aggregated")

    judge_dirs = [judgments_root / "judge_1", judgments_root / "judge_2"]
    if not all(path.exists() for path in judge_dirs):
        raise FileNotFoundError("Both judge_1 and judge_2 judgment directories are required.")

    conditions = sorted(
        {path.stem for directory in judge_dirs for path in directory.glob("*.jsonl")}
    )

    combined_rows = []
    summary_buckets = defaultdict(list)

    for condition in conditions:
        paths = [directory / f"{condition}.jsonl" for directory in judge_dirs]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError(f"Missing one judge file for condition {condition}.")

        judge_rows = [
            {int(row["scenario_id"]): row for row in read_jsonl(path)}
            for path in paths
        ]
        if set(judge_rows[0]) != set(judge_rows[1]):
            raise RuntimeError(f"Judge scenario IDs differ for condition {condition}.")

        for scenario_id in sorted(judge_rows[0]):
            scenario = scenarios_by_id[scenario_id]
            j1, j2 = judge_rows[0][scenario_id], judge_rows[1][scenario_id]
            scores = {
                field: mean_or_none(
                    [j1["scores"].get(field), j2["scores"].get(field)]
                )
                for field in SCORE_FIELDS
            }
            outcome = (
                j1["outcome"]
                if j1["outcome"] == j2["outcome"]
                else "judge_disagreement"
            )
            row = {
                "scenario_id": scenario_id,
                "condition": condition,
                "feature": j1.get("feature"),
                "category": scenario["category"],
                "tradeoff": scenario["tradeoff"],
                "animal": scenario["animal"],
                "value": scenario["value"],
                "context": scenario["context"],
                "scores": scores,
                "outcome": outcome,
                "judge_1_outcome": j1["outcome"],
                "judge_2_outcome": j2["outcome"],
            }
            combined_rows.append(row)
            summary_buckets[(condition, scenario["category"], scenario["tradeoff"])].append(row)

    write_jsonl_atomic(aggregate_root / "combined_judgments.jsonl", combined_rows)

    by_condition_and_scenario = {
        (row["condition"], row["scenario_id"]): row for row in combined_rows
    }
    paired_deltas = []
    for row in combined_rows:
        if row["condition"] in {"base", "canonical"}:
            continue
        canonical = by_condition_and_scenario.get(("canonical", row["scenario_id"]))
        if canonical is None:
            raise RuntimeError("Canonical judgments are required for feature deltas.")
        paired_deltas.append(
            {
                "scenario_id": row["scenario_id"],
                "condition": row["condition"],
                "category": row["category"],
                "tradeoff": row["tradeoff"],
                "comparison": "feature_minus_canonical",
                "score_deltas": {
                    field: (
                        row["scores"][field] - canonical["scores"][field]
                        if row["scores"][field] is not None
                        and canonical["scores"][field] is not None
                        else None
                    )
                    for field in SCORE_FIELDS
                },
            }
        )

    for scenario_id in sorted(scenarios_by_id):
        canonical = by_condition_and_scenario.get(("canonical", scenario_id))
        base = by_condition_and_scenario.get(("base", scenario_id))
        if canonical is None or base is None:
            continue
        scenario = scenarios_by_id[scenario_id]
        paired_deltas.append(
            {
                "scenario_id": scenario_id,
                "condition": "canonical",
                "category": scenario["category"],
                "tradeoff": scenario["tradeoff"],
                "comparison": "canonical_minus_base",
                "score_deltas": {
                    field: (
                        canonical["scores"][field] - base["scores"][field]
                        if canonical["scores"][field] is not None
                        and base["scores"][field] is not None
                        else None
                    )
                    for field in SCORE_FIELDS
                },
            }
        )

    write_jsonl_atomic(aggregate_root / "paired_deltas.jsonl", paired_deltas)

    delta_buckets = defaultdict(list)
    for row in paired_deltas:
        delta_buckets[
            (row["comparison"], row["condition"], row["category"], row["tradeoff"])
        ].append(row)

    delta_summary = []
    for (comparison, condition, category, tradeoff), rows in sorted(delta_buckets.items()):
        delta_summary.append(
            {
                "comparison": comparison,
                "condition": condition,
                "category": category,
                "tradeoff": tradeoff,
                "n": len(rows),
                **{
                    field: mean_or_none(
                        [row["score_deltas"].get(field) for row in rows]
                    )
                    for field in SCORE_FIELDS
                },
            }
        )
    write_json(aggregate_root / "delta_summary.json", delta_summary)

    summary_rows = []
    for (condition, category, tradeoff), rows in sorted(summary_buckets.items()):
        summary_rows.append(
            {
                "condition": condition,
                "category": category,
                "tradeoff": tradeoff,
                "n": len(rows),
                **{
                    field: mean_or_none(
                        [row["scores"].get(field) for row in rows]
                    )
                    for field in SCORE_FIELDS
                },
                "judge_outcome_agreement_rate": mean(
                    [
                        row["judge_1_outcome"] == row["judge_2_outcome"]
                        for row in rows
                    ]
                ),
            }
        )

    write_json(aggregate_root / "summary.json", summary_rows)

    csv_path = aggregate_root / "summary.csv"
    if summary_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    write_manifest(
        config,
        stage="05_aggregate",
        inputs={
            "final_scenarios": resolve_path(config, "final_scenarios"),
            **{
                f"judge_{judge}_{path.stem}": path
                for judge in [1, 2]
                for path in (resolve_path(config, "judgments") / f"judge_{judge}").glob("*.jsonl")
            },
        },
        outputs={
            "combined_judgments": aggregate_root / "combined_judgments.jsonl",
            "paired_deltas": aggregate_root / "paired_deltas.jsonl",
            "summary_json": aggregate_root / "summary.json",
            "summary_csv": aggregate_root / "summary.csv",
            "delta_summary": aggregate_root / "delta_summary.json",
        },
        extra={"conditions": conditions},
    )

    print(
        f"Wrote {len(combined_rows)} combined judgments, "
        f"{len(paired_deltas)} paired deltas, and {len(summary_rows)} summary rows."
    )


if __name__ == "__main__":
    main()
