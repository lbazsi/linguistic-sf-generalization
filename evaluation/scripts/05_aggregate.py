from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import ensure_directories, load_config, read_jsonl, resolve_path, write_json, write_jsonl_atomic


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

    print(f"Wrote {len(combined_rows)} combined judgments and {len(summary_rows)} summary rows.")


if __name__ == "__main__":
    main()
