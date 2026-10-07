from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from huggingface_hub import model_info
from transformers import AutoTokenizer

from common import (
    ensure_directories,
    load_canonical_corpus,
    load_config,
    read_jsonl_tolerant,
    resolve_path,
    write_json_atomic,
    write_jsonl_atomic,
)


def read_jsonl(path: Path) -> list[dict]:
    parsed, errors = read_jsonl_tolerant(path)
    if errors:
        raise RuntimeError(f"{path} contains invalid JSONL: {errors[:3]}")
    rows = []
    for _, row in parsed:
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} contains a non-object row.")
        rows.append(row)
    return rows


def mattr(token_ids: list[int], window: int) -> float | None:
    if not token_ids:
        return None
    if len(token_ids) <= window:
        return len(set(token_ids)) / len(token_ids)
    values = [
        len(set(token_ids[start : start + window])) / window
        for start in range(0, len(token_ids) - window + 1)
    ]
    return mean(values)


def metrics(tokenizer, text: str, window: int) -> dict:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    count = len(token_ids)
    unique = len(set(token_ids))
    return {
        "token_count": count,
        "unique_token_count": unique,
        "ttr": (unique / count) if count else None,
        "mattr": mattr(token_ids, window),
    }


def main() -> None:
    config = load_config()
    ensure_directories(config)
    metric_cfg = config["lexical_diversity"]
    requested_revision = str(metric_cfg.get("tokenizer_revision") or "main")
    model_name = str(metric_cfg["tokenizer_model"])
    info = model_info(model_name, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve tokenizer revision for {model_name}.")
    resolved_revision = info.sha
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=resolved_revision,
        use_fast=True,
    )
    window = int(metric_cfg["mattr_window"])

    conditions: list[dict] = []
    canonical = load_canonical_corpus(config)
    conditions.append(
        {
            "condition": "canonical",
            "kind": "canonical",
            "language": config["canonical_language"],
            "manipulation_level": "control",
            "comparison_control": None,
            "rows": list(canonical.values()),
            "text_field": "canonical",
        }
    )

    control_dir = resolve_path(config, "language_controls")
    for path in sorted(control_dir.glob("*.jsonl")):
        if path.name.endswith(".generated.jsonl"):
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        conditions.append(
            {
                "condition": f"control_{path.stem}",
                "kind": "control",
                "language": path.stem,
                "manipulation_level": "control",
                "comparison_control": None,
                "rows": rows,
                "text_field": "canonical",
            }
        )

    final_dir = resolve_path(config, "final")
    for path in sorted(final_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        first = rows[0]
        source_language = first["canonical_language"]
        control = (
            "canonical"
            if source_language == config["canonical_language"]
            else f"control_{source_language}"
        )
        conditions.append(
            {
                "condition": path.stem,
                "kind": "feature",
                "language": first["feature_variant_language"],
                "manipulation_level": first["manipulation_level"],
                "comparison_control": control,
                "rows": rows,
                "text_field": "feature_variant",
            }
        )

    item_rows = []
    by_condition_id: dict[tuple[str, int], dict] = {}
    for condition in conditions:
        for row in condition["rows"]:
            values = metrics(tokenizer, row[condition["text_field"]], window)
            item = {
                "id": int(row["id"]),
                "condition": condition["condition"],
                "kind": condition["kind"],
                "language": condition["language"],
                "manipulation_level": condition["manipulation_level"],
                "comparison_control": condition["comparison_control"],
                **values,
            }
            item_rows.append(item)
            by_condition_id[(item["condition"], item["id"])] = item

    output_dir = resolve_path(config, "lexical_diversity")
    write_jsonl_atomic(output_dir / "items.jsonl", item_rows)

    grouped = defaultdict(list)
    for row in item_rows:
        grouped[row["condition"]].append(row)

    summary = []
    for condition, rows in sorted(grouped.items()):
        summary.append(
            {
                "condition": condition,
                "kind": rows[0]["kind"],
                "language": rows[0]["language"],
                "manipulation_level": rows[0]["manipulation_level"],
                "comparison_control": rows[0]["comparison_control"],
                "n": len(rows),
                "mean_token_count": mean(row["token_count"] for row in rows),
                "mean_ttr": mean(row["ttr"] for row in rows if row["ttr"] is not None),
                "mean_mattr": mean(row["mattr"] for row in rows if row["mattr"] is not None),
            }
        )
    write_json_atomic(
        output_dir / "summary.json",
        {
            "tokenizer": {
                "model": model_name,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
            },
            "mattr_window": window,
            "conditions": summary,
        },
    )

    shifts = []
    for condition in conditions:
        if condition["kind"] != "feature":
            continue
        control = condition["comparison_control"]
        rows = grouped[condition["condition"]]
        deltas = []
        for row in rows:
            baseline = by_condition_id.get((control, row["id"]))
            if baseline is None:
                raise RuntimeError(
                    f"Missing lexical-diversity baseline {control} for ID {row['id']}."
                )
            deltas.append(
                {
                    "token_count": row["token_count"] - baseline["token_count"],
                    "ttr": (
                        row["ttr"] - baseline["ttr"]
                        if row["ttr"] is not None and baseline["ttr"] is not None
                        else None
                    ),
                    "mattr": (
                        row["mattr"] - baseline["mattr"]
                        if row["mattr"] is not None and baseline["mattr"] is not None
                        else None
                    ),
                }
            )
        baseline_language = grouped[control][0]["language"]
        shifts.append(
            {
                "feature": condition["condition"],
                "manipulation_level": condition["manipulation_level"],
                "feature_language": condition["language"],
                "control": control,
                "control_language": baseline_language,
                "same_language": condition["language"] == baseline_language,
                "n": len(deltas),
                "mean_token_count_delta": mean(d["token_count"] for d in deltas),
                "mean_ttr_delta": mean(d["ttr"] for d in deltas if d["ttr"] is not None),
                "mean_mattr_delta": mean(d["mattr"] for d in deltas if d["mattr"] is not None),
            }
        )

    write_json_atomic(
        output_dir / "pairwise_shifts.json",
        {
            "metric_for_primary_confound_analysis": "mattr",
            "cross_language_policy": (
                "Measure all languages, but exclude cross-language shifts from the primary "
                "lexical-diversity correlation because tokenizer-level diversity is not directly "
                "comparable across languages."
            ),
            "features": shifts,
        },
    )
    print(f"Wrote lexical-diversity metrics for {len(grouped)} training conditions.")


if __name__ == "__main__":
    main()
