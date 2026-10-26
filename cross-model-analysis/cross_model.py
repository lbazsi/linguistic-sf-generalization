"""Cross-family comparison of completed black-box evaluations.

Uses published, per-family statistical tables without recomputing inference or
treating architectures evaluated on shared scenarios as independent samples.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

FAMILIES = {"gemma": "gemma2", "llama31": "llama31",
            "mistral": "mistral", "qwen3": "qwen3"}
FEATURES = (
    "conditionality", "constituent_order", "epistemic_hedging",
    "formality", "fusion", "genericity", "inflectional_synthesis",
    "modal_strength", "narrative_framing", "nominalization",
    "pronoun_distribution", "subordination", "tense_aspect", "voice",
)
CROSS_LANGUAGE = {"constituent_order", "fusion", "inflectional_synthesis"}
TABLES = {
    "overall": "overall_feature_effects.csv",
    "category": "feature_by_category.csv",
    "heldout": "feature_by_heldout_dimensions.csv",
    "tradeoff": "feature_by_category_tradeoff.csv",
    "interaction": "tradeoff_interactions.csv",
    "priority": "feature_tradeoff_priority_by_category.csv",
    "commitment": "overall_feature_effects_behavioral_commitment.csv",
    "coherence": "overall_feature_effects_coherence_relevance.csv",
    "canonical": "canonical_minus_base_by_category.csv",
    "mandarin": "control_minus_base_by_category.csv",
    "judge": "judge_reliability_overall.csv",
    "sensitivity": "sensitivity_overall_feature_effects.csv",
    "quality": "response_quality_by_condition_category.csv",
}

def number(value):
    if value is None or str(value).strip().lower() in ("", "nan", "none", "na"):
        return None
    v = float(value)
    return v if math.isfinite(v) else None

def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")

def average(values):
    v = [float(x) for x in values if x is not None]
    return statistics.mean(v) if v else None

def sign(value):
    if value is None:
        return 0
    return (value > 0) - (value < 0)

def ranks(values):
    """One-based midranks: every member of a tie receives the same rank."""
    order = sorted(range(len(values)), key=lambda k: values[k])
    out = [0.] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        midrank = (start + 1 + stop) / 2
        for pos in range(start, stop):
            out[order[pos]] = midrank
        start = stop
    return out

def pearson(x, y):
    if len(x) < 3:
        return None
    xm, ym = average(x), average(y)
    a, b = [z - xm for z in x], [z - ym for z in y]
    denom = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
    return sum(i*j for i, j in zip(a, b)) / denom if denom else None

def describe_group(entries, field="mean"):
    """Descriptive four-family replication, never an independent-family p-value."""
    ordered = {m: x for m, x in entries.items()}
    vals = {m: number(row.get(field)) for m, row in ordered.items()}
    present = {m: v for m, v in vals.items() if v is not None}
    q = {m: number(row.get("q_value")) for m, row in ordered.items()}
    n = len(present)
    out = {
        "families": n,
        "mean_across_families": average(present.values()),
        "median_across_families": statistics.median(present.values()) if n else None,
        "min": min(present.values()) if n else None,
        "max": max(present.values()) if n else None,
        "sd_across_families": statistics.stdev(present.values()) if n > 1 else None,
        "same_nonzero_sign": n == 4 and len({sign(v) for v in present.values()}) == 1
                             and all(v != 0 for v in present.values()),
        "negative_families": sum(v < 0 for v in present.values()),
        "positive_families": sum(v > 0 for v in present.values()),
        "fdr_significant_families": sum(v is not None and v < .05 for v in q.values()),
        "same_sign_and_fdr_all_four": n == 4 and
            len({sign(v) for v in present.values()}) == 1 and
            all(v != 0 for v in present.values()) and
            len(q) == 4 and all(v is not None and v < .05 for v in q.values()),
    }
    for m in FAMILIES:
        out[m + "_effect"] = vals.get(m)
        out[m + "_q"] = q.get(m)
    return out

def stratum_replication(data, key, columns, value_col="mean"):
    groups = defaultdict(dict)
    for model, tables in data.items():
        for row in tables[key]:
            if row.get("feature") not in FEATURES:
                continue
            signature = tuple(row.get(c) for c in columns)
            if model in groups[signature]:
                raise ValueError(f"Duplicate {key} stratum {model} {signature}")
            groups[signature][model] = row
    results = []
    for keys, matching in sorted(groups.items()):
        record = dict(zip(columns, keys))
        if "feature" in record:
            record["language_contrast"] = ("cross_language" if record["feature"]
                                           in CROSS_LANGUAGE else "within_english")
        record.update(describe_group(matching, value_col))
        results.append(record)
    return results

def load_behavioral(root: Path):
    all_data = {}
    for model in FAMILIES:
        folder = (root / "results" / "black-box" / model / "evaluation_results" /
                  "analysis" / "output")
        tables = {key: read_csv(folder / "tables" / name)
                  for key, name in TABLES.items()}
        features = {row["feature"] for row in tables["overall"]}
        if features != set(FEATURES) or len(tables["overall"]) != len(FEATURES):
            raise ValueError(f"Incomplete or duplicate features for {model}: {features}")
        registered = {row["feature"]: row["control"] for row in tables["overall"]}
        for feature, control in registered.items():
            expected = "control_zh" if feature in {"fusion", "inflectional_synthesis"} else "canonical"
            if control != expected:
                raise ValueError(f"Incorrect registered control: {model}/{feature}")
        for row in tables["overall"]:
            if number(row["mean"]) is None or number(row["q_value"]) is None:
                raise ValueError(f"Missing primary estimate for {model}/{row['feature']}")
        tables["lexical"] = read_json(folder / "lexical_diversity_association.json")
        all_data[model] = tables
    return all_data

def summarize_behavioral(root: Path, dest: Path):
    data = load_behavioral(root)
    out = dest / "behavioral"
    records = stratum_replication(data, "overall", ["feature"])
    for row in records:
        row["control"] = ("control_zh" if row["feature"] in
                           {"fusion", "inflectional_synthesis"} else "canonical")
    write_csv(out / "feature_replication.csv", records)
    for key, cols, col in [
        ("category", ["feature", "category"], "mean"),
        ("heldout", ["feature", "heldout_dimensions"], "mean"),
        ("tradeoff", ["feature", "category", "tradeoff"], "mean"),
        ("interaction", ["feature", "category"], "tradeoff_minus_nontradeoff"),
        ("priority", ["feature", "category"], "mean"),
        ("commitment", ["feature"], "mean"),
        ("coherence", ["feature"], "mean"),
    ]:
        write_csv(out / (key + "_replication.csv"), stratum_replication(data, key, cols, col))
    baseline = []
    for m, tables in data.items():
        for type_, key in [("canonical_minus_base", "canonical"),
                           ("mandarin_control_minus_base", "mandarin")]:
            rows = tables[key]
            valid = [(number(r["mean"]), number(r["n"])) for r in rows]
            if any(mean is None or n is None for mean, n in valid):
                raise ValueError(f"Missing baseline category for {m}/{key}")
            total = sum(n for _, n in valid)
            baseline.append({"model": m, "contrast": type_,
                             "weighted_category_mean": sum(v*n for v, n in valid)/total,
                             "n": total, "categories": len(rows)})
    write_csv(out / "baseline_gains.csv", baseline)
    family_tables = []
    for m, tables in data.items():
        for row in tables["overall"]:
            family_tables.append({"model": m, "feature": row["feature"],
                                  "control": row["control"], "n": row["n"],
                                  "n_clusters": row["n_clusters"],
                                  "mean": row["mean"], "ci_low": row["ci_low"],
                                  "ci_high": row["ci_high"], "q_value": row["q_value"],
                                  "standardized_dz": row["standardized_dz"]})
    write_csv(out / "per_family_effects.csv", family_tables)
    correlations = []
    for scope, features in [("all_14", list(FEATURES)),
                            ("within_english_11", [f for f in FEATURES if f not in CROSS_LANGUAGE])]:
        for i, first in enumerate(FAMILIES):
            for second in list(FAMILIES)[i+1:]:
                a = {r["feature"]: number(r["mean"]) for r in data[first]["overall"]}
                b = {r["feature"]: number(r["mean"]) for r in data[second]["overall"]}
                x = [a[f] for f in features]
                y = [b[f] for f in features]
                loo = [pearson(x[:k]+x[k+1:], y[:k]+y[k+1:]) for k in range(len(x))]
                loo = [v for v in loo if v is not None]
                correlations.append({"scope": scope, "first": first, "second": second,
                                     "n_features": len(features), "pearson": pearson(x,y),
                                     "spearman": pearson(ranks(x), ranks(y)),
                                     "sign_agreement": sum(sign(v)==sign(w) for v,w in zip(x,y))/len(x),
                                     "leave_one_feature_out_r_min": min(loo) if loo else None,
                                     "leave_one_feature_out_r_max": max(loo) if loo else None})
    write_csv(out / "cross_family_correlations.csv", correlations)
    lexical, judge, sensitivity, quality = [], [], [], []
    for m, tables in data.items():
        ls = tables["lexical"]
        lexical.append({"model":m, "n_features":ls.get("n"),
                        "pearson_r":ls.get("pearson_r"),
                        "pearson_permutation_p":ls.get("pearson_permutation_p"),
                        "spearman_rho":ls.get("spearman_rho"),
                        "spearman_permutation_p":ls.get("spearman_permutation_p")})
        judge.extend({"model":m, **r} for r in tables["judge"])
        sensitivity.extend({"model":m, **r} for r in tables["sensitivity"])
        quality.extend({"model":m, **r} for r in tables["quality"])
    write_csv(out / "lexical_associations.csv", lexical)
    write_csv(out / "judge_reliability.csv", judge)
    write_csv(out / "sensitivity.csv", sensitivity)
    write_csv(out / "response_quality.csv", quality)
    summary = {"models":list(FAMILIES), "n_features":len(FEATURES),
               "common_fdr_discoveries": [r["feature"] for r in records
                                           if r["same_sign_and_fdr_all_four"]],
               "four_family_sign_replication": [r["feature"] for r in records
                                                 if r["same_nonzero_sign"]],
               "notes": "Descriptive summaries of existing per-family cluster-aware analyses; no cross-family independence assumed."}
    write_json(out / "summary.json", summary)
    return summary
