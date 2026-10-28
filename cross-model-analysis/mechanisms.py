"""Cross-model synthesis of existing mechanistic scalar outputs.

No model weights, prompts, or activations are loaded. Missing stage results are
reported as missing rather than estimated or imputed.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from cross_model import FAMILIES, FEATURES, CROSS_LANGUAGE, average, number, read_json, read_jsonl, write_csv, write_json, pearson

def depth(module):
    import re
    match = re.search(r"(?:layers|h)\.(\d+)\.", str(module))
    return int(match.group(1)) if match else None

def summarize_mechanisms(root: Path, dest: Path):
    out = dest / "mechanistic"
    inventory, normalized, available = [], [], 0
    for model, family in FAMILIES.items():
        home = root / family
        patterns = {
            "representations": home / "model-difference-analysis",
            "probes": home / "linear-probing",
            "lora": home / "LoRA-analysis",
            "steering": home / "activation-steering",
        }
        for stage, directory in patterns.items():
            files = sorted(directory.glob("*.jsonl")) if directory.exists() else []
            inventory.append({"model": model, "stage": stage, "files": len(files),
                              "directory": str(directory), "status": "present" if files else "missing"})
            available += len(files)
        diff = patterns["representations"]
        for path in sorted(diff.glob("*_final.jsonl")) if diff.exists() else []:
            for row in read_jsonl(path):
                if row.get("condition") not in FEATURES or row.get("category") != "all":
                    continue
                for metric in ("cka", "displacement"):
                    v = number(row.get(metric))
                    if v is None:
                        continue
                    normalized.append({"model":model, "stage":"representations",
                                       "feature":row["condition"],"metric":metric,
                                       "domain":row.get("domain"),"site":row.get("site"),
                                       "pooling":row.get("pooling"),"checkpoint":row.get("checkpoint"),
                                       "layer":row.get("layer"),"value":v})
        probes = patterns["probes"]
        for path in sorted(probes.glob("*.jsonl")) if probes.exists() else []:
            if path.name.startswith("behavior_"):
                # Never pool failed legacy behavior probes with a corrected revision.
                continue
            for row in read_jsonl(path):
                test = row.get("test") or {}
                metric = "r2" if row.get("task") == "behavior" else "roc_auc"
                v = number(test.get(metric))
                if v is None: continue
                normalized.append({"model":model, "stage":"probes",
                                   "feature":row.get("feature") or "all",
                                   "metric":f'{row.get("task")}_{metric}',
                                   "condition":row.get("condition"),
                                   "site":row.get("site"), "pooling":row.get("pooling"),
                                   "layer":row.get("layer"),"value":v})
        # Revised behavioral probes live in a distinct, versioned output tree.
        revised = home / "revision_2" / "linear-probing"
        for path in sorted(revised.glob("behavior_*.jsonl")) if revised.exists() else []:
            for row in read_jsonl(path):
                score = row.get("test") or {}
                v = number(score.get("r2"))
                if v is not None:
                    normalized.append({"model": model, "stage": "probes",
                                       "feature": row.get("feature") or "all",
                                       "metric": "behavior_r2", "condition": row.get("condition"),
                                       "site": row.get("site"), "pooling": row.get("pooling"),
                                       "layer": row.get("layer"), "value": v})
        lora = patterns["lora"]
        geometry = lora / "module_geometry.jsonl"
        if geometry.exists():
            for row in read_jsonl(geometry):
                if row.get("condition") not in FEATURES: continue
                for metric in ("cosine","relative_distance","norm_ratio",
                               "left_subspace_cosine","right_subspace_cosine",
                               "relative_base_weight_norm","effective_rank"):
                    v=number(row.get(metric))
                    if v is None: continue
                    normalized.append({"model":model,"stage":"lora",
                                       "feature":row["condition"],"checkpoint":row.get("checkpoint"),
                                       "metric":metric,"module":row.get("module"),
                                       "layer":depth(row.get("module")),"value":v})
        for checkpoint in ("midpoint","final"):
            file = home / "revision_2" / "LoRA-analysis" / f"functional_{checkpoint}.jsonl"
            if not file.exists(): continue
            for row in read_jsonl(file):
                if row.get("condition") not in FEATURES: continue
                v = number(row.get("value"))
                if v is None:continue
                normalized.append({"model":model,"stage":"lora_functional",
                                   "feature":row["condition"],"checkpoint":checkpoint,
                                   "metric":row.get("metric"),"domain":row.get("domain"),
                                   "pooling":row.get("pooling"),"module":row.get("module"),
                                   "layer":depth(row.get("module")),"value":v})
        steering=patterns["steering"]
        selected=steering / "selected_all.json"
        if selected.exists():
            for row in read_json(selected):
                v = number(row.get("validation_delta"))
                if v is None:continue
                normalized.append({"model":model,"stage":"steering_validation",
                                   "feature":row["feature"],"condition":row.get("adapter"),
                                   "site":row.get("site"),"layer":row.get("layer"),
                                   "metric":"validation_proxy_delta","policy":row.get("policy"),
                                   "strength":row.get("strength"),"value":v})
        judged=steering / "judgments" / "summary.json"
        if judged.exists():
            for row in read_json(judged):
                paired=row.get("paired_delta") or {}
                v=number(paired.get("mean"))
                if v is None:continue
                normalized.append({"model":model,"stage":"steering_heldout",
                                   "feature":row["feature"],"condition":row.get("adapter"),
                                   "metric":"judged_target_value_support_delta",
                                   "value":v,"ci_low":paired.get("low"),
                                   "ci_high":paired.get("high"),"q_value":row.get("q_value")})
    write_csv(out/"availability.csv",inventory)
    write_csv(out/"normalized_metrics.csv",normalized)

    # Stratify by metric and domain/site/pooling/checkpoint/condition to avoid
    # comparing incompatible measurements. Average layers/modules descriptively;
    # retain raw scalar records so all aggregation can be audited.
    keys=("stage","feature","metric","domain","site","pooling","checkpoint","condition","policy")
    buckets=defaultdict(lambda:defaultdict(list))
    for row in normalized:
        signature=tuple(row.get(k) for k in keys)
        buckets[signature][row["model"]].append(row["value"])
    profiles=[]
    for signature,models in sorted(buckets.items(),key=lambda item:str(item[0])):
        r=dict(zip(keys,signature))
        for m in FAMILIES:
            r[m]=average(models.get(m,[]))
            r[m+"_n"]=len(models.get(m,[]))
        r["n_families"]=sum(bool(models.get(m)) for m in FAMILIES)
        profiles.append(r)
    write_csv(out/"comparable_profiles.csv",profiles)

    # Associations to behavioral effects are descriptive only and never pool
    # unmatched prompt domains or use validation-selected steering estimates
    # as confirmatory held-out effects.
    correlations=[]
    for stage,metric in sorted({(r["stage"],r["metric"]) for r in profiles}):
        matched=[r for r in profiles if r["stage"]==stage and r["metric"]==metric]
        if not matched:continue
        for m in FAMILIES:
            # Correlation requires behavioral measurements, joined in run.py
            # separately; this module deliberately exports feature-aligned profiles.
            n=sum(r[m] is not None for r in matched)
            correlations.append({"model":m,"stage":stage,"metric":metric,
                                 "available_profile_strata":n})
    write_csv(out/"metric_coverage.csv",correlations)
    summary={"models":list(FAMILIES),"available_files":available,
             "normalized_measurements":len(normalized),"comparable_profiles":len(profiles),
             "inference":"Descriptive only; no missing results imputed and no hidden-space coordinates aligned."}
    write_json(out/"summary.json",summary)
    return summary
