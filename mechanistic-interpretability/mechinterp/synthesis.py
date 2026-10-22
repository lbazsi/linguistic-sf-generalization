"""Synthesize behavioral and mechanistic effect profiles across model families."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from .common import FEATURES,read_data,location,write_json

def _behavior_path(cfg,family):
    raw=cfg["models"][family].get("paired_deltas")
    if raw:
        p=Path(raw)
        return p if p.is_absolute() else (Path(cfg["_config_file"]).parent.parent/p).resolve()
    if family=="gemma2":
        return location(cfg,"judgments").parent/"paired_deltas.jsonl"
    return None

def summarize(cfg):
    root=location(cfg,"outputs")
    families=list(cfg["models"])
    profiles=defaultdict(lambda:defaultdict(dict))
    for family in families:
        behavior=_behavior_path(cfg,family)
        if behavior is not None and behavior.exists():
            effects=defaultdict(list)
            for row in read_data(behavior):
                if row.get("comparison")!="feature_minus_control":continue
                name=row.get("feature") or row.get("condition")
                score=row.get("score_deltas",{}).get("target_value_support")
                if name in FEATURES and score is not None:
                    effects[name].append(float(score))
            for feature,values in effects.items():
                profiles[feature][family]["behavior_delta"]=sum(values)/len(values)
                profiles[feature][family]["behavior_n"]=len(values)
        diff_dir=root/family/"model-difference-analysis"
        if diff_dir.exists():
            for path in diff_dir.glob("*.jsonl"):
                for row in read_data(path):
                    if row.get("domain")=="welfare" and row["site"]=="residual" and row["pooling"]=="final":
                        name=row["condition"]
                        if name in FEATURES:
                            profiles[name][family].setdefault("_cka",[]).append(float(row["cka"]))
        steering=root/family/"activation-steering"/"selected_all.json"
        if steering.exists():
            for row in read_data(steering):
                profiles[row["feature"]][family].setdefault("_steering",[]).append(
                    float(row["validation_delta"]))
        module_file=root/family/"LoRA-analysis"/"module_geometry.jsonl"
        if module_file.exists():
            for row in read_data(module_file):
                if row.get("condition") in FEATURES and row.get("checkpoint")=="final" and "cosine" in row:
                    profiles[row["condition"]][family].setdefault("_lora_cosine",[]).append(
                        float(row["cosine"]))
    records=[]
    for feature,by_family in sorted(profiles.items()):
        for family,metrics in sorted(by_family.items()):
            row={"feature":feature,"family":family}
            for key,value in metrics.items():
                if key.startswith("_"):
                    if value:row[key[1:]]=sum(value)/len(value)
                else:
                    row[key]=value
            records.append(row)
    output=root/"cross_model_synthesis.json"
    write_json(output,records)
    # Sign replication is reported only when all four behavioral datasets exist.
    sign_table=[]
    for feature in FEATURES:
        data=[r for r in records if r["feature"]==feature and "behavior_delta" in r]
        signs={r["family"]:(1 if r["behavior_delta"]>0 else -1 if r["behavior_delta"]<0 else 0)
               for r in data}
        sign_table.append({"feature":feature,"signs":signs,
             "n_families":len(signs),
             "all_four_same_nonzero_sign":len(signs)==4 and len(set(signs.values()))==1 and 0 not in signs.values()})
    write_json(root/"behavioral_sign_replication.json",sign_table)
    return output
