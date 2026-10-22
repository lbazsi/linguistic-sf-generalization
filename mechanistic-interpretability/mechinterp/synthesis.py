"""Across-family effect-profile comparison; never compare incompatible hidden coordinates."""
from __future__ import annotations
from collections import defaultdict
from .common import FEATURES, read_data,location,write_json

def summarize(cfg):
    root=location(cfg,"outputs")
    families=list(cfg["models"])
    profiles=defaultdict(lambda:defaultdict(dict))
    for family in families:
        for path in (root/family/"model-difference-analysis").glob("*.jsonl"):
            for r in read_data(path):
                if r.get("domain")=="welfare" and r["site"]=="residual" and r["pooling"]=="final":
                    profiles[r["condition"]][family].setdefault("cka",[]).append(r["cka"])
        for path in (root/family/"activation-steering").glob("selected_*.json"):
            for r in read_data(path):
                profiles[r["feature"]][family].setdefault("steering",[]).append(r["validation_delta"])
        module_file=root/family/"LoRA-analysis"/"module_geometry.jsonl"
        if module_file.exists():
            for r in read_data(module_file):
                if r["condition"] in FEATURES and r["checkpoint"]=="final" and "cosine" in r:
                    profiles[r["condition"]][family].setdefault("lora_cosine",[]).append(r["cosine"])
    result=[]
    for feature,by_family in sorted(profiles.items()):
        for family,metrics in sorted(by_family.items()):
            result.append({"feature":feature,"family":family,
                           **{key:sum(values)/len(values) for key,values in metrics.items() if values}})
    output=root/"cross_model_synthesis.json"
    write_json(output,result)
    return output
