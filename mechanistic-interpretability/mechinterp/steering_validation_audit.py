"""Audit pre-selected welfare likelihood effects against zero and control vectors.

Descriptive only: validation was used to choose layer, site, policy and dose.
Run after steering-select and steering-controls; never opens welfare TEST.
"""
import json
from collections import defaultdict,Counter
from pathlib import Path
from statistics import mean
from .common import load_config,run_dir,read_data,write_json


def evaluate(cfg,model):
    root=run_dir(cfg,model,"activation-steering")
    selected=read_data(root/"selected_all.json")
    controls=read_data(root/"selected_controls.jsonl")
    by_control=defaultdict(dict)
    for r in controls:
        key=(r["feature"],r["adapter"],r["scenario_id"])
        name=r["control"]
        if name in by_control[key]:raise ValueError(f"Duplicate control: {key} / {name}")
        by_control[key][name]=r["score"]
    results=[]
    for spec in selected:
        f,a=spec["feature"],spec["adapter"]
        path=root/f"screen_{f}_{a}_{spec['policy']}.jsonl"
        if not path.is_file():raise FileNotFoundError(path)
        base,steered={},{}
        # Streaming, so the full multi-million-row screen isn't held in RAM.
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():continue
                x=json.loads(line)
                if (x["split"]!="validation" or x["site"]!=spec["site"]
                    or int(x["layer"])!=int(spec["layer"])):continue
                sid=int(x["scenario_id"]);s=float(x["strength"])
                if abs(s)<1e-10:
                    if sid in base:raise ValueError(f"Duplicate baseline {f}/{a}/{sid}")
                    base[sid]=float(x["score"])
                elif abs(s-float(spec["strength"]))<1e-8:
                    if sid in steered:raise ValueError(f"Duplicate selected strength {f}/{a}/{sid}")
                    steered[sid]=float(x["score"])
        if len(base)!=80 or set(base)!=set(steered):
            raise ValueError(f"Missing matched 80 validation rows for {f}/{a}: {len(base)} vs {len(steered)}")
        keys=[(f,a,sid) for sid in base]
        if any(set(by_control.get(k,{}))!={"random","shuffled"} for k in keys):
            raise ValueError(f"Missing random/shuffled controls for {f}/{a}")
        delta=mean(steered[sid]-base[sid] for sid in base)
        if abs(delta-float(spec["validation_delta"]))>1e-4:
            raise ValueError(f"Selected validation delta mismatch for {f}/{a}")
        random=mean(by_control[(f,a,sid)]["random"]-base[sid] for sid in base)
        shuffled=mean(by_control[(f,a,sid)]["shuffled"]-base[sid] for sid in base)
        results.append({"feature":f,"adapter":a,"n_validation":80,
                        "policy":spec["policy"],"layer":spec["layer"],"site":spec["site"],
                        "strength":spec["strength"],"selected_delta":delta,
                        "random_delta":random,"shuffled_delta":shuffled,
                        "selected_minus_random":delta-random,
                        "selected_minus_shuffled":delta-shuffled,
                        "exceeds_both_null_abs":abs(delta)>max(abs(random),abs(shuffled))})
    result={"model":model,"version":"steering_validation_audit_v1",
            "interpretation":"Descriptive only: the same validation data selected the configuration; apparent excess over null controls is not a confirmatory p-value.",
            "n_configurations":len(results),
            "at_boundary":sum(abs(float(r["strength"]))>=15.999 for r in results),
            "generation_policy_count":sum(r["policy"]=="generation" for r in results),
            "n_exceed_both_null_abs":sum(r["exceeds_both_null_abs"] for r in results),
            "results":results}
    write_json(root/"selected_validation_audit.json",result)
    print("Configs:",len(results),"At |strength|=16:",result["at_boundary"],
          "Exceed both null magnitudes:",result["n_exceed_both_null_abs"])
    for x in results:
        print(f"{x['feature']:25} {x['adapter']:25} selected={x['selected_delta']:+.4f} random={x['random_delta']:+.4f} shuffled={x['shuffled_delta']:+.4f} exceeds={x['exceeds_both_null_abs']}")
    print(root/"selected_validation_audit.json")
    return result


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,choices=("gemma2","llama31","mistral","qwen3"))
    p.add_argument("--config",default="config/pipeline.yaml")
    x=p.parse_args();evaluate(load_config(x.config),x.model)

if __name__=="__main__":main()
