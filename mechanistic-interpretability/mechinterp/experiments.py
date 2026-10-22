"""GPU experiment runners. Raw activations are reduced and discarded in memory."""
from __future__ import annotations
import json
from collections import defaultdict
import torch
from .common import (FEATURES, control_for, feature_pairs, load_concepts, load_general,
                     load_scenarios, location, run_dir, scenario_prompt, split_groups,
                     write_json, write_rows, adapter_dir, provenance, model_conditions, batches)
from .models import load_model, pooled_batches, unload, blocks
from .stats import linear_cka, svcca, rms_displacement
from .probes import train_probe
from .lora import lora_matrices,geometry,compare

def layers_and_sites(model,cfg):
    return [(layer,site,pooling) for layer in range(len(blocks(model)))
            for site in cfg["runtime"]["sites"] for pooling in cfg["runtime"]["pooling"]]

def collect(model,tok,texts,layer,site,pooling,cfg):
    # Tensor exists exclusively on the accelerator, until this single comparison finishes.
    return torch.cat(list(pooled_batches(model,tok,texts,layer,site,pooling,
                          int(cfg["runtime"]["batch_size"]),int(cfg["runtime"]["max_length"]))),dim=0)

def difference(cfg,family,condition,checkpoint="final"):
    control = control_for(condition) if condition in FEATURES else "base"
    scenarios=load_scenarios(cfg)
    datasets={"welfare":[scenario_prompt(s["scenario"]) for s in scenarios],
              "general":[r["prompt"] for r in load_general(cfg)]}
    out=run_dir(cfg,family,"model-difference-analysis")
    result_file=out/f"{condition}_{checkpoint}.jsonl"
    model_a,tok=load_model(cfg,family,control,checkpoint)
    model_b,_=load_model(cfg,family,condition,checkpoint)
    try:
        records=[]
        for domain,texts in datasets.items():
            for layer,site,pool in layers_and_sites(model_a,cfg):
                a=collect(model_a,tok,texts,layer,site,pool,cfg)
                b=collect(model_b,tok,texts,layer,site,pool,cfg)
                if a.shape != b.shape: raise ValueError("Mismatched activation dimensions")
                metrics={"cka":linear_cka(a,b),"svcca":svcca(a,b,
                    float(cfg["analysis"]["svcca_variance"]),int(cfg["analysis"]["svcca_components_cap"])),
                    "displacement":rms_displacement(a,b)}
                records.append({"model":family,"condition":condition,"control":control,
                                "checkpoint":checkpoint,"domain":domain,"layer":layer,
                                "site":site,"pooling":pool,"n":len(texts),**metrics})
                del a,b
        write_rows(result_file,records)
        write_json(out/f"{condition}_{checkpoint}_manifest.json",
                   provenance(cfg,family,"difference",[location(cfg,"scenarios"),location(cfg,"general")]))
        return result_file
    finally:
        unload(model_a);unload(model_b)

def probe_dataset(cfg,task,feature=None,judgments_path=None,condition=None):
    seed=int(cfg["seed"])
    if task=="concept":
        raw=load_concepts(cfg)
        groups=split_groups((r["pair_id"] for r in raw),seed)
        return [{"text":r["text"],"label":r["label"],"partition":groups[r["pair_id"]]}
                for r in raw]
    if task=="linguistic":
        if feature is None:raise ValueError("Linguistic probe requires feature")
        raw=feature_pairs(cfg,feature)
        parts=split_groups((r["id"] for r in raw),seed)
        result=[]
        for r in raw:
            for label,key in ((0,"canonical"),(1,"feature_variant")):
                result.append({"text":r[key],"label":label,"partition":parts[r["id"]]})
        return result
    if task=="behavior":
        if condition is None:raise ValueError("Behavior probe requires condition")
        rows=load_scenarios(cfg)
        scores={}
        source=location(cfg,"judgments") if judgments_path is None else judgments_path
        from .common import read_data
        for r in read_data(source):
            if r.get("condition")==condition and r.get("scores",{}).get("target_value_support") is not None:
                scores[int(r["scenario_id"])]=r["scores"]["target_value_support"]
        if not scores:raise ValueError(f"No aggregated behavioral labels for {condition}")
        cells={r["id"]:(r["category"],r["animal"],r["value"],r["context"]) for r in rows}
        groups=split_groups((cells[r["id"]] for r in rows),seed)
        return [{"text":scenario_prompt(r["scenario"]),"label":scores[r["id"]],
                 "partition":groups[cells[r["id"]]]} for r in rows if r["id"] in scores]
    raise ValueError(task)

def probing(cfg,family,condition,task,feature=None):
    data=probe_dataset(cfg,task,feature=feature,condition=condition)
    subsets={part:[r for r in data if r["partition"]==part]
             for part in ("train","development","test")}
    model,tok=load_model(cfg,family,condition)
    try:
        records=[]
        for layer,site,pool in layers_and_sites(model,cfg):
            fit=train_probe(model,tok,subsets["train"],subsets["development"],subsets["test"],
                            layer,site,pool,
                            "regression" if task=="behavior" else "classification",cfg)
            records.append({"layer":layer,"site":site,"pooling":pool,
                            "task":task,"feature":feature,"condition":condition,
                            "development":fit["validation"],"test":fit["test"]})
        output=run_dir(cfg,family,"linear-probing")/f"{task}_{feature or 'all'}_{condition}.jsonl"
        write_rows(output,records)
        return output
    finally: unload(model)

def lora_analysis(cfg,family):
    from .common import FEATURES
    out=run_dir(cfg,family,"LoRA-analysis")
    rows=[]
    for checkpoint in ("midpoint","final"):
        weights={}
        for condition in ("canonical","control_zh",*FEATURES):
            path=adapter_dir(cfg,family,condition,checkpoint)
            weights[condition]=lora_matrices(path)[0]
        for condition,modules in weights.items():
            for module,m in modules.items():
                entry={"condition":condition,"checkpoint":checkpoint,
                       "module":module,**geometry(m)}
                if condition in FEATURES:
                    ctrl=control_for(condition)
                    if module not in weights[ctrl]:
                        raise ValueError(f"Module {module} absent from {ctrl}")
                    entry.update({"control":ctrl,**compare(weights[ctrl][module],m)})
                rows.append(entry)
        del weights
    # Compare each module's midpoint and final geometry.
    for condition in ("canonical","control_zh",*FEATURES):
        mid=lora_matrices(adapter_dir(cfg,family,condition,"midpoint"))[0]
        end=lora_matrices(adapter_dir(cfg,family,condition,"final"))[0]
        for module in mid:
            if module in end:
                rows.append({"condition":condition,"checkpoint":"midpoint_to_final",
                             "module":module,**compare(mid[module],end[module])})
    output=out/"module_geometry.jsonl"
    write_rows(output,rows)
    return output
