"""Direction extraction and log-likelihood steering screen; no LLM judge calls."""
from __future__ import annotations
import math
from collections import defaultdict
import torch
from .common import (FEATURES,control_for,feature_pairs,load_general,load_screening,
                     run_dir,scenario_prompt,write_rows,write_json,adapter_dir,location)
from .models import (load_model,pooled_batches,unload,blocks,continuation_scores,steer,encode)
from .stats import cluster_interval

@torch.inference_mode()
def direction_for(model,tok,pairs,layer,site,cfg):
    """Paired average of feature minus canonical for semantic IDs in the training partition."""
    from .common import split_groups
    portions=split_groups((r["id"] for r in pairs),int(cfg["seed"]))
    train=[r for r in pairs if portions[r["id"]]=="train"]
    if not train:raise ValueError("Empty direction training data")
    runtime=cfg["runtime"];pool=cfg["steering"]["directions_pooling"]
    num=None;count=0
    for start in range(0,len(train),runtime["batch_size"]):
        subset=train[start:start+runtime["batch_size"]]
        a=next(pooled_batches(model,tok,[r["canonical"] for r in subset],
                             layer,site,pool,len(subset),runtime["max_length"]))
        b=next(pooled_batches(model,tok,[r["feature_variant"] for r in subset],
                             layer,site,pool,len(subset),runtime["max_length"]))
        delta=(b-a).float().sum(0)
        num=delta if num is None else num+delta
        count+=len(subset)
    direction=num/count
    norm=direction.norm()
    if norm.item()<1e-8:raise ValueError("Degenerate contrastive direction")
    return direction/norm, float(norm)

@torch.inference_mode()
def calibrate(model,tok,texts,layer,site,cfg):
    n=0;scalar=0.
    for batch in pooled_batches(model,tok,texts,layer,site,cfg["steering"]["directions_pooling"],
                                cfg["runtime"]["batch_size"],cfg["runtime"]["max_length"]):
        scalar+=(batch.float().norm(dim=-1)/math.sqrt(batch.shape[-1])).sum().item()
        n+=len(batch)
    return scalar/max(n,1)

def screen(cfg,family,feature,adapter_condition,policy="prompt_last"):
    """Screen every layer, site and alpha; paired-option likelihood requires no judges."""
    if adapter_condition not in (control_for(feature),feature):
        raise ValueError("Feature direction may only be applied to its own adapter or control")
    from .directions import load_direction_bundle
    # Preparation loads and unloads the base model before the target adapter is loaded.
    get_direction=load_direction_bundle(cfg,family,feature)
    scenarios=[r for r in load_screening(cfg)
               if r["split"] in cfg["steering"]["screening_splits"]]
    target,tok=load_model(cfg,family,adapter_condition)
    out=run_dir(cfg,family,"activation-steering")
    try:
        records=[]
        prompts=[scenario_prompt(r["scenario"]) for r in scenarios]
        pos=[r["welfare_supporting"] for r in scenarios]
        neg=[r["welfare_opposing"] for r in scenarios]
        for layer in range(len(blocks(target))):
            for site in cfg["runtime"]["sites"]:
                unit,scale=get_direction(layer,site)
                vector=unit*scale
                for strength in cfg["steering"]["alphas"]:
                    values=continuation_scores(target,tok,prompts,pos,neg,layer,site,vector,
                                                float(strength),policy,cfg["runtime"]["max_length"])
                    for r,val in zip(scenarios,values):
                        records.append({"feature":feature,"adapter":adapter_condition,
                                        "layer":layer,"site":site,"strength":float(strength),
                                        "policy":policy,"scenario_id":r["scenario_id"],
                                        "split":r["split"],"category":r["category"],
                                        "tradeoff":r["tradeoff"],"score":val,
                                        "calibration_scale":scale})
        path=out/f"screen_{feature}_{adapter_condition}_{policy}.jsonl"
        write_rows(path,records)
        return path
    finally:unload(target)

def selection(cfg,family,features=FEATURES,policy=None):
    """Validation selects one configuration per feature/adapter over BOTH token policies.

    All layers/sites/strengths and policies are still reported in screening artifacts.
    No test example is used during selection.
    """
    from .common import read_data
    out=run_dir(cfg,family,"activation-steering")
    selected=[]
    for feature in features:
        for condition in (control_for(feature),feature):
            rows=[]
            for token_policy in ("prompt_last","generation"):
                source=out/f"screen_{feature}_{condition}_{token_policy}.jsonl"
                rows.extend(read_data(source))
            baseline={(r["policy"],r["layer"],r["site"],r["scenario_id"]):r["score"]
                      for r in rows if r["split"]=="validation" and r["strength"]==0}
            options=defaultdict(list)
            for r in rows:
                if r["split"]!="validation" or r["strength"]==0:continue
                key=(r["policy"],r["layer"],r["site"],r["strength"])
                base=baseline.get((r["policy"],r["layer"],r["site"],r["scenario_id"]))
                if base is None:raise ValueError("Missing matched unsteered baseline")
                options[key].append(r["score"]-base)
            ranked=sorted(options.items(),key=lambda kv:(-abs(sum(kv[1])/len(kv[1])),kv[0]))
            if not ranked:raise ValueError("No validation configurations to select")
            (token_policy,layer,site,strength),diffs=ranked[0]
            selected.append({"feature":feature,"adapter":condition,"layer":layer,
                             "site":site,"strength":strength,"policy":token_policy,
                             "validation_delta":sum(diffs)/len(diffs),"n_validation":len(diffs)})
    if len(selected)!=2*len(features):raise RuntimeError("Incomplete steering selection")
    path=out/"selected_all.json"
    write_json(path,selected)
    return path

def selected_controls(cfg,family):
    """Held-out-validation control comparisons; no model generations or judge requests."""
    import json
    from .common import read_data
    from .directions import load_direction_bundle
    out=run_dir(cfg,family,"activation-steering")
    selected=json.loads((out/"selected_all.json").read_text(encoding="utf-8"))
    option_rows=[r for r in load_screening(cfg) if r["split"]=="validation"]
    prompts=[scenario_prompt(r["scenario"]) for r in option_rows]
    positive=[r["welfare_supporting"] for r in option_rows]
    negative=[r["welfare_opposing"] for r in option_rows]
    grouped=defaultdict(list)
    for spec in selected:grouped[spec["adapter"]].append(spec)
    results=[]
    for condition,specs in grouped.items():
        get={entry["feature"]:load_direction_bundle(cfg,family,entry["feature"])
             for entry in specs}
        model,tok=load_model(cfg,family,condition)
        try:
            for entry in specs:
                layer,site=entry["layer"],entry["site"]
                scale_strength=float(entry["strength"])
                for kind in ("random","shuffled"):
                    unit,scale=get[entry["feature"]](layer,site,kind)
                    scores=continuation_scores(model,tok,prompts,positive,negative,
                        layer,site,unit*scale,scale_strength,entry["policy"],
                        cfg["runtime"]["max_length"])
                    for row,value in zip(option_rows,scores):
                        results.append({"feature":entry["feature"],"adapter":condition,
                             "site":site,"layer":layer,"strength":scale_strength,
                             "policy":entry["policy"],"control":kind,
                             "scenario_id":row["scenario_id"],"score":value})
        finally:unload(model)
    path=out/"selected_controls.jsonl"
    write_rows(path,results)
    return path
