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
    if adapter_condition not in (control_for(feature),feature):
        raise ValueError("Feature direction may only be applied to its own adapter or control")
    pairs=feature_pairs(cfg,feature)
    rows=load_screening(cfg)
    scenarios=[r for r in rows if r["split"] in cfg["steering"]["screening_splits"]]
    general=[r["prompt"] for r in load_general(cfg)[:cfg["runtime"]["calibration_n"]]]
    base,tok=load_model(cfg,family,"base")
    target,_=load_model(cfg,family,adapter_condition)
    out=run_dir(cfg,family,"activation-steering")
    try:
        result=[]
        for layer in range(len(blocks(base))):
            for site in cfg["runtime"]["sites"]:
                d,norm=direction_for(base,tok,pairs,layer,site,cfg)
                scale=calibrate(base,tok,general,layer,site,cfg)
                vector=d*scale
                prompts=[scenario_prompt(r["scenario"]) for r in scenarios]
                pos=[r["welfare_supporting"] for r in scenarios]
                neg=[r["welfare_opposing"] for r in scenarios]
                # Inputs are explicitly continuation strings after the fixed evaluation prefix.
                for strength in cfg["steering"]["alphas"]:
                    values=continuation_scores(target,tok,prompts,pos,neg,layer,site,vector,
                                                float(strength),policy,cfg["runtime"]["max_length"])
                    for r,val in zip(scenarios,values):
                        result.append({"feature":feature,"adapter":adapter_condition,
                                       "layer":layer,"site":site,"strength":float(strength),
                                       "policy":policy,"scenario_id":r["scenario_id"],
                                       "split":r["split"],"category":r["category"],
                                       "tradeoff":r["tradeoff"],"score":val,
                                       "direction_norm":norm,"calibration_scale":scale})
        path=out/f"screen_{feature}_{adapter_condition}_{policy}.jsonl"
        write_rows(path,result)
        return path
    finally:unload(base);unload(target)

def selection(cfg,family,features=FEATURES,policy="prompt_last"):
    """Choose one layer/site/strength per feature and paired adapter, on validation only."""
    from .common import read_data
    out=run_dir(cfg,family,"activation-steering")
    selected=[]
    for feature in features:
        for condition in (control_for(feature),feature):
            path=out/f"screen_{feature}_{condition}_{policy}.jsonl"
            rows=read_data(path)
            baseline={(r["layer"],r["site"],r["scenario_id"]):r["score"] for r in rows
                      if r["split"]=="validation" and r["strength"]==0}
            options=defaultdict(list)
            for r in rows:
                if r["split"]!="validation" or r["strength"]==0:continue
                key=(r["layer"],r["site"],r["strength"])
                base=baseline.get((r["layer"],r["site"],r["scenario_id"]))
                if base is None:raise ValueError("Missing paired unsteered baseline")
                options[key].append(r["score"]-base)
            ranked=sorted(options.items(),key=lambda kv:(-abs(sum(kv[1])/len(kv[1])),kv[0]))
            if not ranked:raise ValueError("No validation configurations to select")
            (layer,site,strength),diffs=ranked[0]
            selected.append({"feature":feature,"adapter":condition,"layer":layer,
                             "site":site,"strength":strength,"policy":policy,
                             "validation_delta":sum(diffs)/len(diffs),"n_validation":len(diffs)})
    if len(selected)!=2*len(features):raise RuntimeError("Incomplete steering selection")
    path=out/f"selected_{policy}.json"
    write_json(path,selected)
    return path
