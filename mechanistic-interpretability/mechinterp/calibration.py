"""Shared, stratified per-site activation-norm calibration for steering revision 2.

Caches scales only. Existing linguistic direction vectors remain valid; their
original, category-ordered calibration scalars are replaced when loaded.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
import math
import random
import torch
from .common import load_general,location,run_dir,write_json,digest
from .models import load_model,unload,blocks,site_module,tensor_output,pool,encode

def stratified_general(rows,n,seed):
    groups=defaultdict(list)
    for row in rows:
        groups[row["domain"]].append(row)
    if n< len(groups):
        raise ValueError("Calibration count must cover every domain")
    n_groups=len(groups)
    chosen=[]
    for index,(domain,members) in enumerate(sorted(groups.items())):
        rng=random.Random(f"{seed}:{domain}")
        num=n//n_groups + (1 if index < n % n_groups else 0)
        if len(members)<num:raise ValueError(f"Not enough prompts for {domain}")
        chosen.extend(rng.sample(members,num))
    chosen.sort(key=lambda r:str(r["id"]))
    return chosen

@torch.inference_mode()
def calibrate_sites(cfg,family):
    general_source=location(cfg,"general")
    source_sha=digest(general_source)
    path=run_dir(cfg,family,"activation-steering")/"site_calibration_v2.json"
    size=int(cfg["runtime"]["calibration_n"])
    signature={"protocol":"stratified_20_domains_v2",
               "source_sha256":source_sha,"seed":int(cfg["seed"]),
               "model_revision":cfg["models"][family]["revision"],
               "n_requested":size,
               "pooling":cfg["steering"]["directions_pooling"]}
    if path.exists():
        data=json.loads(path.read_text(encoding="utf-8"))
        if any(data.get(k)!=v for k,v in signature.items()):
            raise ValueError("Calibration provenance mismatch; regenerate explicitly")
        if not data.get("scales"):raise ValueError("Missing calibration site scales")
        return data["scales"]
    general=stratified_general(load_general(cfg),size,int(cfg["seed"]))
    model,tok=load_model(cfg,family,"base")
    try:
        sites=[(i,s) for i in range(len(blocks(model))) for s in cfg["runtime"]["sites"]]
        sums={};counts={}
        for start in range(0,len(general),int(cfg["runtime"]["batch_size"])):
            batch=general[start:start+int(cfg["runtime"]["batch_size"])]
            tokens=encode(tok,[r["prompt"] for r in batch],model.device,
                          int(cfg["runtime"]["max_length"]))
            mask=tokens["attention_mask"]
            handles=[]
            for layer,site in sites:
                key=f"{layer}__{site}"
                def hook(_module,_args,out,key=key):
                    hidden=pool(tensor_output(out).detach(),mask,
                                cfg["steering"]["directions_pooling"]).float()
                    norms=hidden.norm(dim=-1)/math.sqrt(hidden.shape[-1])
                    sums[key]=sums.get(key,0.)+float(norms.double().sum())
                    counts[key]=counts.get(key,0)+len(hidden)
                handles.append(site_module(model,layer,site).register_forward_hook(hook))
            try:
                model(**tokens,use_cache=False)
            finally:
                for h in handles:h.remove()
        if any(counts.get(f"{l}__{s}")!=len(general) for l,s in sites):
            raise ValueError("Missing site calibration hooks")
        scales={key:sums[key]/counts[key] for key in sums}
        if any(not math.isfinite(x) or x<=0 for x in scales.values()):
            raise ValueError("Nonfinite/zero calibration scale")
        signature.update({"n_observed":len(general),
                          "domain_counts":dict(Counter(r["domain"] for r in general)),
                          "scales":scales})
        write_json(path,signature)
        return scales
    finally:
        unload(model)
