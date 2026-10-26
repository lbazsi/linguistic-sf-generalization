"""Stratified, token-mask-aware local LoRA functional readouts (revision 2).

RMS is sqrt(sum squared entries / number of entries), not a mean of batch RMS.
Fixed samples and base activations are matched across conditions/checkpoints.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import random
import torch
from .common import FEATURES, control_for, adapter_dir, load_scenarios, load_general, scenario_prompt, run_dir, write_rows, write_json
from .models import load_model, unload
from .lora import lora_matrices

def normalize_key(key):
    marker = "model.layers."
    if marker not in key:
        raise ValueError(f"Unexpected LoRA target naming: {key}")
    return marker + key.split(marker,1)[1]

def stratified_sample(rows, label, per_stratum, seed):
    strata = defaultdict(list)
    for row in rows:
        strata[label(row)].append(row)
    selected = []
    for name, members in sorted(strata.items(), key=lambda item: str(item[0])):
        if len(members) < per_stratum:
            raise ValueError(f"Insufficient observations in stratum {name}: {len(members)}")
        rng = random.Random(f"{seed}:{name}")
        selected.extend(rng.sample(members, per_stratum))
    selected.sort(key=lambda row: str(row["id"]))
    return selected

class Energy:
    def __init__(self):
        self.squared = 0.
        self.elements = 0
        self.examples = 0
    def add(self, tensor):
        z = tensor.float()
        self.squared += float(z.square().sum(dtype=torch.float64))
        self.elements += z.numel()
        self.examples += z.shape[0]
    def rms(self):
        return (self.squared / self.elements)**0.5 if self.elements else None

@torch.inference_mode()
def functional_analysis(cfg, family, checkpoint="final", samples=None):
    from .models import encode
    model,tok=load_model(cfg,family,"base")
    weights={}
    try:
        for condition in ("canonical","control_zh",*FEATURES):
            mats,_=lora_matrices(adapter_dir(cfg,family,condition,checkpoint))
            weights[condition]={normalize_key(k):v for k,v in mats.items()}
        reference=set(weights["canonical"])
        if any(set(m)!=reference for m in weights.values()):
            raise ValueError("LoRA projection coverage differs between conditions")
        all_modules={normalize_key(k):m for k,m in model.named_modules()
                     if "model.layers." in k and k.endswith(
                         (".q_proj",".k_proj",".v_proj",".o_proj",".gate_proj",".up_proj",".down_proj"))}
        modules={}
        for key in sorted(reference):
            matches=[v for k,v in all_modules.items() if k==key or k.endswith(key)]
            if len(matches)!=1:raise ValueError(f"Ambiguous module {key} ({len(matches)} matches)")
            modules[key]=matches[0]
        welfare=stratified_sample(load_scenarios(cfg),
                       lambda x:(x["category"],bool(x["tradeoff"])),10,int(cfg["seed"]))
        general=stratified_sample(load_general(cfg),lambda x:x["domain"],10,int(cfg["seed"]))
        datasets={"welfare":[scenario_prompt(r["scenario"]) for r in welfare],
                  "general":[r["prompt"] for r in general]}
        sums=defaultdict(Energy)
        for domain,texts in datasets.items():
            for start in range(0,len(texts),int(cfg["runtime"]["batch_size"])):
                batch=texts[start:start+int(cfg["runtime"]["batch_size"])]
                encoded=encode(tok,batch,model.device,int(cfg["runtime"]["max_length"]))
                mask=encoded["attention_mask"].float()
                lengths=mask.sum(-1).clamp_min(1)
                handles=[]
                for key,module in modules.items():
                    def pre_hook(_module,args,key=key):
                        x=args[0].detach().float()
                        final=x[:,-1,:]
                        mean=(x*mask.unsqueeze(-1)).sum(1)/lengths.unsqueeze(-1)
                        for pooling,input_x in (("final",final),("mean",mean)):
                            projections={name:update.apply(input_x)
                                         for name, updates in weights.items()
                                         for update in [updates[key]]}
                            for name,value in projections.items():
                                sums[(domain,name,key,pooling,"rms")].add(value)
                            for feature in FEATURES:
                                control=control_for(feature)
                                sums[(domain,feature,key,pooling,"relative_control_difference")].add(
                                    projections[feature]-projections[control])
                    handles.append(module.register_forward_pre_hook(pre_hook))
                try:
                    model(**encoded,use_cache=False)
                finally:
                    for h in handles:h.remove()
        rows=[{"family":family,"checkpoint":checkpoint,"domain":domain,
               "condition":condition,"module":key,"pooling":pooling,"metric":metric,
               "value":energy.rms(),"n":energy.examples,"elements":energy.elements,
               "sampling":"stratified_v2"}
               for (domain,condition,key,pooling,metric),energy in sums.items()]
        path=run_dir(cfg,family,"LoRA-analysis")/f"functional_{checkpoint}.jsonl"
        write_rows(path,rows)
        write_json(path.with_name(f"functional_{checkpoint}_sampling.json"),
                   {"seed":cfg["seed"],"checkpoint":checkpoint,
                    "welfare_n":len(welfare),"general_n":len(general),
                    "welfare_strata":dict(Counter(f"{r['category']}|{r['tradeoff']}" for r in welfare)),
                    "general_strata":dict(Counter(r["domain"] for r in general)),
                    "rms":"sqrt(sum(element^2)/number_of_elements), over all selected examples"})
        return path
    finally:unload(model)
