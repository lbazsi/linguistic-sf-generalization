"""LoRA functional DeltaW*x analysis on matched transient base-model inputs."""
from __future__ import annotations
from collections import defaultdict
import torch
from .common import FEATURES,control_for,adapter_dir,load_scenarios,load_general,scenario_prompt,run_dir,write_rows
from .models import load_model,unload
from .lora import lora_matrices

def normalize_key(key):
    # PEFT saved key namespaces can differ from live module namespaces.
    marker="model.layers."
    if marker not in key:
        raise ValueError(f"Unexpected LoRA target naming: {key}")
    return marker+key.split(marker,1)[1]

@torch.inference_mode()
def functional_analysis(cfg,family,checkpoint="final",samples=64):
    """Local module perturbations on shared unsteered base-model input activations.

    For each adapted projection, compute ||DeltaW x|| and DeltaW_feature x -
    DeltaW_control x. Only projected scalar statistics leave device memory.
    """
    from .models import encode
    model,tok=load_model(cfg,family,"base")
    weights={}
    try:
        for condition in ("canonical","control_zh",*FEATURES):
            path=adapter_dir(cfg,family,condition,checkpoint)
            mats,_=lora_matrices(path)
            weights[condition]={normalize_key(key):value for key,value in mats.items()}
        control_modules=weights["canonical"].keys()
        for condition in weights:
            if set(weights[condition])!=set(control_modules):
                raise ValueError(f"LoRA module coverage differs: {condition}")
        modules={normalize_key(name):module for name,module in model.named_modules()
                 if "model.layers." in name and name.endswith((".q_proj",".k_proj",".v_proj",
                    ".o_proj",".gate_proj",".up_proj",".down_proj"))}
        missing=set(control_modules)-set(modules)
        if missing:
            # Normalized paths may still differ by model wrapper; use unambiguous suffix matching.
            fixed={}
            for key in control_modules:
                matches=[v for n,v in modules.items() if n.endswith(key)]
                if len(matches)!=1:raise ValueError(f"Cannot locate unique projection {key}: {len(matches)}")
                fixed[key]=matches[0]
            modules=fixed
        sums=defaultdict(lambda:[0.,0])
        datasets={"welfare":[scenario_prompt(r["scenario"]) for r in load_scenarios(cfg)[:samples]],
                  "general":[r["prompt"] for r in load_general(cfg)[:samples]]}
        for domain,texts in datasets.items():
            for start in range(0,len(texts),cfg["runtime"]["batch_size"]):
                batch=texts[start:start+cfg["runtime"]["batch_size"]]
                encoded=encode(tok,batch,model.device,cfg["runtime"]["max_length"])
                mask=encoded["attention_mask"].float()
                lengths=mask.sum(-1).clamp_min(1)
                handles=[]
                for key,module in modules.items():
                    def pre_hook(_module,args,key=key):
                        x=args[0].detach().float()
                        final=x[:,-1,:]
                        mean=(x*mask.unsqueeze(-1)).sum(1)/lengths.unsqueeze(-1)
                        for pooling,input_x in (("final",final),("mean",mean)):
                            projections={}
                            for condition,maps in weights.items():
                                update=maps[key]
                                output=update.apply(input_x)
                                projections[condition]=output
                                value=float(output.square().mean().sqrt().item())
                                total=sums[(domain,condition,key,pooling,"rms")]
                                total[0]+=value*len(batch);total[1]+=len(batch)
                            for feature in FEATURES:
                                control=control_for(feature)
                                delta=projections[feature]-projections[control]
                                value=float(delta.square().mean().sqrt().item())
                                total=sums[(domain,feature,key,pooling,"relative_control_difference")]
                                total[0]+=value*len(batch);total[1]+=len(batch)
                    handles.append(module.register_forward_pre_hook(pre_hook))
                try:
                    model(**encoded,use_cache=False)
                finally:
                    for handle in handles:handle.remove()
                del encoded
        rows=[{"family":family,"checkpoint":checkpoint,"domain":domain,
               "condition":condition,"module":key,"pooling":pooling,"metric":metric,
               "value":total/n,"n":n}
              for (domain,condition,key,pooling,metric),(total,n) in sums.items() if n]
        path=run_dir(cfg,family,"LoRA-analysis")/f"functional_{checkpoint}.jsonl"
        write_rows(path,rows)
        return path
    finally:unload(model)
