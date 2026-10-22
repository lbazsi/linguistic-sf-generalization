"""Effective LoRA update geometry without saving dense update matrices."""
import re
import torch
from .common import control_for

def lora_matrices(adapter_path):
    from safetensors import safe_open
    from pathlib import Path
    from peft import PeftConfig
    adapter_path=Path(adapter_path)
    config=PeftConfig.from_pretrained(str(adapter_path))
    files=list(adapter_path.glob("*.safetensors"))
    if len(files)!=1:
        raise FileNotFoundError(f"Expected one safetensors adapter at {adapter_path}")
    pairs={}
    with safe_open(files[0],framework="pt",device="cpu") as f:
        for key in f.keys():
            if ".lora_A." in key:
                target=key.split(".lora_A.")[0]
                pairs.setdefault(target,{})["A"]=f.get_tensor(key)
            if ".lora_B." in key:
                target=key.split(".lora_B.")[0]
                pairs.setdefault(target,{})["B"]=f.get_tensor(key)
    scaling=(config.lora_alpha / (config.r ** .5 if getattr(config,"use_rslora",False) else config.r))
    out={}
    for key,parts in pairs.items():
        if set(parts)!={"A","B"}:raise ValueError(f"Missing A/B factor: {key}")
        out[key]=(parts["B"].float() @ parts["A"].float())*scaling
    return out,config

def geometry(mat,base=None):
    sv=torch.linalg.svdvals(mat)
    energy=sv.square();p=energy/energy.sum().clamp_min(1e-20)
    rank=float(torch.exp(-(p[p>0]*p[p>0].log()).sum()))
    norm=float(torch.linalg.vector_norm(mat))
    result={"frobenius":norm,"spectral":float(sv.max()),"effective_rank":rank,
            "singular_values":sv[:32].tolist()}
    if base is not None:
        result["relative_norm"]=norm/max(float(torch.linalg.vector_norm(base)),1e-12)
    return result

def compare(x,y):
    a=x.float().flatten();b=y.float().flatten()
    cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))
    distance=float((a-b).norm()/a.norm().clamp_min(1e-12))
    return {"cosine":cosine,"relative_distance":distance,
            "norm_ratio":float(b.norm()/a.norm().clamp_min(1e-12))}

def analyze_adapter_pairs(paths,checkpoint="final"):
    """paths maps conditions to adapter paths; returns descriptive module rows."""
    weights={cond:lora_matrices(path)[0] for cond,path in paths.items()}
    rows=[]
    for condition,modules in weights.items():
        for key,matrix in modules.items():
            row={"condition":condition,"module":key,"checkpoint":checkpoint,**geometry(matrix)}
            ctrl=control_for(condition) if condition not in ("canonical","control_zh") else None
            if ctrl and ctrl in weights and key in weights[ctrl]:
                row.update({"control":ctrl,**compare(weights[ctrl][key],matrix)})
            rows.append(row)
    return rows
