"""Exact effective LoRA geometry in the small rank space.

Never materializes a large dense update for spectral analysis. Uses the effective
(alpha/r) B A product, not separately non-identifiable A/B representations.
"""
from __future__ import annotations
import torch

class LowRankUpdate:
    def __init__(self,A,B,scale):
        A=A.float().contiguous()
        B=B.float().contiguous()
        if A.ndim!=2 or B.ndim!=2 or A.shape[0]!=B.shape[1]:
            raise ValueError("Inconsistent LoRA factor dimensions")
        self.A=A
        self.B=B
        self.scale=float(scale)

    def squared_norm(self):
        ba=self.B.T@self.B
        aa=self.A@self.A.T
        return (self.scale**2*torch.sum(ba*aa.T)).clamp_min(0)

    def norm(self):
        return self.squared_norm().sqrt()

    def spectrum(self):
        # Thin QR of the two factors; core is at most rank x rank.
        _,rb=torch.linalg.qr(self.B,mode="reduced")
        _,ra=torch.linalg.qr(self.A.T,mode="reduced")
        return torch.linalg.svdvals(rb@ra.T)*abs(self.scale)

    def apply(self,x):
        A=self.A.to(x.device,dtype=x.dtype)
        B=self.B.to(x.device,dtype=x.dtype)
        return ((x@A.T)@B.T)*self.scale

    def dot(self,other):
        if self.B.shape[0]!=other.B.shape[0] or self.A.shape[1]!=other.A.shape[1]:
            raise ValueError("Cannot compare different projection shapes")
        ba=self.B.T@other.B
        aa=other.A@self.A.T
        return torch.sum(ba*aa.T)*self.scale*other.scale

    def dense_for_test(self):
        return self.scale*(self.B@self.A)


def lora_matrices(adapter_path):
    """Return effective *factorized* update objects and PEFT configuration."""
    from pathlib import Path
    from safetensors import safe_open
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
        if set(parts)!={"A","B"}:
            raise ValueError(f"Missing A/B factor: {key}")
        out[key]=LowRankUpdate(parts["A"],parts["B"],scaling)
    return out,config


def geometry(update,base=None):
    sv=update.spectrum()
    energy=sv.square()
    p=energy/energy.sum().clamp_min(1e-20)
    rank=float(torch.exp(-(p[p>0]*p[p>0].log()).sum()))
    norm=float(update.norm())
    result={"frobenius":norm,"spectral":float(sv.max()),"effective_rank":rank,
            "singular_values":sv[:32].tolist()}
    if base is not None:
        result["relative_norm"]=norm/max(float(torch.linalg.vector_norm(base)),1e-12)
    return result


def compare(x,y):
    xx=float(x.squared_norm())
    yy=float(y.squared_norm())
    xy=float(x.dot(y))
    denominator=max((xx*yy)**.5,1e-12)
    difference=max(xx+yy-2*xy,0)**.5
    return {"cosine":max(-1,min(1,xy/denominator)),
            "relative_distance":difference/max(xx**.5,1e-12),
            "norm_ratio":yy**.5/max(xx**.5,1e-12)}


def analyze_adapter_pairs(paths,checkpoint="final"):
    weights={cond:lora_matrices(path)[0] for cond,path in paths.items()}
    from .common import control_for,FEATURES
    rows=[]
    for condition,modules in weights.items():
        for key,update in modules.items():
            row={"condition":condition,"module":key,"checkpoint":checkpoint,
                 **geometry(update)}
            if condition in FEATURES:
                ctrl=control_for(condition)
                if ctrl in weights and key in weights[ctrl]:
                    row.update({"control":ctrl,**compare(weights[ctrl][key],update)})
            rows.append(row)
    return rows
