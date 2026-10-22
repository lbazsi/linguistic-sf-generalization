"""Estimate every linguistic activation direction in one streaming base-model pass.

Only unit directions and layer calibration scalars are serialized. Prompt/token
activations are discarded immediately after updating GPU-resident sufficient sums.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from .common import (feature_pairs,load_general,split_groups,run_dir,write_json,digest,
                     location,batches)
from .models import (load_model,unload,blocks,site_module,tensor_output,pool,encode)

@torch.inference_mode()
def prepare(cfg,family,feature,force=False):
    output=run_dir(cfg,family,"activation-steering")
    target=output/f"directions_{feature}.safetensors"
    metadata=output/f"directions_{feature}_metadata.json"
    if target.exists() and not force:
        if not metadata.exists():raise FileNotFoundError(metadata)
        existing=json.loads(metadata.read_text(encoding="utf-8"))
        if existing["feature"]!=feature or existing["family"]!=family:
            raise ValueError("Incompatible cached direction metadata")
        source=location(cfg,"features")/f"{feature}.jsonl"
        if existing["source_sha256"]!=digest(source):
            raise ValueError("Direction source data changed; regenerate with --force")
        if existing["model_revision"]!=cfg["models"][family]["revision"]:
            raise ValueError("Base revision changed; regenerate directions")
        return target
    pairs=feature_pairs(cfg,feature)
    parts=split_groups((r["id"] for r in pairs),int(cfg["seed"]))
    train=[r for r in pairs if parts[r["id"]]=="train"]
    if not train:raise ValueError("No training pairs for direction estimation")
    model,tok=load_model(cfg,family,"base")
    pooling=cfg["steering"]["directions_pooling"]
    sites=[(i,site) for i in range(len(blocks(model))) for site in cfg["runtime"]["sites"]]
    directions={}
    shuffled={}
    calibrations={}
    try:
        for batch in batches(train,int(cfg["runtime"]["batch_size"])):
            texts=[r["canonical"] for r in batch]+[r["feature_variant"] for r in batch]
            tokens=encode(tok,texts,model.device,int(cfg["runtime"]["max_length"]))
            mask=tokens["attention_mask"]
            signs=torch.tensor([1.0 if (int(r["id"])+int(cfg["seed"]))%2 else -1.0
                                for r in batch],device=model.device)
            handles=[]
            for layer,site in sites:
                def hook(_module,_inputs,output,layer=layer,site=site):
                    sample=pool(tensor_output(output).detach(),mask,pooling)
                    contrast=(sample[len(batch):]-sample[:len(batch)]).float()
                    offset=contrast.sum(0)
                    null_offset=(contrast*signs[:,None]).sum(0)
                    key=(layer,site)
                    directions[key]=offset if key not in directions else directions[key]+offset
                    shuffled[key]=null_offset if key not in shuffled else shuffled[key]+null_offset
                handles.append(site_module(model,layer,site).register_forward_hook(hook))
            try:
                model(**tokens,use_cache=False)
            finally:
                for handle in handles:handle.remove()
            del tokens
        general=load_general(cfg)[:int(cfg["runtime"]["calibration_n"])]
        for batch in batches(general,int(cfg["runtime"]["batch_size"])):
            tokens=encode(tok,[r["prompt"] for r in batch],model.device,
                          int(cfg["runtime"]["max_length"]))
            mask=tokens["attention_mask"]
            handles=[]
            for layer,site in sites:
                def hook(_module,_inputs,output,layer=layer,site=site):
                    sample=pool(tensor_output(output).detach(),mask,pooling).float()
                    value=float((sample.norm(dim=-1)/sample.shape[-1]**.5).sum().item())
                    key=(layer,site)
                    calibrations[key]=calibrations.get(key,0.)+value
                handles.append(site_module(model,layer,site).register_forward_hook(hook))
            try:
                model(**tokens,use_cache=False)
            finally:
                for handle in handles:handle.remove()
            del tokens
        saved={};norms={}
        for layer,site in sites:
            key=(layer,site)
            vector=directions[key]/len(train)
            norm=vector.norm()
            if not torch.isfinite(norm) or norm.item()<1e-8:
                raise ValueError(f"Degenerate linguistic contrast at {key}")
            name=f"{layer}__{site}"
            saved[name+"__direction"]=(vector/norm).detach().cpu().contiguous()
            control=shuffled[key]/len(train)
            if control.norm().item()<1e-8:
                raise ValueError(f"Degenerate label-shuffled direction at {key}")
            saved[name+"__shuffled"]=(control/control.norm()).detach().cpu().contiguous()
            generator=torch.Generator(device="cpu")
            generator.manual_seed(int(cfg["seed"])+1009*layer+(0 if site=="residual" else 1))
            random_vec=torch.randn(saved[name+"__direction"].shape,generator=generator)
            saved[name+"__random"]=(random_vec/random_vec.norm()).contiguous()
            saved[name+"__scale"]=torch.tensor([calibrations[key]/len(general)],dtype=torch.float32)
            norms[name]=float(norm.item())
        save_file(saved,str(target))
        write_json(metadata,{"feature":feature,"family":family,"seed":cfg["seed"],
                 "model_revision":cfg["models"][family]["revision"],
                 "source_sha256":digest(location(cfg,"features")/f"{feature}.jsonl"),
                 "train_pairs":len(train),"calibration_prompts":len(general),
                 "pooling":pooling,"direction_raw_norms":norms})
        return target
    finally:unload(model)

def load_direction_bundle(cfg,family,feature):
    path=prepare(cfg,family,feature)
    bundle=load_file(str(path))
    def get(layer,site,kind="direction"):
        key=f"{layer}__{site}"
        if kind not in ("direction","random","shuffled"):
            raise ValueError(kind)
        direction=bundle[key+"__"+kind].float()
        scale=float(bundle[key+"__scale"].item())
        return direction,scale
    return get
