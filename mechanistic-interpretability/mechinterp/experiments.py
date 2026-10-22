"""GPU experiment runners. Raw activations are reduced and discarded in memory."""
from __future__ import annotations
import json
from collections import defaultdict
import torch
from .common import (FEATURES, control_for, feature_pairs, load_concepts, load_general,
                     load_scenarios, location, run_dir, scenario_prompt, split_groups,
                     write_json, write_rows, adapter_dir, provenance, model_conditions, batches)
from .models import load_model, pooled_batches, unload, blocks, tensor_output
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

def _capture_grid(cfg,family,condition,texts,checkpoint):
    """One model at a time: accumulate only final/mean pooled observations on the GPU.

    Keeps memory bounded even when two bf16 9B models cannot coexist on one A100.
    """
    from .models import site_module,pool,tensor_output,encode
    from collections import defaultdict
    model,tok=load_model(cfg,family,condition,checkpoint)
    sites=[(layer,site) for layer in range(len(blocks(model)))
           for site in cfg["runtime"]["sites"]]
    collected=defaultdict(list)
    try:
        bs=int(cfg["runtime"]["batch_size"])
        for batch in batches(texts,bs):
            encoded=encode(tok,batch,model.device,int(cfg["runtime"]["max_length"]))
            mask=encoded["attention_mask"]
            handles=[]
            for layer,site in sites:
                def hook(_module,_args,out,layer=layer,site=site):
                    tensor=tensor_output(out).detach()
                    for how in cfg["runtime"]["pooling"]:
                        collected[(layer,site,how)].append(pool(tensor,mask,how))
                handles.append(site_module(model,layer,site).register_forward_hook(hook))
            try:
                with torch.inference_mode():
                    model(**encoded,use_cache=False)
            finally:
                for h in handles:h.remove()
            del encoded
        return {key:torch.cat(parts,dim=0) for key,parts in collected.items()}
    finally:
        unload(model)

def difference(cfg,family,condition,checkpoint="final"):
    control = control_for(condition) if condition in FEATURES else "base"
    scenarios=load_scenarios(cfg)
    datasets={"welfare":[scenario_prompt(s["scenario"]) for s in scenarios],
              "general":[r["prompt"] for r in load_general(cfg)]}
    out=run_dir(cfg,family,"model-difference-analysis")
    result_file=out/f"{condition}_{checkpoint}.jsonl"
    records=[]
    for domain,texts in datasets.items():
        a=_capture_grid(cfg,family,control,texts,checkpoint)
        b=_capture_grid(cfg,family,condition,texts,checkpoint)
        if set(a)!=set(b):raise ValueError("Hook coverage differs between adapters")
        for (layer,site,pool),x in a.items():
            y=b[(layer,site,pool)]
            if x.shape != y.shape:raise ValueError("Mismatched activation shapes")
            metrics={"cka":linear_cka(x,y),"svcca":svcca(x,y,
                float(cfg["analysis"]["svcca_variance"]),int(cfg["analysis"]["svcca_components_cap"])),
                "displacement":rms_displacement(x,y)}
            records.append({"model":family,"condition":condition,"control":control,
                            "checkpoint":checkpoint,"domain":domain,"layer":layer,
                            "site":site,"pooling":pool,"n":len(texts),**metrics})
            del y
        del a,b
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write_rows(result_file,records)
    write_json(out/f"{condition}_{checkpoint}_manifest.json",
               provenance(cfg,family,"difference",[location(cfg,"scenarios"),location(cfg,"general")]))
    return result_file

def probe_dataset(cfg,task,feature=None,judgments_path=None,condition=None,family=None):
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
        if judgments_path is not None:
            source=judgments_path
        elif family=="gemma2":
            source=location(cfg,"judgments")
        else:
            specified=cfg["models"][family].get("judgments")
            if not specified:
                raise ValueError(f"Set models.{family}.judgments to its own unsteered evaluation scores")
            from pathlib import Path
            source=Path(specified)
            if not source.is_absolute():
                source=(Path(cfg["_config_file"]).resolve().parent.parent/source).resolve()
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
    """Run one model, all layers and sites, with transient minibatch activations."""
    from .streaming import fit_all
    data=probe_dataset(cfg,task,feature=feature,condition=condition,family=family)
    subsets={part:[r for r in data if r["partition"]==part]
             for part in ("train","development","test")}
    model,tok=load_model(cfg,family,condition)
    try:
        out=run_dir(cfg,family,"linear-probing")
        stem=f"{task}_{feature or 'all'}_{condition}"
        metrics=fit_all(model,tok,subsets["train"],subsets["development"],subsets["test"],
                        "regression" if task=="behavior" else "classification",cfg,
                        out/f"{stem}_weights.safetensors")
        records=[{**item,"task":task,"feature":feature,"condition":condition,
                  "family":family} for item in metrics]
        output=out/f"{stem}.jsonl"
        write_rows(output,records)
        return output
    finally:unload(model)

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
