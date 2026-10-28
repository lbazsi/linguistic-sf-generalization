"""All-layer streaming linear probing with no activation dump.

Each minibatch is forwarded once through the frozen model. Hooks yield final/mean
pooled vectors on GPU to a bank of independent regularized linear predictors.
The bank is trained jointly for efficiency, with no nonlinear probe layers.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import torch
from safetensors.torch import save_file
from .common import batches
from .models import blocks,site_module,tensor_output,pool,encode

@contextmanager
def pooled_hooks(model,mask,keys):
    gathered={}
    handles=[]
    for layer,site in sorted(set((l,s) for l,s,_ in keys)):
        def hook(_module,_args,output,layer=layer,site=site):
            h=tensor_output(output).detach()
            for how in ("final","mean"):
                key=(layer,site,how)
                if key in keys:
                    gathered[key]=pool(h,mask,how).float()
        handles.append(site_module(model,layer,site).register_forward_hook(hook))
    try:
        yield gathered
    finally:
        for handle in handles:handle.remove()

@torch.inference_mode()
def batch_grid(model,tok,rows,cfg,keys):
    text=[r["text"] for r in rows]
    tokens=encode(tok,text,model.device,cfg["runtime"]["max_length"])
    with pooled_hooks(model,tokens["attention_mask"],keys) as gathered:
        model(**tokens,use_cache=False)
    if set(gathered)!=set(keys):
        raise ValueError("Missing requested activation hooks")
    return torch.stack([gathered[k] for k in keys],dim=0)

def model_keys(model,cfg):
    return [(i,site,how) for i in range(len(blocks(model)))
            for site in cfg["runtime"]["sites"] for how in cfg["runtime"]["pooling"]]

@torch.inference_mode()
def train_statistics(model,tok,train,cfg,keys):
    total=squared=None;n=0
    for batch in batches(train,int(cfg["runtime"]["batch_size"])):
        h=batch_grid(model,tok,batch,cfg,keys)
        ss=h.sum(1);sq=h.square().sum(1)
        total=ss if total is None else total+ss
        squared=sq if squared is None else squared+sq
        n+=h.shape[1]
    if n<=1:raise ValueError("Not enough observations to standardize probes")
    mean=total/n
    std=(squared/n-mean.square()).clamp_min(1e-6).sqrt()
    return mean,std

class ProbeBank(torch.nn.Module):
    def __init__(self,mean,std,task,lr,wd):
        super().__init__()
        self.register_buffer("mean",mean)
        self.register_buffer("std",std)
        self.weights=torch.nn.Parameter(torch.zeros_like(mean))
        self.bias=torch.nn.Parameter(torch.zeros(mean.shape[0],device=mean.device))
        self.task=task
        self.optim=torch.optim.AdamW([self.weights,self.bias],lr=lr,weight_decay=wd)

    def forward(self,x):
        z=(x.float()-self.mean[:,None,:])/self.std[:,None,:]
        return (z*self.weights[:,None,:]).sum(-1)+self.bias[:,None]

    def train_batch(self,x,y):
        self.optim.zero_grad(set_to_none=True)
        p=self(x)
        if self.task=="classification":
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                p,y.float()[None,:].expand_as(p))
        else:
            loss=torch.nn.functional.mse_loss(p,y.float()[None,:].expand_as(p))
        loss.backward()
        self.optim.step()
        return float(loss.detach().item())

@torch.inference_mode()
def evaluate_bank(bank,model,tok,rows,cfg,keys):
    observations=[];targets=[]
    for batch in batches(rows,int(cfg["runtime"]["batch_size"])):
        h=batch_grid(model,tok,batch,cfg,keys)
        observations.append(bank(h).detach().cpu())
        targets.extend(float(r["label"]) for r in batch)
    if not observations:return []
    preds=torch.cat(observations,dim=1)
    target=torch.tensor(targets)
    results=[]
    for p in preds:
        if bank.task=="classification":
            positive=p[target==1]
            negative=p[target==0]
            tpr=float((positive>=0).float().mean()) if len(positive) else None
            tnr=float((negative<0).float().mean()) if len(negative) else None
            acc=(tpr+tnr)/2 if tpr is not None and tnr is not None else None
            auc=float(((positive[:,None]>negative).float()+
                       .5*(positive[:,None]==negative)).mean()) if len(positive) and len(negative) else None
            results.append({"balanced_accuracy":acc,"roc_auc":auc,"n":len(target)})
        else:
            err=p-target
            total=(target-target.mean()).square().sum()
            results.append({"rmse":float(err.square().mean().sqrt()),
                "mae":float(err.abs().mean()),
                "r2":float(1-err.square().sum()/total.clamp_min(1e-8)),
                "n":len(target)})
    return results

def fit_all(model,tok,train,development,test,task,cfg,weights_path):
    if task == "regression":
        from .ridge import fit_behavior
        return fit_behavior(model,tok,train,development,test,cfg,weights_path)
    keys=model_keys(model,cfg)
    mean,std=train_statistics(model,tok,train,cfg,keys)
    pcfg=cfg["probing"]
    bank=ProbeBank(mean,std,task,float(pcfg["learning_rate"]),float(pcfg["weight_decay"]))
    torch.manual_seed(int(cfg["seed"]))
    for epoch in range(int(pcfg["epochs"])):
        for batch in batches(train,int(cfg["runtime"]["batch_size"])):
            h=batch_grid(model,tok,batch,cfg,keys).clone()  # normal tensor for probe gradients
            y=torch.tensor([float(r["label"]) for r in batch],device=h.device)
            bank.train_batch(h,y)
    dev=evaluate_bank(bank,model,tok,development,cfg,keys)
    held=evaluate_bank(bank,model,tok,test,cfg,keys)
    weights_path.parent.mkdir(parents=True,exist_ok=True)
    save_file({"weights":bank.weights.detach().cpu().contiguous(),
               "bias":bank.bias.detach().cpu().contiguous(),
               "mean":bank.mean.detach().cpu().contiguous(),
               "std":bank.std.detach().cpu().contiguous()},
              str(weights_path))
    return [{"layer":l,"site":s,"pooling":pooling,
             "development":dev[i],"test":held[i]}
            for i,(l,s,pooling) in enumerate(keys)]
