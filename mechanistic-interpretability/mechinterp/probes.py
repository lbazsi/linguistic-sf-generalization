"""Fixed GPU-resident streaming linear probes, trained without activation caches."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .models import pooled_batches
from .common import split_groups, batches

class OnlineProbe:
    def __init__(self, width, task, device="cuda", lr=.01, wd=.001, mean=None, std=None):
        self.task=task
        self.mean=mean
        self.std=std
        self.w=torch.nn.Parameter(torch.zeros(width,device=device,dtype=torch.float32))
        self.b=torch.nn.Parameter(torch.zeros((),device=device,dtype=torch.float32))
        self.optimizer=torch.optim.AdamW((self.w,self.b),lr=lr,weight_decay=wd)

    def predict(self,x):
        z=(x.float()-self.mean)/self.std
        return z @ self.w + self.b

    def update(self,x,y):
        self.optimizer.zero_grad(set_to_none=True)
        p=self.predict(x)
        loss=F.binary_cross_entropy_with_logits(p,y.float()) if self.task=="classification" else F.mse_loss(p,y.float())
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self,model,tok,rows,layer,site,pool,batch,max_length):
        preds=[]; ys=[]
        for part in batches(rows,batch):
            x=next(pooled_batches(model,tok,[r["text"] for r in part],layer,site,pool,len(part),max_length))
            z=self.predict(x)
            preds.extend(z.cpu().tolist());ys.extend(float(r["label"]) for r in part)
        if not preds: return {}
        y=torch.tensor(ys);p=torch.tensor(preds)
        if self.task=="classification":
            acc=float(((p>=0).float()==y).float().mean())
            # AUC as Mann-Whitney rank statistic; meaningful only when both labels occur.
            pos=p[y==1];neg=p[y==0]
            auc=float(((pos[:,None]>neg).float()+.5*(pos[:,None]==neg)).mean()) if len(pos) and len(neg) else None
            return {"balanced_accuracy":acc,"roc_auc":auc,"n":len(y)}
        err=p-y
        return {"rmse":float(err.square().mean().sqrt()),"mae":float(err.abs().mean()),
                "r2":float(1-err.square().sum()/(y-y.mean()).square().sum().clamp_min(1e-8)),"n":len(y)}

@torch.inference_mode()
def feature_statistics(model,tok,rows,layer,site,pooling,cfg):
    """Training-partition-only streaming standardization in GPU memory."""
    n=0;total=None;squares=None
    rt=cfg["runtime"]
    for part in batches(rows,int(rt["batch_size"])):
        x=next(pooled_batches(model,tok,[r["text"] for r in part],layer,site,pooling,len(part),rt["max_length"])).float()
        a=x.sum(0);b=x.square().sum(0)
        total=a if total is None else total+a
        squares=b if squares is None else squares+b
        n+=len(x)
    if n<2:raise ValueError("Insufficient training observations")
    mean=total/n
    variance=(squares/n-mean.square()).clamp_min(1e-6)
    return mean,variance.sqrt()

def train_probe(model,tok,train,development,test,layer,site,pooling,task,cfg):
    """Fixed streaming logistic/ridge predictors, standardized using training data."""
    runtime=cfg["runtime"];pcfg=cfg["probing"]
    mean,std=feature_statistics(model,tok,train,layer,site,pooling,cfg)
    probe=OnlineProbe(mean.numel(),task,mean.device,pcfg["learning_rate"],pcfg["weight_decay"],mean,std)
    for epoch in range(int(pcfg["epochs"])):
        for part in batches(train,int(runtime["batch_size"])):
            x=next(pooled_batches(model,tok,[r["text"] for r in part],layer,site,pooling,
                                  len(part),int(runtime["max_length"])))
            labels=torch.tensor([float(r["label"]) for r in part],device=x.device)
            probe.update(x,labels)
    dev=probe.evaluate(model,tok,development,layer,site,pooling,runtime["batch_size"],runtime["max_length"])
    held=probe.evaluate(model,tok,test,layer,site,pooling,runtime["batch_size"],runtime["max_length"])
    return {"validation":dev,"test":held,
            "weights":probe.w.detach().cpu().tolist(),"bias":float(probe.b.item()),
            "epochs":pcfg["epochs"]}
