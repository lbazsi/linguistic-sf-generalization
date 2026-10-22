"""GPU-safe descriptive statistics and clustered paired inference."""
import math
import numpy as np
import torch

def linear_cka(x, y):
    x, y = x.float() - x.float().mean(0), y.float() - y.float().mean(0)
    xx, yy = x @ x.T, y @ y.T
    numerator = torch.sum(xx * yy)
    denom = (torch.sum(xx * xx) * torch.sum(yy * yy)).sqrt().clamp_min(1e-20)
    return float((numerator / denom).item())

def svcca(x, y, threshold=.99, cap=256):
    """Regularized SVCCA using a shared heldout set and PCA derived from fit sets."""
    x, y = x.double()-x.double().mean(0), y.double()-y.double().mean(0)
    def reduce(z):
        _, s, vh = torch.linalg.svd(z, full_matrices=False)
        variance = s.square()
        retain = int(torch.searchsorted(variance.cumsum(0) / variance.sum().clamp_min(1e-20),
                                       threshold).item()) + 1
        k = max(1, min(retain, cap, z.shape[0]-2))
        return z @ vh[:k].T
    a,b = reduce(x),reduce(y)
    def whiten(z):
        q, s, _ = torch.linalg.svd(z, full_matrices=False)
        valid = s > s.max().clamp_min(1e-10)*1e-6
        return q[:, valid]
    qa,qb=whiten(a),whiten(b)
    if qa.numel()==0 or qb.numel()==0: return float("nan")
    return float(torch.linalg.svdvals(qa.T @ qb).mean().clamp(0,1).item())

def rms_displacement(x,y):
    num=(x.float()-y.float()).square().mean().sqrt()
    den=x.float().square().mean().sqrt().clamp_min(1e-9)
    return float((num/den).item())

def bh(pvalues):
    p=np.asarray(pvalues,dtype=float)
    order=np.argsort(p); q=np.empty_like(p)
    ranks=np.arange(1,len(p)+1)
    q[order]=np.minimum.accumulate((p[order]*len(p)/ranks)[::-1])[::-1].clip(0,1)
    return q.tolist()

def cluster_interval(values,clusters,seed=42,reps=2000):
    groups={}
    for v,k in zip(values,clusters):
        groups.setdefault(str(k),[]).append(float(v))
    if not groups: return {"mean":None,"low":None,"high":None}
    rng=np.random.default_rng(seed)
    keys=list(groups)
    observed=float(np.mean([v for vs in groups.values() for v in vs]))
    boots=[]
    for _ in range(reps):
        chosen=rng.choice(len(keys),size=len(keys),replace=True)
        vals=[v for i in chosen for v in groups[keys[int(i)]]]
        boots.append(np.mean(vals))
    lo,hi=np.quantile(boots,[.025,.975])
    return {"mean":observed,"low":float(lo),"high":float(hi),"clusters":len(keys)}

def paired_summary(rows,key="delta",cluster="cell",seed=42,reps=2000):
    data=[r for r in rows if r.get(key) is not None]
    return cluster_interval([r[key] for r in data],[r[cluster] for r in data],seed,reps)
