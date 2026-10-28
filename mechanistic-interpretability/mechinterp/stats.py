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
    """Legacy, in-sample SVCCA is intentionally withdrawn.

    Its high sample-dimensional PCA/CCA estimates are systematically optimistic,
    including near-unity canonical correlations for independent representations.
    None of the archived values should be used for inference or publication.
    """
    return None


def heldout_svcca(x_fit, y_fit, x_test, y_test, threshold=.99, cap=32,
                   ridge=0.1):
    """Train PCA/CCA only on fit examples; evaluate canonical correlations on test.

    Fit and test must comprise independent semantic cells. Rank is capped well
    below sample size; centering, PCA, and CCA projections are fit-only.
    """
    if any(a.ndim!=2 for a in (x_fit,y_fit,x_test,y_test)):
        raise ValueError("Expected matrices")
    if x_fit.shape[0]!=y_fit.shape[0] or x_test.shape[0]!=y_test.shape[0]:
        raise ValueError("Paired sample counts differ")
    if x_fit.shape[0] < 20 or x_test.shape[0] < 20:
        return float("nan")
    def pca(a,b):
        a=a.double();b=b.double()
        mean=a.mean(0)
        ac=a-mean
        _, s, vh=torch.linalg.svd(ac,full_matrices=False)
        ratios=s.square().cumsum(0)/s.square().sum().clamp_min(1e-20)
        k=int(torch.searchsorted(ratios,threshold).item())+1
        k=max(1,min(k,cap,a.shape[0]//5,b.shape[0]//5))
        return ac@vh[:k].T,(b-mean)@vh[:k].T
    a,at=pca(x_fit,x_test);b,bt=pca(y_fit,y_test)
    def whiten(z):
        c=z.T@z/max(1,len(z)-1)
        ev,q=torch.linalg.eigh(c)
        inv=q@torch.diag((ev.clamp_min(0)+ridge)**-.5)@q.T
        return inv
    wa=whiten(a);wb=whiten(b)
    cross=(wa@a.T@b@wb)/max(1,len(a)-1)
    u,_,vh=torch.linalg.svd(cross,full_matrices=False)
    aa=at@wa@u
    bb=bt@wb@vh.T
    aa=aa-aa.mean(0);bb=bb-bb.mean(0)
    aa=aa/(aa.square().sum(0).sqrt().clamp_min(1e-10))
    bb=bb/(bb.square().sum(0).sqrt().clamp_min(1e-10))
    return float((aa*bb).sum(0).abs().clamp(0,1).mean())

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

def cluster_signflip(values,clusters,seed=42,reps=2000):
    """Two-sided cluster sign-flip p-value, with a +1 Monte Carlo correction."""
    groups={}
    for value,cluster in zip(values,clusters):
        groups.setdefault(str(cluster),[]).append(float(value))
    if not groups:return None
    cluster_means=np.array([np.mean(v) for v in groups.values()])
    cluster_weights=np.array([len(v) for v in groups.values()],dtype=float)
    observed=abs(float(np.average(cluster_means,weights=cluster_weights)))
    rng=np.random.default_rng(seed)
    exceed=0
    for _ in range(reps):
        signs=rng.choice(np.array([-1.0,1.0]),size=len(groups))
        perm=abs(float(np.average(cluster_means*signs,weights=cluster_weights)))
        exceed+=perm>=observed-1e-12
    return float((exceed+1)/(reps+1))
