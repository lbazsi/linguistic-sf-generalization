import math
import torch
from mechinterp.stats import linear_cka,svcca,rms_displacement,bh,cluster_interval

def test_identical_activations():
    x=torch.randn(80,20)
    assert abs(linear_cka(x,x)-1)<1e-5
    assert svcca(x,x) is None
    assert rms_displacement(x,x)==0

def test_bh_monotone():
    q=bh([.05,.001,.1,.03])
    assert all(0<=x<=1 for x in q)
    assert q[1]<=q[3]<=q[0]<=q[2]

def test_cluster_interval_constant():
    s=cluster_interval([2,2,2,2],[1,1,2,2],reps=100)
    assert s["mean"]==s["low"]==s["high"]==2

def test_holdout_svcca_identical_and_independent():
    from mechinterp.stats import heldout_svcca
    gen=torch.Generator().manual_seed(48)
    x=torch.randn(320,22,generator=gen)
    y=torch.randn(320,22,generator=gen)
    assert heldout_svcca(x[:200],x[:200],x[200:],x[200:])>0.99
    assert heldout_svcca(x[:200],y[:200],x[200:],y[200:])<0.45
