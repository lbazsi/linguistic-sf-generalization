import math
import torch
from mechinterp.stats import linear_cka,svcca,rms_displacement,bh,cluster_interval

def test_identical_activations():
    x=torch.randn(80,20)
    assert abs(linear_cka(x,x)-1)<1e-5
    assert svcca(x,x)>0.99
    assert rms_displacement(x,x)==0

def test_bh_monotone():
    q=bh([.05,.001,.1,.03])
    assert all(0<=x<=1 for x in q)
    assert q[1]<=q[3]<=q[0]<=q[2]

def test_cluster_interval_constant():
    s=cluster_interval([2,2,2,2],[1,1,2,2],reps=100)
    assert s["mean"]==s["low"]==s["high"]==2
