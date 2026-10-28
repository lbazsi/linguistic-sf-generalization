import torch
from mechinterp.functional import stratified_sample,Energy

def test_strata_and_determinism():
    rows=[{"id":j*10+i,"category":str(j),"tradeoff":bool(i%2)}
          for j in range(8) for i in range(10)]
    chosen=stratified_sample(rows,lambda x:(x["category"],x["tradeoff"]),2,42)
    assert len(chosen)==32
    assert chosen==stratified_sample(rows,lambda x:(x["category"],x["tradeoff"]),2,42)
    assert len({r["category"] for r in chosen})==8

def test_global_rms_not_average_batch_rms():
    e=Energy()
    e.add(torch.tensor([[1.],[1.]]))
    e.add(torch.tensor([[4.]]))
    assert abs(e.rms()-(18/3)**0.5)<1e-6
    assert e.examples==3
