import torch
from mechinterp.lora import LowRankUpdate,geometry,compare

def test_rank_geometry_matches_dense():
    g=torch.Generator().manual_seed(42)
    a=torch.randn(3,7,generator=g)
    b=torch.randn(9,3,generator=g)
    update=LowRankUpdate(a,b,2.0)
    dense=2*b@a
    assert torch.allclose(update.norm(),dense.norm(),rtol=1e-5)
    assert torch.allclose(update.spectrum()[:3],torch.linalg.svdvals(dense)[:3],rtol=1e-5)
    stats=compare(update,update)
    assert abs(stats["cosine"]-1)<1e-5
    assert stats["relative_distance"]<1e-5

def test_functional_application():
    a=torch.randn(2,5);b=torch.randn(6,2)
    update=LowRankUpdate(a,b,1.75)
    x=torch.randn(3,5)
    assert torch.allclose(update.apply(x),x@(1.75*b@a).T,atol=1e-5)
