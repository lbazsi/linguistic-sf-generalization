import torch
from torch import nn
from types import SimpleNamespace
from mechinterp.models import blocks,site_module,pool,steer,continuation_scores

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Linear(4,4,bias=False)
    def forward(self,x):
        return x+self.mlp(x)

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=SimpleNamespace(layers=nn.ModuleList([Block(),Block()]))
    def forward(self,x):
        for layer in self.model.layers:x=layer(x)
        return x

def test_layer_and_mlp_hook():
    toy=Toy()
    assert len(blocks(toy))==2
    x=torch.ones(1,3,4)
    baseline=toy(x)
    with steer(toy,0,"residual",torch.ones(4),1.,"prompt_last"):
        changed=toy(x)
    assert not torch.allclose(baseline,changed)
    assert torch.allclose(pool(changed,torch.ones(1,3),"final"),changed[:,-1])

def test_mean_pool_ignores_padding():
    x=torch.tensor([[[100.],[2.],[4.]]])
    m=torch.tensor([[0,1,1]])
    assert pool(x,m,"mean").item()==3
