import json
import torch
from mechinterp.probe_audit import evaluate_saved, selected_probe

def test_reconstruct_saved_probe_forward():
    h=torch.tensor([[1.,2.],[3.,4.]])
    saved={"mean":torch.tensor([[1.,1.]]),"std":torch.tensor([[2.,1.]]),
           "weights":torch.tensor([[3.,4.]]),"bias":torch.tensor([.5])}
    got=evaluate_saved(h,saved,0)
    assert torch.allclose(got,torch.tensor([4.5,15.5]))

def test_select_development_not_test(tmp_path):
    p=tmp_path/"scores.jsonl"
    records=[
        {"method":"dual_ridge_v2","layer":0,"site":"mlp","pooling":"final",
         "development":{"rmse":.2},"test":{"rmse":5.}},
        {"method":"dual_ridge_v2","layer":1,"site":"mlp","pooling":"final",
         "development":{"rmse":.4},"test":{"rmse":.01}},
    ]
    p.write_text("".join(json.dumps(row) + chr(10) for row in records))
    _,i=selected_probe(p)
    assert i==0
