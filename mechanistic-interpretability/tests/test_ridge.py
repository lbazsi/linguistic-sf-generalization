import torch
from mechinterp.ridge import fit_ridge_activations, scores

def test_ridge_recovers_holdout_signal_and_beats_mean():
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(140, 18, generator=gen)
    beta = torch.randn(18, generator=gen)
    y = (x @ beta) / 5 + 2
    y = y.clamp(0, 4)
    inputs = (x[:80].unsqueeze(0), x[80:110].unsqueeze(0), x[110:].unsqueeze(0))
    rows, weights = fit_ridge_activations([(0, "residual", "final")],
                                           *inputs, y[:80], y[80:110], y[110:])
    assert rows[0]["test"]["r2"] > 0.4
    assert rows[0]["test"]["rmse"] < rows[0]["baseline"]["test"]["rmse"]
    assert rows[0]["selected_penalty"] is not None
    assert weights["weights"].shape == (1, 18)

def test_constant_label_uses_baseline_without_invalid_r2():
    x=torch.randn(1, 25, 12)
    y=torch.ones(25)*3
    rows,_=fit_ridge_activations([(0,"mlp","mean")],x[:,:15],x[:,15:20],x[:,20:],
                                  y[:15],y[15:20],y[20:])
    assert rows[0]["selected_baseline"]
    assert rows[0]["test"]["rmse"] == 0
    assert rows[0]["test"]["r2"] is None

def test_no_test_label_leakage_in_tuning():
    x = torch.randn(1, 80, 10, generator=torch.Generator().manual_seed(55))
    y = 2 + x[0] @ torch.arange(10).float() / 30
    args = ([(0,"residual","final")],x[:,:40],x[:,40:60],x[:,60:],
            y[:40],y[40:60])
    a,_=fit_ridge_activations(*args,y[60:])
    b,_=fit_ridge_activations(*args,4-y[60:])
    assert a[0]["selected_penalty"]==b[0]["selected_penalty"]
    assert a[0]["development"]==b[0]["development"]
    assert scores(torch.ones(3),torch.zeros(3))["r2"] is None
