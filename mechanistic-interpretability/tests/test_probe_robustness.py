import numpy as np
import pytest

from mechinterp.probe_robustness import grouped_bootstrap, judge_diagnostics, inspect_heldout


def test_bootstrap_recovers_improvement_with_shared_cell_replicates():
    y=np.array([1.,2.,3.,4.]*15)
    pred=y+0.1
    base=np.ones_like(y)*y.mean()
    clusters=[f"cell_{i//2}" for i in range(len(y))]
    result=grouped_bootstrap(y,pred,base,clusters,seed=15,reps=300)
    assert result["n_scenarios"]==60
    assert result["n_semantic_cells"]==30
    assert result["clipped"]["r2"]["estimate"]>.9
    assert result["unclipped"]["rmse_improvement"]["ci_low"]>0
    assert result["clipped"]["relative_rmse_improvement"]["ci_low"]>0


def test_clipping_is_a_reported_sensitivity_not_hidden():
    y=np.array([1.,2.,3.,4.]*15)
    pred=y+2.0
    base=np.full_like(y,2.5)
    clusters=list(range(len(y)))
    result=grouped_bootstrap(y,pred,base,clusters,reps=100)
    assert result["clipped"]["rmse"]["estimate"]<result["unclipped"]["rmse"]["estimate"]


def test_two_judges_and_means_are_checked():
    target=[1.5,2.5,3.5,2.]
    pairs=[(1,2),(2,3),(3,4),(2,2)]
    result=judge_diagnostics(target,[1.2,2.2,3.2,2.],pairs)
    assert result["n_paired_judges"]==4
    assert result["mean_absolute_disagreement"]==.75
    with pytest.raises(ValueError,match="Aggregate behavioral target"):
        judge_diagnostics([1.,2.,3.,4.],[1.,2.,3.,4.],pairs)


def test_predictions_and_clusters_reported_without_selection_leakage():
    rows=[{"scenario_id":i,"semantic_cell":f"c_{i//2}","judge_pair":(2,2)}
          for i in range(40)]
    summary, cases=inspect_heldout(rows,[2.]*40,[2.]*40,[2.]*40,reps=100)
    assert summary["judges"]["status"]=="complete"
    assert summary["clipped"]["r2"]["estimate"] is None
    assert len(cases)==40
