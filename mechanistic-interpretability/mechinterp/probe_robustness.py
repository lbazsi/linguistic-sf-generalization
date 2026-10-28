"""Condition-level robustness of development-selected behavioral ridge probes.

A fixed development-selected probe is the unit of reporting. Test examples,
rather than the 128 correlated layers/sites, form the bootstrap observations.
Resampling preserves semantic-cell grouping; no held-out labels influence
hyperparameter or layer selection.

Judge comparisons characterize *rater* disagreement conditional on the original
model responses. They cannot identify stochastic generation variability or a
ground-truth upper bound on predictability.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np


def _metrics(pred, target):
    p=np.asarray(pred,dtype=np.float64).reshape(-1)
    y=np.asarray(target,dtype=np.float64).reshape(-1)
    if len(p)!=len(y) or not len(y) or not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Invalid holdout predictions or targets")
    d=p-y
    denom=float(((y-y.mean())**2).sum())
    return {
        "n":len(y),
        "rmse":float(np.sqrt(np.mean(d*d))),
        "mae":float(np.mean(np.abs(d))),
        "r2":float(1-np.dot(d,d)/denom) if denom>1e-10 else None,
    }


def grouped_bootstrap(target, raw_pred, baseline_pred, cluster_ids, seed=42, reps=2000,
                      bounds=(0.,4.)):
    """Paired percentile CIs for R² and RMSE gain, clipped and raw.

    Conditional on the already selected model: does not account for estimation
    or model-selection uncertainty and is not a cross-family p-value.
    """
    y=np.asarray(target,dtype=np.float64)
    raw=np.asarray(raw_pred,dtype=np.float64)
    baseline=np.asarray(baseline_pred,dtype=np.float64)
    if any(len(v)!=len(y) for v in (raw,baseline,cluster_ids)):
        raise ValueError("Bootstrap input lengths differ")
    if len(y)<3 or reps<100:raise ValueError("Insufficient cases or bootstrap replicates")
    groups=defaultdict(list)
    for i,c in enumerate(cluster_ids):
        groups[str(c)].append(i)
    if len(groups)<2:raise ValueError("At least two independent semantic cells required")
    index_groups=[np.asarray(v,dtype=int) for _,v in sorted(groups.items())]
    clipped=np.clip(raw,*bounds)
    key="clipped","unclipped"
    def one(index):
        obs=y[index];constant=baseline[index]
        base=_metrics(constant,obs)
        out={}
        for name,pred in (("clipped",clipped[index]),("unclipped",raw[index])):
            cur=_metrics(pred,obs)
            out[name]={
                "r2":cur["r2"],
                "rmse":cur["rmse"],
                "rmse_improvement":base["rmse"]-cur["rmse"],
                "relative_rmse_improvement":(
                    (base["rmse"]-cur["rmse"])/base["rmse"]
                    if base["rmse"]>1e-12 else None),
            }
        return out
    observed=one(np.arange(len(y)))
    rng=np.random.default_rng(seed)
    samples={name:{metric:[] for metric in observed[name]} for name in key}
    for _ in range(reps):
        chosen=rng.integers(0,len(index_groups),size=len(index_groups))
        idx=np.concatenate([index_groups[int(g)] for g in chosen])
        estimate=one(idx)
        for name in key:
            for metric,value in estimate[name].items():
                if value is not None and math.isfinite(value):
                    samples[name][metric].append(value)
    summary={"seed":int(seed),"bootstrap_replicates":int(reps),
             "n_scenarios":len(y),"n_semantic_cells":len(index_groups),
             "inference":"Cluster bootstrap conditional on development-selected probe; no layer-wise independence assumed."}
    for name in key:
        summary[name]={}
        for metric,value in observed[name].items():
            values=samples[name][metric]
            limits=np.quantile(values,[.025,.975]) if len(values)>=reps*.8 else (None,None)
            summary[name][metric]={
                "estimate":value,
                "ci_low":float(limits[0]) if limits[0] is not None else None,
                "ci_high":float(limits[1]) if limits[1] is not None else None,
                "valid_bootstrap_draws":len(values),
            }
    return summary


def judge_diagnostics(target, raw_pred, judge_pairs, bounds=(0.,4.)):
    """Paired original judge scores on precisely the *same* held-out responses."""
    if judge_pairs is None:
        return {"status":"not_available",
                "reason":"Original per-judge score files missing; cannot quantify rater disagreement"}
    if len(judge_pairs)!=len(target):
        raise ValueError("Judge-pair length differs from holdout")
    valid=[(i,float(a),float(b)) for i,(a,b) in enumerate(judge_pairs)
           if a is not None and b is not None]
    result={"status":"complete" if len(valid)==len(target) else "partial",
            "n_paired_judges":len(valid),"n_holdout":len(target),
            "source":"Original paired judges of the same generated response",
            "limitation":"Does not measure generation-seed noise or provide a causal noise ceiling"}
    if len(valid)<3:
        result["status"]="insufficient_pairs"
        return result
    idx=np.asarray([i for i,_,_ in valid],dtype=int)
    j1=np.asarray([a for _,a,_ in valid])
    j2=np.asarray([b for _,_,b in valid])
    predictions=np.clip(np.asarray(raw_pred,dtype=np.float64)[idx],*bounds)
    result["exact_agreement"]=float(np.mean(j1==j2))
    result["mean_absolute_disagreement"]=float(np.mean(abs(j1-j2)))
    result["mean_squared_disagreement"]=float(np.mean((j1-j2)**2))
    result["judge_1_test"]= _metrics(predictions,j1)
    result["judge_2_test"]= _metrics(predictions,j2)
    if j1.std()>1e-12 and j2.std()>1e-12:
        result["judge_correlation"]=float(np.corrcoef(j1,j2)[0,1])
    else:
        result["judge_correlation"]=None
    avg=(j1+j2)/2
    source=np.asarray(target,dtype=float)[idx]
    result["max_aggregate_vs_judge_mean_difference"]=float(np.max(abs(source-avg)))
    if result["max_aggregate_vs_judge_mean_difference"]>1e-5:
        raise ValueError("Aggregate behavioral target differs from raw judge mean")
    return result


def inspect_heldout(rows, target, raw_pred, baseline_pred, seed=42, reps=2000):
    if len(rows)!=len(target):
        raise ValueError("Scenario metadata differs from test predictions")
    semantic=[r.get("semantic_cell") for r in rows]
    if any(key is None for key in semantic):
        raise ValueError("Semantic-cell keys needed for clustered inference")
    pairs=([r.get("judge_pair",(None,None)) for r in rows]
           if all("judge_pair" in r for r in rows) else None)
    summary=grouped_bootstrap(target,raw_pred,baseline_pred,semantic,seed,reps)
    summary["judges"]=judge_diagnostics(target,raw_pred,pairs)
    by_case=[{"scenario_id":r["scenario_id"],"semantic_cell":str(r["semantic_cell"]),
              "target":float(y),"prediction_raw":float(p),
              "prediction_clipped":float(np.clip(p,0.,4.)),
              "train_mean_baseline":float(b),
              "judge_1":r.get("judge_pair",(None,None))[0],
              "judge_2":r.get("judge_pair",(None,None))[1],
              } for r,y,p,b in zip(rows,target,raw_pred,baseline_pred)]
    return summary,by_case
