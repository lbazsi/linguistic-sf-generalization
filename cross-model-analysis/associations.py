"""Exploratory within-family feature-to-mechanism associations.

Correlation is computed only on matched feature-level measurements. The pooled
analysis is deliberately not fitted: the four family samples are dependent and
mechanistic metric scales differ across architectures.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from cross_model import CROSS_LANGUAGE, FAMILIES, FEATURES, average, number, pearson, ranks, read_csv, write_csv, write_json

def run_associations(dest:Path):
    behavior=dest/"behavioral"/"per_family_effects.csv"
    mechanism=dest/"mechanistic"/"comparable_profiles.csv"
    if not (behavior.exists() and mechanism.exists()):
        return {"status":"skipped","reason":"Behavioral and mechanistic summary tables both required"}
    b={(r["model"],r["feature"]):number(r["mean"]) for r in read_csv(behavior)}
    profiles=read_csv(mechanism)
    groups=defaultdict(lambda:defaultdict(list))
    # Predeclared site, pooling, domain and checkpoint stratification.
    for row in profiles:
        if row.get("feature") not in FEATURES:continue
        signature=tuple(row.get(k) for k in
                        ("stage","metric","domain","site","pooling","checkpoint","condition","policy"))
        for model in FAMILIES:
            val=number(row.get(model))
            if val is not None:
                groups[(model,signature)][row["feature"]].append(val)
    results=[]
    for (model,signature),feature_values in sorted(groups.items(), key=lambda x:str(x[0])):
        effects=[]
        for scope,chosen in (("within_english",[f for f in FEATURES if f not in CROSS_LANGUAGE]),
                             ("all_features",list(FEATURES))):
            observed=[(b.get((model,f)),average(feature_values.get(f,[]))) for f in chosen]
            observed=[(x,y) for x,y in observed if x is not None and y is not None]
            x,y=([r[0] for r in observed],[r[1] for r in observed]) if observed else ([],[])
            r=pearson(x,y)
            loo=[pearson(x[:i]+x[i+1:],y[:i]+y[i+1:]) for i in range(len(x))]
            loo=[v for v in loo if v is not None]
            stage,metric,domain,site,pooling,checkpoint,condition,policy=signature
            results.append({"model":model,"scope":scope,"stage":stage,"metric":metric,
                            "domain":domain,"site":site,"pooling":pooling,
                            "checkpoint":checkpoint,"condition":condition,"policy":policy,
                            "n_features":len(observed),"pearson_r":r,
                            "spearman_rho":pearson(ranks(x),ranks(y)) if len(x)>=3 else None,
                            "leave_one_out_r_min":min(loo) if loo else None,
                            "leave_one_out_r_max":max(loo) if loo else None,
                            "exploratory":True})
    write_csv(dest/"cross_stage_associations.csv",results)
    summary={"status":"complete","strata":len(results),
             "note":"Exploratory; no multiple-testing claim, causal claim, or cross-architecture pooling."}
    write_json(dest/"cross_stage_summary.json",summary)
    return summary
