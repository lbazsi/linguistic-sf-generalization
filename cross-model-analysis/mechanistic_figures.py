"""Mechanistic and cross-stage supplementary figures; no inferred missing measurements."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figures import MODEL_NAMES, COLORS, FEATURE_ORDER, label
from cross_model import FAMILIES, CROSS_LANGUAGE, number, read_csv

def plot_depth(pub, root, stage, metric, title, name, filter_domain=None):
    src=root/"mechanistic"/"normalized_metrics.csv"
    if not src.is_file():
        pub.skipped(name,"No mechanistic metrics table");return
    rows=[r for r in read_csv(src) if r.get("stage")==stage and r.get("metric")==metric
          and (filter_domain is None or r.get("domain")==filter_domain)]
    if not rows:
        pub.skipped(name,f"No {stage}/{metric} data");return
    fig,axes=plt.subplots(2,2,figsize=(11,7.9),sharex=True,sharey=True)
    plotted=[]
    for m,ax in zip(FAMILIES,axes.flat):
        selected=[r for r in rows if r["model"]==m and r.get("layer") not in (None,"")]
        if not selected:
            ax.set_title(MODEL_NAMES[m]+" (unavailable)")
            continue
        # Layer numbers normalized by *observed* layer count; do not align neurons.
        layers=sorted({int(r["layer"]) for r in selected})
        high=max(layers)
        per=defaultdict(list)
        for r in selected:
            v=number(r.get("value"))
            if v is not None:per[int(r["layer"])].append(v)
        if not per:continue
        x=[l/max(1,high) for l in layers if l in per]
        y=[float(np.mean(per[l])) for l in layers if l in per]
        ax.plot(x,y,lw=1.9,marker=".",color=COLORS[m])
        for l,xx,v in zip((l for l in layers if l in per),x,y):
            plotted.append({"model":m,"layer":l,"relative_layer_depth":xx,
                            "mean_metric":v,"n_records":len(per[l]),
                            "stage":stage,"metric":metric,"domain":filter_domain})
        ax.set_title(MODEL_NAMES[m]);ax.grid(alpha=.4)
    for ax in axes[-1]:ax.set_xlabel("Normalized layer depth (0–1)")
    for ax in axes[:,0]:ax.set_ylabel(metric)
    fig.suptitle(title,x=.09,ha="left",fontsize=13)
    fig.subplots_adjust(left=.11,right=.98,top=.88,bottom=.10,hspace=.35,wspace=.17)
    pub.publish(name,fig,plotted,title,
                ["mechanistic/normalized_metrics.csv"],
                "Descriptive averages across available conditions/sites/pooling; raw values remain in exported source tables. Do not interpret averages as matched model coordinates.",
                family="mechanistic")

def plot_mechanistic_heatmap(pub,root,stage,metric,name,title):
    src=root/"mechanistic"/"comparable_profiles.csv"
    if not src.is_file():pub.skipped(name,"No comparable profiles");return
    rows=[r for r in read_csv(src) if r.get("stage")==stage and r.get("metric")==metric
          and r.get("feature") in FEATURE_ORDER]
    if not rows:pub.skipped(name,f"No {stage}/{metric} profiles");return
    arr=np.full((len(FEATURE_ORDER),len(FAMILIES)),np.nan)
    plotted=[]
    for i,f in enumerate(FEATURE_ORDER):
        match=[r for r in rows if r["feature"]==f]
        for j,m in enumerate(FAMILIES):
            vals=[number(r.get(m)) for r in match]
            vals=[v for v in vals if v is not None]
            if vals:
                arr[i,j]=float(np.mean(vals))
                plotted.append({"feature":f,"model":m,"mean_metric":arr[i,j],
                                "n_strata":len(vals),"stage":stage,"metric":metric})
    if not np.isfinite(arr).any():pub.skipped(name,"No finite metric values");return
    fig,ax=plt.subplots(figsize=(8.1,8.0))
    from matplotlib.colors import Normalize
    cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#eeeeee")
    im=ax.imshow(np.ma.masked_invalid(arr),aspect="auto",interpolation="nearest",cmap=cmap)
    ax.set_xticks(range(4),list(MODEL_NAMES.values()),rotation=35,ha="right")
    ax.set_yticks(range(len(FEATURE_ORDER)),[label(f)+(" †" if f in CROSS_LANGUAGE else "") for f in FEATURE_ORDER])
    ax.axhline(sum(f not in CROSS_LANGUAGE for f in FEATURE_ORDER)-.5,color="white",lw=1.4)
    ax.set_title(title,loc="left",pad=12)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.04);cb.set_label(metric)
    fig.subplots_adjust(left=.33,right=.91,top=.93,bottom=.12)
    pub.publish(name,fig,plotted,title,
                ["mechanistic/comparable_profiles.csv"],
                "Feature-level averages are descriptive; mechanism availability and layer/site coverage can differ. †: language-changing contrasts.",
                family="mechanistic")

def plot_cross_stage(pub,root):
    file=root/"cross_stage_associations.csv"
    if not file.is_file():pub.skipped("fig17_behavior_mechanism","Cross-stage associations unavailable");return
    rows=[r for r in read_csv(file) if r.get("scope")=="within_english"
          and number(r.get("pearson_r")) is not None
          and number(r.get("n_features")) is not None
          and number(r.get("n_features"))>=8]
    if not rows:pub.skipped("fig17_behavior_mechanism","Fewer than eight matched within-English features");return
    stages=sorted({r["stage"]+"/"+r["metric"] for r in rows})
    fig,ax=plt.subplots(figsize=(max(8,len(stages)*.95),5.1))
    plotted=[]
    for j,metric in enumerate(stages):
        for i,m in enumerate(FAMILIES):
            vals=[number(r["pearson_r"]) for r in rows
                  if r["model"]==m and r["stage"]+"/"+r["metric"]==metric]
            vals=[v for v in vals if v is not None]
            if not vals:continue
            # Distribution across explicitly different strata, not an uncertainty interval.
            for v in vals:
                ax.scatter(j+(i-1.5)*.13,v,color=COLORS[m],s=13,alpha=.34)
                plotted.append({"model":m,"metric":metric,"pearson_r":v})
            ax.plot(j+(i-1.5)*.13,float(np.median(vals)),marker="_",color=COLORS[m],ms=13,mew=2)
    ax.axhline(0,color="#333",lw=.85,ls="--")
    ax.set_xticks(range(len(stages)),stages,rotation=50,ha="right")
    ax.set_ylim(-1.03,1.03)
    ax.set_ylabel("Exploratory feature-level Pearson r")
    ax.set_title("Behavior–mechanism associations across measurement strata",loc="left")
    ax.grid(axis="y",alpha=.35)
    fig.subplots_adjust(left=.11,right=.97,top=.88,bottom=.34)
    pub.publish("fig17_behavior_mechanism",fig,plotted,
                "Associations between welfare-support effects and comparable scalar mechanistic measurements.",
                ["cross_stage_associations.csv"],
                "Exploratory; points represent differing sites, domains and checkpoint strata, not independent replicates. No causal mediation or corrected significance claim.",
                family="mechanistic")

def generate_mechanistic(pub,root):
    plot_depth(pub,root,"representations","cka",
               "Representation similarity over network depth (welfare prompts)",
               "fig12_cka_depth","welfare")
    pub.skipped("fig13_svcca_depth",
                "Legacy in-sample SVCCA invalid; requires independent fit/evaluation rerun")
    plot_depth(pub,root,"probes","linguistic_roc_auc",
               "Held-out linguistic-feature decodability over depth",
               "fig14_linguistic_probes")
    plot_depth(pub,root,"probes","behavior_r2",
               "Held-out welfare-behavior linear-probe performance",
               "fig15_behavior_probes")
    plot_mechanistic_heatmap(pub,root,"lora","relative_distance",
                            "fig16_lora_geometry","LoRA update distance versus registered controls")
    plot_mechanistic_heatmap(pub,root,"steering_heldout",
                            "judged_target_value_support_delta",
                            "fig18_steering_judged",
                            "Independently judged held-out steering effects")
    plot_cross_stage(pub,root)
