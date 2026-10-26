"""Publication figures from audited cross-model scalar tables.

The generated source-data CSV beside each figure is the exact plotted dataset.
Per-family confidence intervals and q-values are inherited from the original
semantic-cell-cluster analysis, not recomputed from independent models.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from cross_model import FAMILIES, FEATURES, CROSS_LANGUAGE, read_csv, number, write_csv

MODEL_NAMES = {"gemma":"Gemma 2", "llama31":"Llama 3.1",
               "mistral":"Mistral", "qwen3":"Qwen 3"}
COLORS = {"gemma":"#275d88", "llama31":"#c47732",
          "mistral":"#41917b", "qwen3":"#986ba6"}
ENGLISH = [f for f in FEATURES if f not in CROSS_LANGUAGE]
OTHER = [f for f in FEATURES if f in CROSS_LANGUAGE]
FEATURE_ORDER = ENGLISH + OTHER
CATEGORIES = ("id","animal","value","context","animal_value",
              "animal_context","value_context","animal_value_context")

def label(name):
    return str(name).replace("_", " ").capitalize()

def style():
    return {
        "font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":11,
        "axes.labelsize":9, "axes.spines.top":False,
        "axes.spines.right":False, "xtick.labelsize":8,
        "ytick.labelsize":8, "legend.fontsize":8,
        "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
        "savefig.facecolor":"white", "figure.facecolor":"white",
        "axes.facecolor":"white", "axes.edgecolor":"#474747",
        "text.color":"#202020", "axes.labelcolor":"#202020",
        "xtick.color":"#383838", "ytick.color":"#383838",
        "grid.color":"#dcdcdc", "grid.linewidth":.5,
    }

class FigurePublisher:
    def __init__(self, output: Path, dpi=600):
        self.root = output / "figures"
        self.dpi = int(dpi)
        self.entries = []
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, name, fig, rows, caption, sources, note="", family="behavioral"):
        if not rows:
            plt.close(fig)
            return
        folder = self.root / family
        folder.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf","svg","png"):
            fig.savefig(folder / (name+"."+extension), format=extension,
                        dpi=self.dpi if extension=="png" else None,
                        bbox_inches="tight", pad_inches=.14)
        write_csv(folder / (name+".csv"), rows)
        plt.close(fig)
        self.entries.append({
            "figure":name, "family":family, "caption":caption, "interpretive_note":note,
            "source_files":sources, "data_rows":len(rows),
            "exports":[str((folder/(name+"."+ext)).relative_to(self.root.parent))
                       for ext in ("pdf","svg","png","csv")],
        })

    def skipped(self, name, reason, family="mechanistic"):
        self.entries.append({"figure":name,"family":family,"status":"not_generated",
                             "reason":reason})

def centered_heatmap(ax, matrix, row_labels, col_labels, title, vlim, q=None, xlabel=None):
    cmap=plt.get_cmap("RdBu").copy()
    cmap.set_bad("#ededed")
    norm=TwoSlopeNorm(vmin=-vlim,vcenter=0,vmax=vlim)
    im=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",
                 interpolation="nearest",cmap=cmap,norm=norm)
    ax.set_title(title,pad=10)
    ax.set_xticks(range(len(col_labels)),col_labels,rotation=42,ha="right")
    ax.set_yticks(range(len(row_labels)),[label(f) for f in row_labels])
    ax.set_xlim(-.5,len(col_labels)-.5)
    ax.set_ylim(len(row_labels)-.5,-.5)
    ax.set_xticks(np.arange(-.5,len(col_labels),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(row_labels),1),minor=True)
    ax.grid(which="minor",color="white",linewidth=.6)
    ax.tick_params(which="minor",bottom=False,left=False)
    if q is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i,j]) and np.isfinite(q[i,j]) and q[i,j]<.05:
                    ax.text(j,i,"*",ha="center",va="center",fontsize=10,fontweight="bold",
                            color="white" if abs(matrix[i,j])>.55*vlim else "#1d1d1d")
    if xlabel: ax.set_xlabel(xlabel)
    return im

def array_by_model(data, metric="mean", qkey="q_value", columns=CATEGORIES,
                   row_features=FEATURE_ORDER):
    lookup={(r["feature"],r.get("category",r.get("heldout_dimensions"))):r for r in data}
    arr=np.full((len(row_features),len(columns)),np.nan)
    q=np.full_like(arr,np.nan)
    for i,f in enumerate(row_features):
        for j,c in enumerate(columns):
            r=lookup.get((f,str(c)),lookup.get((f,c)))
            if r is None:continue
            v=number(r.get(metric))
            if v is not None:arr[i,j]=v
            v=number(r.get(qkey))
            if v is not None:q[i,j]=v
    return arr,q

def plot_forest(pub, source):
    data=read_csv(source/"behavioral"/"per_family_effects.csv")
    by={(r["feature"],r["model"]):r for r in data}
    fig,ax=plt.subplots(figsize=(10.7,8.2))
    y=np.arange(len(FEATURE_ORDER))[::-1]
    offsets=[-.27,-.09,.09,.27]
    used=[]
    for shift,(m,n) in zip(offsets,MODEL_NAMES.items()):
        subset=[]
        for f,base in zip(FEATURE_ORDER,y):
            row=by.get((f,m))
            if row is None:continue
            val,low,high=(*(number(row.get(k)) for k in ("mean","ci_low","ci_high")),)
            if None in (val,low,high):continue
            ax.plot([low,high],[base+shift]*2,color=COLORS[m],lw=1.45,zorder=2)
            ax.plot([low,low],[base+shift-.045,base+shift+.045],color=COLORS[m],lw=.85)
            ax.plot([high,high],[base+shift-.045,base+shift+.045],color=COLORS[m],lw=.85)
            sig=number(row["q_value"]) is not None and number(row["q_value"])<.05
            ax.scatter([val],[base+shift],s=28,marker="o",
                       color=COLORS[m] if sig else "white",
                       edgecolor=COLORS[m],linewidth=.95,zorder=3)
            subset.append(row)
        used+=subset
    ax.axvline(0,color="#444",lw=.9,ls="--")
    ax.axhline((y[len(ENGLISH)-1]+y[len(ENGLISH)])/2,color="#999",lw=.8,ls=":")
    ax.set_yticks(y,[label(f)+(" †" if f in CROSS_LANGUAGE else "") for f in FEATURE_ORDER])
    ax.set_xlabel("Difference in target-value support (0–4 judge scale)")
    ax.set_title("Feature effects across four model families",loc="left",pad=14)
    ax.grid(axis="x",alpha=.55)
    ax.legend([Line2D([0],[0],color=COLORS[m],marker="o",lw=1.5,label=n)
               for m,n in MODEL_NAMES.items()],list(MODEL_NAMES.values()),
              frameon=False,ncol=4,loc="lower left",bbox_to_anchor=(0,1.005))
    fig.subplots_adjust(left=.30,right=.98,bottom=.09,top=.88)
    pub.publish("fig01_feature_effects_forest",fig,used,
        "Feature-versus-registered-control effects in all four models; whiskers are the original 95% semantic-cell bootstrap intervals.",
        ["behavioral/per_family_effects.csv"],
        "Filled markers: within-family BH q<0.05; hollow: not significant. † marks a language-changing contrast. No pooled family-level confidence interval.")

def plot_replication(pub, source):
    data=read_csv(source/"behavioral"/"per_family_effects.csv")
    look={(r["feature"],r["model"]):r for r in data}
    z=np.full((len(FEATURE_ORDER),4),np.nan);q=z.copy()
    for i,f in enumerate(FEATURE_ORDER):
        for j,m in enumerate(FAMILIES):
            row=look.get((f,m))
            if row:
                v=number(row.get("mean"));z[i,j]=v if v is not None else np.nan
                v=number(row.get("q_value"));q[i,j]=v if v is not None else np.nan
    fig,ax=plt.subplots(figsize=(8.1,8.0))
    lim=max(.05,float(np.nanmax(np.abs(z))) * 1.04)
    im=centered_heatmap(ax,z,FEATURE_ORDER,list(MODEL_NAMES.values()),
                        "Cross-model replication of linguistic effects",lim,q=q)
    ax.axhline(len(ENGLISH)-.5,color="#222",lw=1.4)
    cb=fig.colorbar(im,ax=ax,fraction=.055,pad=.035,shrink=.82)
    cb.set_label("Feature minus control (0–4)")
    fig.subplots_adjust(left=.32,right=.90,top=.94,bottom=.13)
    pub.publish("fig02_replication_heatmap",fig,data,
                "Per-model welfare-support effects for all linguistic features.",
                ["behavioral/per_family_effects.csv"],
                "* indicates q<0.05 within a model; three bottom rows change training language. Missing values appear in grey.")

def plot_correlations(pub, source):
    raw=read_csv(source/"behavioral"/"cross_family_correlations.csv")
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.9),sharex=True,sharey=True)
    rows=[]
    for ax,(scope,title) in zip(axes,[("all_14","All 14 transformations"),
                                      ("within_english_11","11 within-English transformations")]):
        part=[r for r in raw if r["scope"]==scope]
        z=np.eye(4)
        for r in part:
            i=list(FAMILIES).index(r["first"]);j=list(FAMILIES).index(r["second"])
            z[i,j]=z[j,i]=number(r["pearson"])
            rows.append(r)
        im=ax.imshow(z,cmap="PuOr",vmin=-1,vmax=1)
        ax.set_xticks(range(4),list(MODEL_NAMES.values()),rotation=40,ha="right")
        ax.set_yticks(range(4),list(MODEL_NAMES.values()))
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f"{z[i,j]:.2f}",ha="center",va="center",
                        fontsize=9,color="white" if abs(z[i,j])>.8 else "#202020")
        ax.set_title(title)
    cb=fig.colorbar(im,ax=list(axes),shrink=.72,fraction=.03,pad=.035)
    cb.set_label("Pearson correlation across feature effects")
    fig.subplots_adjust(left=.13,right=.89,bottom=.21,top=.89,wspace=.18)
    pub.publish("fig03_architecture_correlation",fig,rows,
        "Pairwise descriptive correlations of feature-effect profiles by model.",
        ["behavioral/cross_family_correlations.csv"],
        "Correlations are across 14 or 11 feature estimates, not independent samples of model families; the diagonal equals one by definition.")

def plot_category_matrices(pub, root, key, filename, title, colkey,
                           metric="mean", note=""):
    gathered=[]
    fig,axes=plt.subplots(2,2,figsize=(13.2,13.3))
    matrices=[];qs=[]
    for m in FAMILIES:
        file=root/"results"/m/"evaluation_results"/"analysis"/"output"/"tables"/filename
        rows=read_csv(file)
        columns=CATEGORIES if colkey=="category" else tuple(str(i) for i in sorted({int(r[colkey]) for r in rows}))
        # Preserve labels even when a stratum is absent for a model.
        arr,q=array_by_model(rows,metric=metric,columns=columns)
        gathered += [{"model":m,**r} for r in rows]
        matrices.append((arr,columns))
        qs.append(q)
    observed=np.concatenate([arr.ravel() for arr,_ in matrices])
    finite=observed[np.isfinite(observed)]
    if not len(finite):
        plt.close(fig);pub.skipped(key,"No finite stratum effects",family="behavioral")
        return
    lim=max(.05,float(np.max(np.abs(finite)))*1.02)
    im=None
    for ax,(m,(mat,columns),q) in zip(axes.flat,zip(FAMILIES,matrices,qs)):
        im=centered_heatmap(ax,mat,FEATURE_ORDER,
                            [label(s) for s in columns],MODEL_NAMES[m],lim,q)
        ax.axhline(len(ENGLISH)-.5,color="#222",lw=1.2)
    fig.suptitle(title,x=.14,ha="left",fontsize=13)
    cb=fig.colorbar(im,ax=list(axes.flat),fraction=.016,pad=.014,shrink=.58)
    cb.set_label("Feature minus control (0–4)" if metric=="mean" else "Tradeoff interaction (0–4)")
    fig.subplots_adjust(left=.12,right=.91,top=.93,bottom=.14,wspace=.35,hspace=.45)
    pub.publish(key,fig,gathered,
        title+". Original cluster-adjusted statistics are retained in the source-data export.",
        [f"results/{m}/evaluation_results/analysis/output/tables/{filename}" for m in FAMILIES],
        note+" Asterisks denote the original within-family FDR-adjusted q<0.05; grey cells are missing.")

def plot_baselines(pub,root):
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),sharey=True)
    used=[]
    for ax,filename,title in zip(axes,
         ("canonical_minus_base_by_category.csv","control_minus_base_by_category.csv"),
         ("English canonical − base","Mandarin control − base")):
        for m in FAMILIES:
            file=root/"results"/m/"evaluation_results"/"analysis"/"output"/"tables"/filename
            rows=read_csv(file);by={r["category"]:r for r in rows}
            y=[number(by[c]["mean"]) if c in by else np.nan for c in CATEGORIES]
            ax.plot(range(len(CATEGORIES)),y,marker="o",ms=4,lw=1.5,
                    color=COLORS[m],label=MODEL_NAMES[m])
            for j,c in enumerate(CATEGORIES):
                if c in by:
                    low,high=number(by[c]["ci_low"]),number(by[c]["ci_high"])
                    if low is not None and high is not None:
                        ax.plot([j,j],[low,high],color=COLORS[m],lw=1,alpha=.70)
            used.extend({"model":m,"contrast":title,**r} for r in rows)
        ax.axhline(0,color="#444",lw=.85)
        ax.set_xticks(range(len(CATEGORIES)),[label(c) for c in CATEGORIES],
                      rotation=55,ha="right")
        ax.set_title(title)
        ax.set_xlabel("Generalization category")
        ax.grid(axis="y",alpha=.5)
    axes[0].set_ylabel("Gain in target-value support (0–4)")
    axes[1].legend(frameon=False,ncol=2,loc="upper right")
    fig.suptitle("Fine-tuning gains relative to pretrained models",fontsize=13,x=.11,ha="left")
    fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.29,wspace=.13)
    pub.publish("fig04_baseline_gains",fig,used,
        "Canonical and source-language control gains relative to each pretrained model, stratified by generalization category.",
        ["results/<model>/evaluation_results/analysis/output/tables/{canonical_minus_base,control_minus_base}_by_category.csv"],
        "Whiskers are within-family category-level 95% cluster-bootstrap confidence intervals; comparisons are not cross-family hypothesis tests.")

def plot_sensitivity(pub, source):
    rows=read_csv(source/"behavioral"/"sensitivity.csv")
    fig,axes=plt.subplots(2,2,figsize=(10.2,8.5),sharex=True,sharey=True)
    xy=[]
    for ax,m in zip(axes.flat,FAMILIES):
        part=[r for r in rows if r["model"]==m]
        x=np.array([number(r.get("primary_mean")) for r in part],dtype=float)
        y=np.array([number(r.get("high_coherence_mean")) for r in part],dtype=float)
        ok=np.isfinite(x)&np.isfinite(y)
        if ok.any():
            ax.scatter(x[ok],y[ok],s=28,facecolor=COLORS[m],alpha=.78)
            xy.extend(x[ok].tolist()+y[ok].tolist())
        ax.set_title(MODEL_NAMES[m]);ax.grid(alpha=.45)
    if xy:
        low=min(xy)-.025;high=max(xy)+.025
        for ax in axes.flat:
            ax.plot([low,high],[low,high],color="#444",ls="--",lw=.85)
            ax.set_xlim(low,high);ax.set_ylim(low,high)
    for ax in axes[-1]:ax.set_xlabel("Primary mean")
    for ax in axes[:,0]:ax.set_ylabel("High-coherence subset mean")
    fig.suptitle("Robustness to the response-coherence restriction",
                 x=.08,ha="left",fontsize=13)
    fig.subplots_adjust(left=.11,right=.98,top=.88,bottom=.10,hspace=.32,wspace=.22)
    pub.publish("fig08_high_coherence_sensitivity",fig,rows,
        "Comparison of primary feature estimates with estimates on high-coherence responses.",
        ["behavioral/sensitivity.csv"],
        "Each marker represents a feature; the dashed diagonal denotes equality. These are sensitivity estimates, not independent replications.")

def plot_lexical(pub,source):
    rows=read_csv(source/"behavioral"/"lexical_associations.csv")
    fig,ax=plt.subplots(figsize=(7.7,4.4))
    x=np.arange(4)
    vals=[number(next(r for r in rows if r["model"]==m)["pearson_r"]) for m in FAMILIES]
    ps=[number(next(r for r in rows if r["model"]==m)["pearson_permutation_p"]) for m in FAMILIES]
    ax.bar(x,vals,color=[COLORS[m] for m in FAMILIES],width=.63)
    for i,(v,p) in enumerate(zip(vals,ps)):
        ax.annotate(f"r = {v:.2f}\np = {p:.3f}",(i,v),ha="center",
                    va="bottom",xytext=(0,5),textcoords="offset points",fontsize=8)
    ax.axhline(0,color="#444",lw=.9)
    ax.set_xticks(x,list(MODEL_NAMES.values()))
    ax.set_ylim(min(0,min(vals)-.1),max(1,max(vals)+.15))
    ax.set_ylabel("Pearson r, feature-level lexical change vs. welfare effect")
    ax.set_title("Lexical diversity as a potential alternative explanation",loc="left")
    ax.grid(axis="y",alpha=.38)
    fig.subplots_adjust(left=.13,right=.98,top=.89,bottom=.15)
    pub.publish("fig09_lexical_associations",fig,rows,
        "Association between feature-level lexical-diversity change and welfare-support differences.",
        ["behavioral/lexical_associations.csv"],
        "Exploratory: 11 within-English features per model, permutation p-values from the upstream analyses; no pooled inference or mediation claim.")

def plot_reliability(pub,source):
    rows=read_csv(source/"behavioral"/"judge_reliability.csv")
    metrics=["target_value_support","behavioral_commitment","tradeoff_priority"]
    fig,ax=plt.subplots(figsize=(8.9,4.4))
    offsets=np.linspace(-.29,.29,4)
    for m,off in zip(FAMILIES,offsets):
        part={(r["metric"]):r for r in rows if r["model"]==m}
        ys=[number(part[t]["quadratic_weighted_kappa"]) if t in part else np.nan for t in metrics]
        ax.bar(np.arange(len(metrics))+off,ys,width=.19,color=COLORS[m],
               label=MODEL_NAMES[m])
    ax.set_xticks(range(len(metrics)),[label(s) for s in metrics])
    ax.set_ylim(0,1);ax.set_ylabel("Quadratic-weighted Cohen's kappa")
    ax.set_title("Inter-judge reliability across model families",loc="left")
    ax.legend(frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.01))
    ax.grid(axis="y",alpha=.36)
    fig.subplots_adjust(left=.11,right=.99,top=.90,bottom=.16)
    pub.publish("fig10_judge_reliability",fig,rows,
        "Inter-judge agreement on welfare support, behavioral commitment, and tradeoff priority.",
        ["behavioral/judge_reliability.csv"],
        "Kappa is reported on the upstream judged scores; high agreement does not establish construct validity.")

def plot_quality_relation(pub,source):
    support=read_csv(source/"behavioral"/"per_family_effects.csv")
    coh=read_csv(source/"behavioral"/"coherence_replication.csv")
    by={(r["feature"],m):number(r.get(m+"_effect")) for r in coh for m in FAMILIES}
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    plotted=[]
    for ax,m in zip(axes.flat,FAMILIES):
        part=[r for r in support if r["model"]==m]
        x,y=[],[]
        for r in part:
            w=number(r["mean"]);c=by.get((r["feature"],m))
            if w is not None and c is not None:
                x.append(c);y.append(w);plotted.append({"model":m,"feature":r["feature"],
                                                          "delta_coherence":c,"delta_welfare":w})
        ax.scatter(x,y,color=COLORS[m],s=26,alpha=.85)
        ax.axvline(0,ls="--",lw=.75,color="#555")
        ax.axhline(0,ls="--",lw=.75,color="#555")
        ax.set_title(MODEL_NAMES[m]);ax.grid(alpha=.34)
    for ax in axes[-1]:ax.set_xlabel("Change in coherence/relevance")
    for ax in axes[:,0]:ax.set_ylabel("Change in target-value support")
    fig.suptitle("Behavioral effects versus response-coherence changes",
                 x=.10,ha="left",fontsize=13)
    fig.subplots_adjust(left=.12,right=.97,top=.89,bottom=.10,wspace=.18,hspace=.32)
    pub.publish("fig11_response_coherence",fig,plotted,
        "Feature-level relationship between changes in welfare-support ratings and response-coherence ratings.",
        ["behavioral/per_family_effects.csv","behavioral/coherence_replication.csv"],
        "Descriptive association; it cannot rule out shared judging or training-related confounding.")

def generate_behavioral(pub, source:Path, root:Path):
    plot_forest(pub,source)
    plot_replication(pub,source)
    plot_correlations(pub,source)
    plot_baselines(pub,root)
    plot_category_matrices(pub,root,"fig05_generalization_categories",
                           "feature_by_category.csv","Generalization-category heterogeneity","category")
    plot_category_matrices(pub,root,"fig06_heldout_dimensions",
                           "feature_by_heldout_dimensions.csv",
                           "Effects by number of held-out dimensions","heldout_dimensions")
    plot_category_matrices(pub,root,"fig07_tradeoff_interactions",
                           "tradeoff_interactions.csv",
                           "Tradeoff versus non-tradeoff interactions","category",
                           metric="tradeoff_minus_nontradeoff",
                           note="Interaction estimates are not raw tradeoff-specific effects.")
    plot_sensitivity(pub,source)
    plot_lexical(pub,source)
    plot_reliability(pub,source)
    plot_quality_relation(pub,source)
