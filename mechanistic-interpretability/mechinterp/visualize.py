"""Static scientific plots from aggregate JSONL outputs; no model inference required."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
from .common import read_data,location

def _import_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _heatmap(path,rows,index,columns,value,title,plt):
    xlabels=sorted(set(r[columns] for r in rows),key=str)
    ylabels=sorted(set(r[index] for r in rows),key=str)
    if not xlabels or not ylabels:return
    arrays=defaultdict(list)
    for r in rows:
        if r.get(value) is not None:arrays[(r[index],r[columns])].append(float(r[value]))
    grid=np.full((len(ylabels),len(xlabels)),np.nan)
    for i,y in enumerate(ylabels):
        for j,x in enumerate(xlabels):
            values=arrays[(y,x)]
            if values:grid[i,j]=np.mean(values)
    fig,ax=plt.subplots(figsize=(max(7,len(xlabels)*.22),max(3,len(ylabels)*.55)))
    im=ax.imshow(grid,aspect="auto",interpolation="nearest")
    ax.set_xticks(range(len(xlabels)),[str(v) for v in xlabels],rotation=90)
    ax.set_yticks(range(len(ylabels)),[str(v) for v in ylabels])
    ax.set_title(title)
    fig.colorbar(im,ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=130)
    plt.close(fig)

def plots(cfg,family):
    plt=_import_plot()
    root=location(cfg,"outputs")/family
    out=root/"plots";out.mkdir(parents=True,exist_ok=True)
    generated=[]
    diff=root/"model-difference-analysis"
    if diff.is_dir():
        for file in diff.glob("*.jsonl"):
            data=read_data(file)
            for metric in ("cka","svcca","displacement"):
                rows=[{**r,"row":f"{r['domain']}/{r.get('category','all')}/{r['site']}/{r['pooling']}"}
                      for r in data if r.get("category","all")=="all"]
                target=out/f"diff_{file.stem}_{metric}.png"
                _heatmap(target,rows,"row","layer",metric,
                         f"{file.stem}: {metric} by layer",plt)
                if target.exists():generated.append(str(target))
    probes=root/"linear-probing"
    if probes.is_dir():
        for file in probes.glob("*.jsonl"):
            data=read_data(file)
            if not data:continue
            metric="r2" if data[0].get("task")=="behavior" else "roc_auc"
            rows=[]
            for row in data:
                score=row.get("test",{}).get(metric)
                if score is not None:
                    rows.append({"layer":row["layer"],"row":row["site"]+"/"+row["pooling"],
                                 "value":score})
            target=out/f"probe_{file.stem}.png"
            _heatmap(target,rows,"row","layer","value",f"{file.stem}: test {metric}",plt)
            if target.exists():generated.append(str(target))
    lora=root/"LoRA-analysis"/"module_geometry.jsonl"
    if lora.exists():
        raw=read_data(lora)
        for checkpoint in ("midpoint","final"):
            rows=[]
            for row in raw:
                if row["checkpoint"]!=checkpoint:continue
                match=re.search(r"model\.layers\.(\d+).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",row["module"])
                if match:
                    rows.append({"layer":int(match[1]),
                                 "row":row["condition"]+"/"+match[2],
                                 "value":row["frobenius"]})
            target=out/f"lora_{checkpoint}_norm.png"
            _heatmap(target,rows,"row","layer","value",f"LoRA {checkpoint}: update norm",plt)
            if target.exists():generated.append(str(target))
    steering=root/"activation-steering"
    if steering.exists():
        for file in steering.glob("screen_*.jsonl"):
            raw=read_data(file)
            baselines={(r["layer"],r["site"],r["scenario_id"]):r["score"] for r in raw
                       if r["split"]=="validation" and r["strength"]==0}
            for site in ("residual","mlp"):
                rows=[]
                for r in raw:
                    if r["site"]!=site or r["split"]!="validation":continue
                    base=baselines.get((r["layer"],site,r["scenario_id"]))
                    if base is None:continue
                    rows.append({"layer":r["layer"],"strength":r["strength"],"delta":r["score"]-base})
                target=out/f"{file.stem}_{site}_dose.png"
                _heatmap(target,rows,"strength","layer","delta",
                         f"{file.stem} {site}: mean welfare-option log-odds delta",plt)
                if target.exists():generated.append(str(target))
    return {"count":len(generated),"directory":str(out)}
