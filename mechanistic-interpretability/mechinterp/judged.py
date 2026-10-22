"""Explicitly authorized judged-evaluation generation, batch judging, aggregation.

No network request occurs without the --execute-judges flag. Generation runs separately,
and can be inspected before invocation of remote judges.
"""
from __future__ import annotations
import asyncio
import json
import math
import os
from collections import Counter,defaultdict
from pathlib import Path
import random
import torch
import yaml
from .common import (location,load_screening,load_scenarios,feature_pairs,load_general,
                     read_data,write_json,write_rows,run_dir,scenario_prompt,provenance,
                     control_for,batches)
from .models import load_model,steer,encode,unload,blocks
from .steering import direction_for,calibrate
from .stats import cluster_interval,bh

def selected_conditions(cfg,family,policy="prompt_last"):
    path=run_dir(cfg,family,"activation-steering")/f"selected_{policy}.json"
    selected=json.loads(path.read_text(encoding="utf-8"))
    keys=[(r["feature"],r["adapter"]) for r in selected]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate selected steering conditions")
    for row in selected:
        if row["adapter"] not in (control_for(row["feature"]),row["feature"]):
            raise ValueError("Selected feature applied to unrelated adapter")
        if row["policy"]!=policy:raise ValueError("Token policy mismatch")
    return selected

def workload(cfg,family,policy="prompt_last"):
    selected=selected_conditions(cfg,family,policy)
    test=[r for r in load_screening(cfg) if r["split"]==cfg["steering"]["judged_split"]]
    n=len(test)*len(selected)
    judge_batches=math.ceil(len(test)/cfg["steering"]["judging"]["batch_size"])*len(selected)*2
    result={"family":family,"configurations":len(selected),"scenarios_per_configuration":len(test),
            "generations":n,"individual_judgments":n*2,"api_requests_without_retries":judge_batches,
            "judge_batch_size":cfg["steering"]["judging"]["batch_size"],
            "token_estimate_note":"API tokens vary with prompt length and judge reasoning; configured unit prices and token estimates are needed to price requests."}
    price=cfg["steering"]["judging"]
    if price.get("input_usd_per_million") is not None and price.get("output_usd_per_million") is not None:
        # Deliberately do not fabricate a cost without model-specific token estimates.
        result["estimated_cost_usd"]=None
    return result

def write_workload(cfg,family,policy="prompt_last"):
    result=workload(cfg,family,policy)
    path=run_dir(cfg,family,"activation-steering")/f"workload_{policy}.json"
    write_json(path,result)
    return path

def generate(cfg,family,policy="prompt_last"):
    """Only use held-out 100 scenarios. Save generation metadata with every response."""
    selected=selected_conditions(cfg,family,policy)
    test=[r for r in load_screening(cfg) if r["split"]==cfg["steering"]["judged_split"]]
    if len(test)!=cfg["steering"]["judged_n"]:raise ValueError("Unexpected held-out scenario count")
    out=run_dir(cfg,family,"activation-steering")/"judged"
    out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for row in selected:grouped[row["adapter"]].append(row)
    for condition,specs in grouped.items():
        target,tok=load_model(cfg,family,condition)
        base,_=load_model(cfg,family,"base")
        try:
            for entry in specs:
                feature=entry["feature"];layer=entry["layer"];site=entry["site"]
                direction,_=direction_for(base,tok,feature_pairs(cfg,feature),layer,site,cfg)
                scale=calibrate(base,tok,[r["prompt"] for r in load_general(cfg)[:cfg["runtime"]["calibration_n"]]],layer,site,cfg)
                path=out/f"{feature}__{condition}__{policy}.jsonl"
                if path.exists():
                    raise FileExistsError(f"Judged generation already exists: {path}")
                rows=[]
                generation=cfg["steering"]["generation"]
                for j,row in enumerate(test):
                    # One scenario per batch for clear cache/padding and deterministic control.
                    prompt=scenario_prompt(row["scenario"])
                    tokens=encode(tok,[prompt],target.device,cfg["runtime"]["max_length"])
                    torch.manual_seed(int(generation["seed"])+int(row["scenario_id"]))
                    with torch.inference_mode(),steer(target,layer,site,direction*scale,entry["strength"],policy):
                        output=target.generate(**tokens,max_new_tokens=generation["max_new_tokens"],
                            do_sample=generation["do_sample"],temperature=generation["temperature"],
                            top_p=generation["top_p"],pad_token_id=tok.pad_token_id)
                    continuation=tok.decode(output[0,tokens["input_ids"].shape[1]:],skip_special_tokens=True).strip()
                    rows.append({"scenario_id":row["scenario_id"],"feature":feature,
                                 "adapter":condition,"layer":layer,"site":site,
                                 "strength":entry["strength"],"policy":policy,
                                 "condition":f"{feature}__{condition}__{policy}",
                                 "continuation":continuation,"prompt":prompt})
                write_rows(path,rows)
        finally:unload(target);unload(base)
    return out

def judge_plan(cfg,family,policy="prompt_last"):
    """Return only the generated response artifacts; no implicit API calls."""
    directory=run_dir(cfg,family,"activation-steering")/"judged"
    selected=selected_conditions(cfg,family,policy)
    files=[directory/f"{x['feature']}__{x['adapter']}__{policy}.jsonl" for x in selected]
    for path in files:
        if not path.is_file():raise FileNotFoundError(path)
        if len(read_data(path))!=cfg["steering"]["judged_n"]:
            raise ValueError(f"Incomplete response file {path}")
    return files

async def judge(cfg,family,policy="prompt_last",execute=False):
    if not execute:raise PermissionError("Judge stage requires explicit --execute-judges")
    import sys
    evaluation_scripts=location(cfg,"evaluation_config").parent.parent/"scripts"
    sys.path.insert(0,str(evaluation_scripts))
    from common import OpenRouterClient, render_prompt, json_text
    from schemas import judgment_batch_schema
    ecfg=yaml.safe_load(location(cfg,"evaluation_config").read_text(encoding="utf-8"))
    files=judge_plan(cfg,family,policy)
    scenarios={int(r["id"]):r for r in load_scenarios(cfg)}
    client=OpenRouterClient(ecfg)
    out=run_dir(cfg,family,"activation-steering")/"judgments"
    try:
        for judge_no in (1,2):
            (out/f"judge_{judge_no}").mkdir(parents=True,exist_ok=True)
            for path in files:
                target=out/f"judge_{judge_no}"/path.name
                if target.exists():continue
                source=read_data(path)
                partial=target.with_suffix(".partial.jsonl")
                completed={int(r["scenario_id"]):r for r in read_data(partial)} if partial.exists() else {}
                items=[]
                for row in source:
                    sid=int(row["scenario_id"])
                    scenario=scenarios[sid]
                    items.append({"scenario_id":sid,"animal":scenario["animal"],"value":scenario["value"],
                                  "context":scenario["context"],"tradeoff":scenario["tradeoff"],
                                  "scenario":scenario["scenario"],"continuation":row["continuation"]})
                random.Random(ecfg["judging"]["shuffle_seeds"][f"judge_{judge_no}"]).shuffle(items)
                for group in batches([r for r in items if r["scenario_id"] not in completed],
                                     int(ecfg["judging"]["batch_size"])):
                    template=location(cfg,"evaluation_prompt").read_text(encoding="utf-8")
                    prompt=template.replace("{{ITEMS_JSON}}",json_text(group))
                    response=await client.request_json(model_key=f"response_judge_{judge_no}",
                        temperature_key=f"response_judge_{judge_no}",seed_key=f"response_judge_{judge_no}",
                        prompt=prompt,schema=judgment_batch_schema(len(group)),
                        schema_name=f"steering_judgments_{judge_no}")
                    for item,judgment in zip(group,response["judgments"],strict=True):
                        completed[item["scenario_id"]]={"scenario_id":item["scenario_id"],
                             "judge":judge_no,"scores":judgment["scores"],"outcome":judgment["outcome"]}
                    write_rows(partial,[completed[k] for k in sorted(completed)])
                if len(completed)!=len(source):raise RuntimeError("Missing judged responses")
                write_rows(target,[completed[k] for k in sorted(completed)])
                partial.unlink(missing_ok=True)
    finally:await client.close()
    write_json(out/"api_usage.json",client.stats())
    return out

def aggregate(cfg,family,policy="prompt_last"):
    source=judge_plan(cfg,family,policy)
    directory=run_dir(cfg,family,"activation-steering")/"judgments"
    rows=[];scenario={int(r["id"]):r for r in load_scenarios(cfg)}
    for file in source:
        conditions=read_data(file)
        j1={r["scenario_id"]:r for r in read_data(directory/"judge_1"/file.name)}
        j2={r["scenario_id"]:r for r in read_data(directory/"judge_2"/file.name)}
        for r in conditions:
            sid=r["scenario_id"]
            judgments=[j1[sid],j2[sid]]
            vals=[j["scores"]["target_value_support"] for j in judgments
                  if j["scores"]["target_value_support"] is not None]
            value=sum(vals)/len(vals) if vals else None
            s=scenario[sid];cell="|".join(map(str,(s["category"],s["animal"],s["value"],s["context"])))
            rows.append({**r,"target_value_support":value,"semantic_cell":cell,
                         "judge_1":j1[sid]["scores"],"judge_2":j2[sid]["scores"]})
    output=directory/"combined.jsonl"
    write_rows(output,rows)
    summary=[]
    grouped=defaultdict(list)
    for row in rows:grouped[(row["feature"],row["adapter"])].append(row)
    for (feature,condition),part in grouped.items():
        usable=[r for r in part if r["target_value_support"] is not None]
        summary.append({"feature":feature,"adapter":condition,
            "score":cluster_interval([r["target_value_support"] for r in usable],
                [r["semantic_cell"] for r in usable],cfg["seed"],cfg["stats"]["bootstrap_replicates"])})
    write_json(directory/"summary.json",summary)
    return output
