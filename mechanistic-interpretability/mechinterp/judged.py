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

def verified_batch_judgments(group, returned):
    expected={int(r["scenario_id"]) for r in group}
    matched={}
    for judgment in returned:
        sid=int(judgment["scenario_id"])
        if sid not in expected or sid in matched:
            raise ValueError(f"Unknown or duplicate scenario_id: {sid}")
        matched[sid]=judgment
    if set(matched)!=expected:
        raise ValueError("Missing scenario IDs from judge response")
    return matched


def id_keyed_judgment_schema(count):
    from schemas import JUDGMENT_SCHEMA
    item={"type":"object","additionalProperties":False,
          "required":["scenario_id",*JUDGMENT_SCHEMA["required"]],
          "properties":{"scenario_id":{"type":"integer","minimum":1},
                        **JUDGMENT_SCHEMA["properties"]}}
    return {"type":"object","additionalProperties":False,"required":["judgments"],
            "properties":{"judgments":{"type":"array","minItems":count,"maxItems":count,
                                       "items":item}}}


def selected_conditions(cfg,family,policy="prompt_last"):
    path=run_dir(cfg,family,"activation-steering")/"selected_all.json"
    selected=json.loads(path.read_text(encoding="utf-8"))
    keys=[(r["feature"],r["adapter"]) for r in selected]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate selected steering conditions")
    for row in selected:
        if row["adapter"] not in (control_for(row["feature"]),row["feature"]):
            raise ValueError("Selected feature applied to unrelated adapter")
    return selected

def workload(cfg,family,policy="prompt_last"):
    selected=selected_conditions(cfg,family,policy)
    test=[r for r in load_screening(cfg) if r["split"]==cfg["steering"]["judged_split"]]
    paired=len(test)*len(selected)
    generation_count=2*paired
    judge_batches=math.ceil(len(test)/int(cfg["steering"]["judging"]["batch_size"]))*len(selected)*2*2
    return {
        "family":family,"protocol":"paired_revised_v2",
        "configurations":len(selected),"scenarios_per_configuration":len(test),
        "matched_pairs":paired,"generations":generation_count,
        "unsteered_generations":paired,"steered_generations":paired,
        "individual_judgments":generation_count*2,
        "api_requests_without_retries":judge_batches,
        "judge_batch_size":int(cfg["steering"]["judging"]["batch_size"]),
        "token_estimate_note":"Judge and response tokens vary; these are exact request counts before retries.",
    }

def write_workload(cfg,family,policy="prompt_last"):
    result=workload(cfg,family,policy)
    path=run_dir(cfg,family,"activation-steering")/"workload_all.json"
    write_json(path,result)
    return path

def validate_generation_pairs(pairs, test, feature, adapter, policy, max_new_tokens, seed):
    """Validate resumable pair checkpoints before accepting them."""
    reference={int(item["scenario_id"]):item for item in test}
    used=set()
    for pair in pairs:
        sid=int(pair["scenario_id"])
        if sid not in reference or sid in used:
            raise ValueError(f"Unknown or duplicate completed scenario: {sid}")
        used.add(sid)
        a,b=pair["unsteered"],pair["steered"]
        expected_prompt=scenario_prompt(reference[sid]["scenario"])
        for arm,row in (("unsteered",a),("steered",b)):
            if (row["scenario_id"]!=sid or row["feature"]!=feature or
                row["adapter"]!=adapter or row["policy"]!=policy or
                row["arm"]!=arm or row["prompt"]!=expected_prompt or
                row["generation_seed"]!=int(seed)+sid or
                row.get("max_new_tokens")!=int(max_new_tokens)):
                raise ValueError(f"Incompatible or corrupted {arm} record: {sid}")
        if abs(float(a["strength"]))>1e-10:
            raise ValueError("The unsteered arm has nonzero intervention strength")
    return used


def generate(cfg,family,policy="prompt_last"):
    """Produce same-seed matched pairs, atomically checkpointing every five pairs.

    A restart resumes only complete matching pairs and cannot silently overwrite
    old target files. Original 96-token output files require explicit archiving:
    files without 256-token generation metadata are not reused.
    """
    from .directions import load_direction_bundle
    selected=selected_conditions(cfg,family,policy)
    test=[r for r in load_screening(cfg) if r["split"]==cfg["steering"]["judged_split"]]
    if len(test)!=cfg["steering"]["judged_n"]:
        raise ValueError("Unexpected held-out scenario count")
    out=run_dir(cfg,family,"activation-steering")/"judged"
    out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for entry in selected:
        grouped[entry["adapter"]].append(entry)
    generation=cfg["steering"]["generation"]
    max_new=int(generation["max_new_tokens"])
    if max_new<1:raise ValueError("Invalid max_new_tokens")
    for condition,specs in grouped.items():
        # Skip fully complete, compatible conditions without loading model.
        todo=[]
        for entry in specs:
            stem=f"{entry['feature']}__{condition}__{entry['policy']}"
            files={arm:out/f"{stem}__{arm}.jsonl" for arm in ("unsteered","steered")}
            partial=out/f"{stem}__paired.partial.jsonl"
            if all(f.exists() for f in files.values()) and not partial.exists():
                left=read_data(files["unsteered"])
                right=read_data(files["steered"])
                if len(left)!=len(test) or len(right)!=len(test):
                    raise ValueError(f"Incomplete previously generated pair files: {stem}")
                pairs=[{"scenario_id":a["scenario_id"],"unsteered":a,"steered":b}
                       for a,b in zip(left,right)]
                if len({a["scenario_id"] for a in left})!=len(left) or any(
                    a["scenario_id"]!=b["scenario_id"] for a,b in zip(left,right)):
                    raise ValueError(f"Misaligned previously generated pair files: {stem}")
                validate_generation_pairs(pairs,test,entry["feature"],condition,
                                          entry["policy"],max_new,generation["seed"])
                print(f"Skipping verified completed generations: {stem}",flush=True)
                continue
            if any(f.exists() for f in files.values()) and not partial.exists():
                raise FileExistsError(f"One output arm missing and no pair checkpoint: {stem}")
            todo.append((entry,files,partial))
        if not todo:continue
        bundles={s["feature"]:load_direction_bundle(cfg,family,s["feature"])
                 for s,_,_ in todo}
        model,tok=load_model(cfg,family,condition)
        try:
            for entry,files,partial in todo:
                feature=entry["feature"]
                layer=entry["layer"];site=entry["site"]
                direction,scale=bundles[feature](layer,site)
                pairs=read_data(partial) if partial.exists() else []
                complete=validate_generation_pairs(pairs,test,feature,condition,
                                                    entry["policy"],max_new,
                                                    generation["seed"])
                for item in test:
                    sid=int(item["scenario_id"])
                    if sid in complete:continue
                    prompt=scenario_prompt(item["scenario"])
                    tokens=encode(tok,[prompt],model.device,cfg["runtime"]["max_length"])
                    seed=int(generation["seed"])+sid
                    arms={}
                    for arm in ("unsteered","steered"):
                        strength=0. if arm=="unsteered" else float(entry["strength"])
                        torch.manual_seed(seed)
                        with torch.inference_mode(),steer(
                            model,layer,site,direction*scale,strength,entry["policy"]):
                            output=model.generate(
                                **tokens,max_new_tokens=max_new,
                                do_sample=generation["do_sample"],
                                temperature=generation["temperature"],
                                top_p=generation["top_p"],
                                pad_token_id=tok.pad_token_id)
                        tail=output[0,tokens["input_ids"].shape[1]:]
                        eos_ids=model.generation_config.eos_token_id
                        if eos_ids is None:eos_ids=tok.eos_token_id
                        eos_ids={int(x) for x in (eos_ids if isinstance(eos_ids,(list,tuple)) else [eos_ids]) if x is not None}
                        ended=bool(len(tail) and int(tail[-1]) in eos_ids)
                        continuation=tok.decode(tail,skip_special_tokens=True).strip()
                        arms[arm]={
                            "scenario_id":sid,"feature":feature,"adapter":condition,
                            "layer":layer,"site":site,"arm":arm,
                            "strength":strength,"policy":entry["policy"],
                            "condition":f"{feature}__{condition}__{entry['policy']}",
                            "generation_seed":seed,"continuation":continuation,
                            "prompt":prompt,"generated_tokens":int(len(tail)),
                            "ended_with_eos":ended,
                            "reached_token_limit":bool(len(tail)>=max_new and not ended),
                            "max_new_tokens":max_new,
                        }
                    pairs.append({"scenario_id":sid,**arms})
                    complete.add(sid)
                    if len(pairs)%5==0:
                        write_rows(partial,pairs)
                        print(f"Generated {feature}/{condition}: {len(pairs)}/{len(test)} pairs",flush=True)
                validate_generation_pairs(pairs,test,feature,condition,
                                          entry["policy"],max_new,generation["seed"])
                if len(pairs)!=len(test):
                    raise ValueError(f"Incomplete matched generation for {feature}/{condition}")
                write_rows(partial,pairs)
                for arm,path in files.items():
                    expected=[pair[arm] for pair in pairs]
                    if path.exists():
                        if read_data(path)!=expected:
                            raise ValueError(f"Existing output mismatch; refusing overwrite: {path}")
                    else:
                        write_rows(path,expected)
                partial.unlink(missing_ok=True)
                print(f"Completed matched generations: {feature}/{condition}",flush=True)
        finally:
            unload(model)
    return out


def judge_plan(cfg,family,policy="prompt_last"):
    """Every feature/configuration must have both separately judged arms."""
    directory=run_dir(cfg,family,"activation-steering")/"judged"
    selected=selected_conditions(cfg,family,policy)
    files=[]
    for s in selected:
        for arm in ("unsteered","steered"):
            f=directory/f"{s['feature']}__{s['adapter']}__{s['policy']}__{arm}.jsonl"
            if not f.is_file():raise FileNotFoundError(f)
            rows=read_data(f)
            if len(rows)!=cfg["steering"]["judged_n"] or len(set(r["scenario_id"] for r in rows))!=len(rows):
                raise ValueError(f"Incomplete or duplicate judged file: {f}")
            if any(row.get("arm")!=arm for row in rows):
                raise ValueError(f"Arm mismatch in {f}")
            files.append(f)
    return files

async def judge(cfg,family,policy="prompt_last",execute=False):
    if not execute:raise PermissionError("Judge stage requires explicit --execute-judges")
    import sys
    evaluation_scripts=location(cfg,"evaluation_config").parent.parent/"scripts"
    sys.path.insert(0,str(evaluation_scripts))
    from common import OpenRouterClient, render_prompt, json_text
    ecfg=yaml.safe_load(location(cfg,"evaluation_config").read_text(encoding="utf-8"))
    ecfg["judging"]["batch_size"]=int(cfg["steering"]["judging"]["batch_size"])
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
                    prompt += "\nInclude each item's original scenario_id in its judgment. Order does not matter.\n"
                    matched=None
                    for attempt in range(int(cfg["steering"]["judging"].get("invalid_batch_retries",3))):
                        response=await client.request_json(model_key=f"response_judge_{judge_no}",
                            temperature_key=f"response_judge_{judge_no}",seed_key=f"response_judge_{judge_no}",
                            prompt=prompt,schema=id_keyed_judgment_schema(len(group)),
                            schema_name=f"steering_judgments_{judge_no}")
                        try:
                            matched=verified_batch_judgments(group,response["judgments"])
                            break
                        except (ValueError,KeyError,TypeError) as exc:
                            if attempt+1>=int(cfg["steering"]["judging"].get("invalid_batch_retries",3)):
                                raise RuntimeError(f"Judge {judge_no} repeatedly returned mismatched scenario IDs") from exc
                            print(f"Retrying malformed judge batch (judge {judge_no}, attempt {attempt+1})",flush=True)
                    for scenario_id,judgment in matched.items():
                        completed[scenario_id]={"scenario_id":scenario_id,
                             "judge":judge_no,"scores":judgment["scores"],"outcome":judgment["outcome"]}
                    write_rows(partial,[completed[k] for k in sorted(completed)])
                if len(completed)!=len(source):raise RuntimeError("Missing judged responses")
                write_rows(target,[completed[k] for k in sorted(completed)])
                partial.unlink(missing_ok=True)
    finally:await client.close()
    write_json(out/"api_usage.json",client.stats())
    return out

def aggregate(cfg,family,policy="prompt_last"):
    """Judge-matched steered minus newly generated unsteered responses.

    Judge files are distinct per arm. Pair IDs, prompt and seeds are asserted.
    A legacy historical-baseline comparison may be analyzed separately, but
    never mixed with this paired-v2 inferential family.
    """
    from .stats import cluster_signflip
    source=judge_plan(cfg,family,policy)
    folder=run_dir(cfg,family,"activation-steering")/"judgments"
    scenarios={int(r["id"]):r for r in load_scenarios(cfg)}
    by_name={f.name:f for f in source}
    rows=[]
    for spec in selected_conditions(cfg,family,policy):
        stem=f"{spec['feature']}__{spec['adapter']}__{spec['policy']}"
        arm_data={}
        for arm in ("unsteered","steered"):
            name=f"{stem}__{arm}.jsonl"
            response=read_data(by_name[name])
            j1={int(r["scenario_id"]):r for r in read_data(folder/"judge_1"/name)}
            j2={int(r["scenario_id"]):r for r in read_data(folder/"judge_2"/name)}
            arm_data[arm]={}
            for r in response:
                sid=int(r["scenario_id"])
                if sid not in j1 or sid not in j2:
                    raise ValueError(f"Missing judge result for {sid} in {name}")
                vals=[j["scores"].get("target_value_support") for j in (j1[sid],j2[sid])]
                usable=[float(v) for v in vals if v is not None]
                arm_data[arm][sid]={
                    "source":r,"judge_1":j1[sid]["scores"],"judge_2":j2[sid]["scores"],
                    "score":sum(usable)/len(usable) if usable else None,
                }
        if set(arm_data["unsteered"])!=set(arm_data["steered"]):
            raise ValueError("Arm scenario IDs differ")
        for sid,baseline in arm_data["unsteered"].items():
            intervention=arm_data["steered"][sid]
            a,b=baseline["source"],intervention["source"]
            if a["prompt"]!=b["prompt"] or a["generation_seed"]!=b["generation_seed"]:
                raise ValueError("Unmatched prompt or seed in judged comparison")
            score0=baseline["score"];score1=intervention["score"]
            sc=scenarios[sid]
            cell="|".join(map(str,(sc["category"],sc["animal"],sc["value"],sc["context"])))
            rows.append({
                **b,"baseline_continuation":a["continuation"],
                "unsteered_score":score0,"target_value_support":score1,
                "paired_delta":score1-score0 if score0 is not None and score1 is not None else None,
                "semantic_cell":cell,"category":sc["category"],"tradeoff":sc["tradeoff"],
                "judge_1":intervention["judge_1"],"judge_2":intervention["judge_2"],
                "baseline_judge_1":baseline["judge_1"],"baseline_judge_2":baseline["judge_2"],
                "comparison_protocol":"matched_generated_baseline_v2",
            })
    dest=folder/"combined.jsonl"
    write_rows(dest,rows)
    groups=defaultdict(list)
    for row in rows:
        groups[(row["feature"],row["adapter"])].append(row)
    summaries=[]
    for (feature,adapter),part in groups.items():
        usable=[r for r in part if r["paired_delta"] is not None]
        deltas=[r["paired_delta"] for r in usable]
        clusters=[r["semantic_cell"] for r in usable]
        if not deltas:raise ValueError(f"No valid paired outcomes for {feature}/{adapter}")
        summaries.append({
            "feature":feature,"adapter":adapter,"n":len(usable),
            "comparison_protocol":"matched_generated_baseline_v2",
            "paired_delta":cluster_interval(deltas,clusters,int(cfg["seed"]),
                                            int(cfg["stats"]["bootstrap_replicates"])),
            "p_value":cluster_signflip(deltas,clusters,int(cfg["seed"]),
                                       int(cfg["stats"]["permutations"])),
        })
    q=bh([item["p_value"] for item in summaries])
    for row,qvalue in zip(summaries,q):row["q_value"]=qvalue
    write_json(folder/"summary.json",summaries)
    return dest
