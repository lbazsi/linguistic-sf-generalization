"""Versioned, resumable, matched-arm batched steering generation.

This sampling protocol is DIFFERENT from sequential scenario-seeded sampling.
Each fixed scenario batch has a deterministic batch seed, reset for both arms.
Do not mix sequential and batched samples within a judged experiment.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import torch

from .common import load_screening,run_dir,scenario_prompt,read_data,write_rows,batches
from .models import load_model,unload,encode,steer
from .directions import load_direction_bundle


PROTOCOL = "paired_batch_rng_v1"


def batch_seed(seed, scenarios):
    """Stable for a full dataset-order batch, independent of process restarts."""
    if not scenarios:
        raise ValueError("Empty generation batch")
    return int(seed) + int(scenarios[0]["scenario_id"])


def decode_generated_batch(output, prompt_length, tokenizer, model, max_new):
    """Trim at the first EOS, ignoring subsequent batch-padding tokens."""
    eos=model.generation_config.eos_token_id
    if eos is None:
        eos=tokenizer.eos_token_id
    eos_set={int(v) for v in (eos if isinstance(eos,(list,tuple)) else [eos]) if v is not None}
    result=[]
    for seq in output:
        tail=seq[prompt_length:].tolist()
        stop=next((i for i,t in enumerate(tail) if int(t) in eos_set),None)
        ended=stop is not None
        tokens=tail[:stop+1] if ended else tail
        result.append({
            "continuation":tokenizer.decode(tokens,skip_special_tokens=True).strip(),
            "generated_tokens":len(tokens),
            "ended_with_eos":ended,
            "reached_token_limit":bool(not ended and len(tokens)>=max_new),
            "max_new_tokens":int(max_new),
        })
    return result


def validate_batch_pairs(pairs, scenarios, feature, adapter, policy, strength,
                         max_new, requested_batch_size, master_seed):
    """Check paired metadata and frozen canonical batch grouping on every restart."""
    reference={int(x["scenario_id"]):x for x in scenarios}
    groups={}
    for batch in batches(scenarios,requested_batch_size):
        seed=batch_seed(master_seed,batch)
        group_ids=[int(x["scenario_id"]) for x in batch]
        for item in batch:
            groups[int(item["scenario_id"])]=(seed,group_ids)
    seen=set()
    for pair in pairs:
        sid=int(pair["scenario_id"])
        if sid not in reference or sid in seen:
            raise ValueError(f"Unknown/duplicate sampled scenario ID: {sid}")
        seen.add(sid)
        expected_seed,ids=groups[sid]
        for arm in ("unsteered","steered"):
            row=pair[arm]
            expected_strength=0. if arm=="unsteered" else float(strength)
            if (int(row["scenario_id"])!=sid or row["feature"]!=feature
                or row["adapter"]!=adapter or row["policy"]!=policy
                or row["arm"]!=arm
                or row["prompt"]!=scenario_prompt(reference[sid]["scenario"])
                or row.get("sampling_protocol")!=PROTOCOL
                or int(row["generation_seed"])!=expected_seed
                or int(row["batch_size_requested"])!=requested_batch_size
                or list(row["batch_scenario_ids"])!=ids
                or int(row["max_new_tokens"])!=max_new
                or abs(float(row["strength"])-expected_strength)>1e-7):
                raise ValueError(f"Incompatible batched generation metadata {feature}/{adapter}/{sid}/{arm}")
    return seen


def safe_restart_check(root):
    """Refuse to mix previous sequential outputs with batched ones."""
    for path in root.glob("*__*.jsonl"):
        if (path.name.endswith("__unsteered.jsonl")
            or path.name.endswith("__steered.jsonl")
            or path.name.endswith("__paired.partial.jsonl")):
            # Some files are partial; the first row is sufficient for protocol check.
            with path.open(encoding="utf-8") as f:
                first=next((json.loads(line) for line in f if line.strip()),None)
            if first is None or (first.get("sampling_protocol")!=PROTOCOL
                and first.get("unsteered",{}).get("sampling_protocol")!=PROTOCOL):
                raise FileExistsError(
                    f"Sequential/unknown generation file present: {path}. "
                    "Stop the old process and archive the ENTIRE judged/ directory "
                    "before starting batched generation. Never mix protocols.")


@torch.inference_mode()
def generate_batched(cfg,family,policy="prompt_last"):
    from .judged import selected_conditions

    gen=cfg["steering"]["generation"]
    size=int(gen.get("batch_size",1))
    max_new=int(gen["max_new_tokens"])
    if not 2<=size<=32:
        raise ValueError("Batched generation requires batch_size between 2 and 32")
    if max_new<=0:
        raise ValueError("Invalid generation length")
    scenarios=[x for x in load_screening(cfg) if x["split"]==cfg["steering"]["judged_split"]]
    if len(scenarios)!=int(cfg["steering"]["judged_n"]):
        raise ValueError("Unexpected held-out scenario count")
    groups=list(batches(scenarios,size))
    root=run_dir(cfg,family,"activation-steering")/"judged"
    root.mkdir(parents=True,exist_ok=True)
    safe_restart_check(root)
    selected=selected_conditions(cfg,family,policy)
    by_adapter=defaultdict(list)
    for spec in selected:
        by_adapter[spec["adapter"]].append(spec)
    for adapter,specs in by_adapter.items():
        todo=[]
        for spec in specs:
            stem=f"{spec['feature']}__{adapter}__{spec['policy']}"
            arm_files={arm:root/f"{stem}__{arm}.jsonl" for arm in ("unsteered","steered")}
            partial=root/f"{stem}__paired.partial.jsonl"
            if all(p.exists() for p in arm_files.values()) and not partial.exists():
                left=read_data(arm_files["unsteered"])
                right=read_data(arm_files["steered"])
                if len(left)!=len(scenarios) or len(right)!=len(scenarios):
                    raise ValueError(f"Incomplete previously generated condition {stem}")
                if [x["scenario_id"] for x in left]!=[x["scenario_id"] for x in right]:
                    raise ValueError(f"Misaligned completed arms: {stem}")
                pairs=[{"scenario_id":a["scenario_id"],"unsteered":a,"steered":z}
                       for a,z in zip(left,right)]
                validate_batch_pairs(pairs,scenarios,spec["feature"],adapter,
                                     spec["policy"],spec["strength"],max_new,size,gen["seed"])
                print(f"Skipping verified batch generation {stem}",flush=True)
                continue
            if any(p.exists() for p in arm_files.values()) and not partial.exists():
                raise FileExistsError(f"One arm is missing without a pair checkpoint: {stem}")
            todo.append((spec,arm_files,partial))
        if not todo:continue
        getters={x["feature"]:load_direction_bundle(cfg,family,x["feature"]) for x,_,_ in todo}
        model,tok=load_model(cfg,family,adapter)
        try:
            for spec,arm_files,partial in todo:
                feature=spec["feature"]
                unit,scale=getters[feature](spec["layer"],spec["site"])
                saved=read_data(partial) if partial.exists() else []
                complete=validate_batch_pairs(
                    saved,scenarios,feature,adapter,spec["policy"],spec["strength"],
                    max_new,size,gen["seed"])
                for group in groups:
                    ids=[int(x["scenario_id"]) for x in group]
                    used=[sid in complete for sid in ids]
                    if all(used):continue
                    if any(used):
                        raise ValueError(
                            f"Partial batched scenario group {ids}; refusing to change "
                            "the random-sampling protocol on resume")
                    prompts=[scenario_prompt(x["scenario"]) for x in group]
                    tokens=encode(tok,prompts,model.device,int(cfg["runtime"]["max_length"]))
                    prompt_length=tokens["input_ids"].shape[1]
                    seed=batch_seed(gen["seed"],group)
                    results={}
                    for arm in ("unsteered","steered"):
                        strength=0. if arm=="unsteered" else float(spec["strength"])
                        torch.manual_seed(seed)
                        with steer(model,int(spec["layer"]),spec["site"],unit*scale,
                                   strength,spec["policy"]):
                            output=model.generate(
                                **tokens,max_new_tokens=max_new,
                                do_sample=gen["do_sample"],
                                temperature=gen["temperature"],
                                top_p=gen["top_p"],
                                pad_token_id=tok.pad_token_id)
                        results[arm]=decode_generated_batch(output,prompt_length,tok,model,max_new)
                        if len(results[arm])!=len(group):
                            raise RuntimeError("Inconsistent batched output size")
                    for index,item in enumerate(group):
                        sid=int(item["scenario_id"])
                        arms={}
                        for arm in ("unsteered","steered"):
                            arms[arm]={
                                "scenario_id":sid,"feature":feature,"adapter":adapter,
                                "layer":spec["layer"],"site":spec["site"],
                                "arm":arm,
                                "strength":0. if arm=="unsteered" else float(spec["strength"]),
                                "policy":spec["policy"],
                                "condition":f"{feature}__{adapter}__{spec['policy']}",
                                "prompt":prompts[index],
                                "generation_seed":seed,
                                "sampling_protocol":PROTOCOL,
                                "batch_size_requested":size,
                                "batch_scenario_ids":ids,
                                **results[arm][index]
                            }
                        saved.append({"scenario_id":sid,**arms})
                        complete.add(sid)
                    # Atomic write after both arms of the entire batch: safely resumable.
                    write_rows(partial,saved)
                    print(f"Generated {feature}/{adapter}: {len(saved)}/{len(scenarios)} pairs",flush=True)
                validate_batch_pairs(saved,scenarios,feature,adapter,spec["policy"],
                                     spec["strength"],max_new,size,gen["seed"])
                if len(saved)!=len(scenarios):
                    raise ValueError(f"Incomplete generation {feature}/{adapter}")
                for arm,path in arm_files.items():
                    expected=[x[arm] for x in saved]
                    if path.exists():
                        if read_data(path)!=expected:
                            raise ValueError(f"Conflicting existing arm: {path}")
                    else:
                        write_rows(path,expected)
                partial.unlink(missing_ok=True)
                print(f"Finished {feature}/{adapter}",flush=True)
        finally:unload(model)
    return root
