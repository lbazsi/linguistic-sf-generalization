"""Explicit validation-only steering pilot; no judges or heldout-test access."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import torch
from .common import load_config, load_screening, run_dir, scenario_prompt, write_rows, write_json, feature_pairs, split_groups
from .models import load_model, unload, steer, encode, continuation_scores
from .directions import load_direction_bundle

DEFAULT_FEATURES=("epistemic_hedging","nominalization","narrative_framing","constituent_order")
FRACTIONS=(0.,.03,.1,.25)

def relative_to_strength(fraction, width):
    if fraction<0 or fraction>.5:raise ValueError("Intervention fraction out of range")
    return float(fraction)*math.sqrt(width)

@torch.inference_mode()
def run_pilot(cfg,family,features=DEFAULT_FEATURES,n=20):
    legacy=run_dir(cfg,family,"activation-steering").parent.parent.parent/family/"activation-steering"/"selected_all.json"
    if not legacy.is_file():
        # External restored outputs may contain archived legacy screen selections.
        legacy=run_dir(cfg,family,"activation-steering")/"selected_all.json"
    if not legacy.is_file():
        raise FileNotFoundError("Restore original selected_all.json to the original output tree")
    candidates=json.loads(legacy.read_text())
    selected=[s for s in candidates if s["feature"] in features and s["adapter"]==s["feature"]]
    if len(selected)!=len(features):raise ValueError("Missing pilot configuration for feature adapter")
    val=sorted([v for v in load_screening(cfg) if v["split"]=="validation"],key=lambda x:x["scenario_id"])[:n]
    if len(val)!=n:raise ValueError("Insufficient validation scenarios")
    out=run_dir(cfg,family,"activation-steering")/"pilot"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for spec in selected:
        getter=load_direction_bundle(cfg,family,spec["feature"])
        unit,scale=getter(spec["layer"],spec["site"])
        if abs(float(unit.norm())-1)>1e-3:raise ValueError("Nonunit direction")
        model,tok=load_model(cfg,family,spec["adapter"])
        try:
            gen=cfg["steering"]["generation"]
            prompts=[scenario_prompt(r["scenario"]) for r in val]
            yes=[r["welfare_supporting"] for r in val]
            no=[r["welfare_opposing"] for r in val]
            # Independent linguistic-ID split; contrast directions use TRAIN
            # matched pairs and this diagnostic uses TEST matched pairs only.
            raw=feature_pairs(cfg,spec["feature"])
            split=split_groups((p["id"] for p in raw),int(cfg["seed"]))
            held=[p for p in raw if split[p["id"]]=="test"]
            random.Random(int(cfg["seed"])).shuffle(held)
            neutral="Complete the following passage naturally:\n"
            max_len=int(cfg["runtime"]["max_length"])
            held=[p for p in held
                  if len(tok(neutral,add_special_tokens=True)["input_ids"])+
                     max(len(tok(p[k],add_special_tokens=False)["input_ids"])
                         for k in ("canonical","feature_variant"))<=max_len][:n]
            if len(held)<min(10,n):
                raise ValueError("Insufficient held-out linguistic pairs within token budget")
            lp=[neutral]*len(held)
            lv=[p["feature_variant"] for p in held]
            lc=[p["canonical"] for p in held]
            for fraction in FRACTIONS:
                for sign in ((-1,1) if fraction else (0,)):
                    alpha=sign*relative_to_strength(fraction,unit.numel())
                    scores=continuation_scores(model,tok,prompts,yes,no,
                        spec["layer"],spec["site"],unit*scale,alpha,spec["policy"],
                        int(cfg["runtime"]["max_length"]))
                    for row,score in zip(val,scores):
                        rows.append({"feature":spec["feature"],"adapter":spec["adapter"],
                            "scenario_id":row["scenario_id"],"fraction":fraction,
                            "alpha":alpha,"policy":spec["policy"],"kind":"option_gap",
                            "score":score})
                    # This is a paired, length-normalized likelihood proxy;
                    # independent TEST IDs were not used in direction fitting.
                    linguistic=continuation_scores(model,tok,lp,lv,lc,
                        spec["layer"],spec["site"],unit*scale,alpha,spec["policy"],max_len)
                    for pair,score in zip(held,linguistic):
                        rows.append({"feature":spec["feature"],"adapter":spec["adapter"],
                            "pair_id":pair["id"],"fraction":fraction,"alpha":alpha,
                            "policy":spec["policy"],"kind":"linguistic_variant_log_gap",
                            "score":score,"split":"linguistic_test"})
                    if fraction not in (0.,.1):continue
                    for row,prompt in zip(val,prompts):
                        tokens=encode(tok,[prompt],model.device,cfg["runtime"]["max_length"])
                        seed=int(gen["seed"])+int(row["scenario_id"])
                        torch.manual_seed(seed)
                        audit={}
                        with steer(model,spec["layer"],spec["site"],unit*scale,alpha,
                                   spec["policy"],audit=audit):
                            output=model.generate(**tokens,max_new_tokens=gen["max_new_tokens"],
                                do_sample=gen["do_sample"],temperature=gen["temperature"],
                                top_p=gen["top_p"],pad_token_id=tok.pad_token_id)
                        continuation=tok.decode(output[0,tokens["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
                        if not audit.get("positions"):raise AssertionError("Steering hook did not execute")
                        if abs(audit["delta_sum"]/audit["positions"]-abs(alpha)*scale)>1e-3*max(1.,abs(alpha)*scale):
                            raise AssertionError("Actual intervention magnitude does not match requested norm")
                        rows.append({"feature":spec["feature"],"adapter":spec["adapter"],
                            "scenario_id":row["scenario_id"],"fraction":fraction,
                            "alpha":alpha,"policy":spec["policy"],"kind":"generation",
                            "generation_seed":seed,"prompt":prompt,"continuation":continuation,
                            **audit,"realized_relative_norm":audit["relative_sum"]/audit["positions"]})
            # Norm-matched shuffled-label and repeated independent random
            # direction controls at the 10%-norm dose; do not select using them.
            for index in range(6):
                if index==0:
                    control_direction,_=getter(spec["layer"],spec["site"],"shuffled")
                    control_name="shuffled_labels"
                else:
                    rg=torch.Generator(device="cpu")
                    rg.manual_seed(int(cfg["seed"])+100003*DEFAULT_FEATURES.index(spec["feature"])+
                                   43*spec["layer"]+index)
                    control_direction=torch.randn(unit.shape,generator=rg)
                    control_direction=control_direction/control_direction.norm()
                    control_name=f"random_{index}"
                alpha=relative_to_strength(.1,unit.numel())
                for task,prefixes,positive,negative,ids in (
                    ("welfare_option_gap",prompts,yes,no,[r["scenario_id"] for r in val]),
                    ("linguistic_variant_log_gap",lp,lv,lc,[p["id"] for p in held])):
                    control_scores=continuation_scores(model,tok,prefixes,positive,negative,
                        spec["layer"],spec["site"],control_direction*scale,alpha,
                        spec["policy"],max_len)
                    for sample_id,score in zip(ids,control_scores):
                        rows.append({"feature":spec["feature"],"adapter":spec["adapter"],
                            "sample_id":sample_id,"fraction":.1,"alpha":alpha,
                            "policy":spec["policy"],"kind":"direction_control",
                            "task":task,"control":control_name,"score":score})
        finally:unload(model)
    path=out/"pilot.jsonl"
    write_rows(path,rows)
    write_json(out/"manifest.json",{"version":2,"n_validation":len(val),
        "features":features,"fractions":FRACTIONS,"welfare_heldout_test_used":False,"linguistic_heldout_test_used":True,
        "interpretation":"Pilot scores are welfare-option proxies, not verified linguistic manipulation. The independent paired linguistic contrast is a proxy, not proof of realized grammar. Inspect generated output forms, coherence, language switching, and randomized controls before expansion."})
    return path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=("gemma2","qwen3","llama31","mistral"),required=True)
    p.add_argument("--config",default="config/pipeline.yaml")
    p.add_argument("--n",type=int,default=20)
    args=p.parse_args()
    print(run_pilot(load_config(args.config),args.model,n=args.n))
if __name__=="__main__":main()
