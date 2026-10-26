"""Explicit validation-only steering pilot; no judges or heldout-test access."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from .common import load_config, load_screening, run_dir, scenario_prompt, write_rows, write_json
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
        finally:unload(model)
    path=out/"pilot.jsonl"
    write_rows(path,rows)
    write_json(out/"manifest.json",{"version":2,"n_validation":len(val),
        "features":features,"fractions":FRACTIONS,"heldout_test_used":False,
        "interpretation":"Pilot scores are welfare-option proxies, not verified linguistic manipulation. Human/feature-specific manipulation checks are mandatory before expansion."})
    return path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=("gemma2","qwen3","llama31","mistral"),required=True)
    p.add_argument("--config",default="config/pipeline.yaml")
    p.add_argument("--n",type=int,default=20)
    args=p.parse_args()
    print(run_pilot(load_config(args.config),args.model,n=args.n))
if __name__=="__main__":main()
