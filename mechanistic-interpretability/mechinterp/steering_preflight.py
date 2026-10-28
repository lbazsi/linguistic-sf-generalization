"""Generation-quality preflight on steering VALIDATION scenarios only.

Tests every selected feature-adapter pair at its selected layer/site/sign/dose.
Does not open the welfare TEST partition and does not use external judges.
"""
import argparse
import json
import torch
from collections import defaultdict
from pathlib import Path
from .common import load_config,load_screening,run_dir,scenario_prompt,write_rows,write_json
from .models import load_model,unload,encode,steer
from .directions import load_direction_bundle


def termination(output,prompt_length,tok,model,max_new_tokens):
    tail=output[0,prompt_length:]
    eos=model.generation_config.eos_token_id
    if eos is None:eos=tok.eos_token_id
    eos={int(v) for v in (eos if isinstance(eos,(tuple,list)) else [eos]) if v is not None}
    ended=bool(len(tail) and int(tail[-1]) in eos)
    return {"generated_tokens":int(len(tail)),
            "ended_with_eos":ended,
            "reached_token_limit":bool(len(tail)>=max_new_tokens and not ended),
            "continuation":tok.decode(tail,skip_special_tokens=True).strip()}


@torch.inference_mode()
def preflight(cfg,family,n=3):
    if not 1<=n<=10:raise ValueError("n must be 1-10")
    root=run_dir(cfg,family,"activation-steering")
    selected=json.loads((root/"selected_all.json").read_text())
    scenarios=sorted((r for r in load_screening(cfg) if r["split"]=="validation"),
                     key=lambda x:x["scenario_id"])[:n]
    if len(scenarios)!=n:raise ValueError("Insufficient validation scenarios")
    output=root/"preflight_v3"
    output.mkdir(parents=True,exist_ok=True)
    filename=output/f"preflight_n{n}.jsonl"
    if filename.exists():raise FileExistsError(f"Preserve previous preflight: {filename}")
    grouped=defaultdict(list)
    for selection in selected:grouped[selection["adapter"]].append(selection)
    results=[]
    gen=cfg["steering"]["generation"]
    length=int(gen["max_new_tokens"])
    for adapter,specs in grouped.items():
        bundles={spec["feature"]:load_direction_bundle(cfg,family,spec["feature"])
                 for spec in specs}
        model,tok=load_model(cfg,family,adapter)
        try:
            for spec in specs:
                direction,scale=bundles[spec["feature"]](spec["layer"],spec["site"])
                for s in scenarios:
                    tokens=encode(tok,[scenario_prompt(s["scenario"])],
                                  model.device,cfg["runtime"]["max_length"])
                    prompt_length=tokens["input_ids"].shape[1]
                    seed=int(gen["seed"])+int(s["scenario_id"])
                    for arm in ("unsteered","steered"):
                        strength=0 if arm=="unsteered" else float(spec["strength"])
                        torch.manual_seed(seed)
                        audit={}
                        with steer(model,spec["layer"],spec["site"],direction*scale,
                                   strength,spec["policy"],audit=audit):
                            response=model.generate(
                                **tokens,max_new_tokens=length,
                                do_sample=gen["do_sample"],
                                temperature=gen["temperature"],
                                top_p=gen["top_p"],pad_token_id=tok.pad_token_id)
                        if not audit.get("positions"):
                            raise AssertionError("Steering hook did not execute")
                        decoded=termination(response,prompt_length,tok,model,length)
                        results.append({
                            "feature":spec["feature"],"adapter":adapter,
                            "scenario_id":s["scenario_id"],"arm":arm,
                            "policy":spec["policy"],"layer":spec["layer"],
                            "site":spec["site"],"strength":strength,
                            "generation_seed":seed,
                            "prompt":scenario_prompt(s["scenario"]),
                            "calls":audit["calls"],"positions":audit["positions"],
                            "realized_relative_norm":audit["relative_sum"]/audit["positions"],
                            **decoded})
                        if len(results)%10==0:
                            write_rows(filename,results)
                print(f"Preflight: {spec['feature']} / {adapter} completed",flush=True)
        finally:unload(model)
    write_rows(filename,results)
    if len(results)!=len(selected)*n*2:raise RuntimeError("Preflight output count mismatch")
    metrics={}
    for arm in ("unsteered","steered"):
        part=[r for r in results if r["arm"]==arm]
        metrics[arm]={
            "n":len(part),
            "reached_token_limit":sum(r["reached_token_limit"] for r in part),
            "empty":sum(not r["continuation"] for r in part),
            "without_terminal_punctuation":sum(
                not r["continuation"].endswith((".","!","?","。","！","？"))
                for r in part),
        }
    write_json(output/f"preflight_n{n}_summary.json",{
        "family":family,"n_scenarios_per_configuration":n,
        "n_configurations":len(selected),"split":"welfare_validation",
        "max_new_tokens":length,"metrics":metrics,
        "interpretation":"Exploratory quality check only: does not justify a causal or confirmatory effect."})
    print("Preflight complete:",filename)
    print("Quality metrics:",metrics)
    return filename


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,choices=("gemma2","llama31","mistral","qwen3"))
    p.add_argument("--config",default="config/pipeline.yaml")
    p.add_argument("--n",type=int,default=3)
    args=p.parse_args()
    preflight(load_config(args.config),args.model,args.n)

if __name__=="__main__":main()
