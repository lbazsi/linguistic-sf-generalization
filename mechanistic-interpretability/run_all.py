#!/usr/bin/env python3
"""Explicit orchestration of local (non-judge) analysis stages.

Default is a task listing; --execute launches experiments sequentially.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from mechinterp.common import FEATURES,control_for,load_config,location

def plan(model,stage):
    if stage=="difference":
        return [{"stage":"difference","model":model,"condition":x}
                for x in ("canonical","control_zh",*FEATURES)]
    if stage=="behavior-probes":
        return [{"stage":"probe","model":model,"task":"behavior","condition":condition}
                for condition in ("base","canonical","control_zh",*FEATURES)]
    if stage=="functional":
        return [{"stage":"lora-functional","model":model,"checkpoint":x}
                for x in ("midpoint","final")]
    if stage=="probes":
        tasks=[]
        for condition in ("base","canonical","control_zh",*FEATURES):
            tasks.append({"stage":"probe","model":model,"task":"concept","condition":condition})
            tasks.append({"stage":"probe","model":model,"task":"behavior","condition":condition})
        # Linguistic probes compare the same feature over base, its control and
        # its own feature adapter, not all unrelated feature-specific adapters.
        for feature in FEATURES:
            for condition in dict.fromkeys(("base",control_for(feature),feature)):
                tasks.append({"stage":"probe","model":model,"task":"linguistic",
                              "condition":condition,"feature":feature})
        return tasks
    if stage=="lora":
        return [{"stage":"lora","model":model},
                {"stage":"lora-functional","model":model,"checkpoint":"midpoint"},
                {"stage":"lora-functional","model":model,"checkpoint":"final"}]
    if stage=="steering":
        return [{"stage":"steering-screen","model":model,
                 "feature":f,"condition":c,"policy":p}
                for f in FEATURES for c in (control_for(f),f)
                for p in ("prompt_last","generation")]
    raise ValueError(stage)

def execute(cfg,task):
    from mechinterp import experiments, steering
    from mechinterp.functional import functional_analysis
    model=task["model"];stage=task["stage"]
    if stage=="difference":return experiments.difference(cfg,model,task["condition"])
    if stage=="probe":
        return experiments.probing(cfg,model,task["condition"],task["task"],task.get("feature"))
    if stage=="lora":return experiments.lora_analysis(cfg,model)
    if stage=="lora-functional":return functional_analysis(cfg,model,task["checkpoint"])
    if stage=="steering-screen":
        return steering.screen(cfg,model,task["feature"],task["condition"],task["policy"])
    raise ValueError(stage)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=("gemma2","qwen3","llama31","mistral"),required=True)
    parser.add_argument("--stage",choices=("difference","probes","behavior-probes","functional","lora","steering"),required=True)
    parser.add_argument("--config",default="config/pipeline.yaml")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    cfg=load_config(args.config)
    tasks=plan(args.model,args.stage)
    print(f"Planned {len(tasks)} local tasks; remote judge calls: 0")
    for task in tasks:
        print(task)
        if not args.execute:continue
        if args.resume:
            outputs=location(cfg,"outputs")/args.model
            if task["stage"]=="difference":
                candidate=outputs/"model-difference-analysis"/f"{task['condition']}_final.jsonl"
            elif task["stage"]=="probe":
                candidate=outputs/"linear-probing"/f"{task['task']}_{task.get('feature') or 'all'}_{task['condition']}.jsonl"
            elif task["stage"]=="lora":candidate=outputs/"LoRA-analysis"/"module_geometry.jsonl"
            elif task["stage"]=="lora-functional":candidate=outputs/"LoRA-analysis"/("functional_"+task["checkpoint"]+".jsonl")
            else:
                candidate=outputs/"activation-steering"/f"screen_{task['feature']}_{task['condition']}_{task['policy']}.jsonl"
            if candidate.exists():
                print(f"Skipping existing {candidate}")
                continue
        result=execute(cfg,task)
        if isinstance(result,(str,Path)) and Path(result).is_file():
            from mechinterp.common import record_run
            record_run(cfg,args.model,task["stage"],Path(result))

if __name__=="__main__":main()
