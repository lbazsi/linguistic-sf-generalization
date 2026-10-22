"""Command-line interface. Dangerous judge stage is always a separate explicit command."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
from .common import (FEATURES, load_config,load_concepts,load_general,load_screening,
                     load_scenarios,location,resolve_base,adapter_dir,control_for)

def main():
    p=argparse.ArgumentParser(description="Mechanistic interpretability experiment pipeline")
    p.add_argument("stage",choices=("validate","difference","probe","lora","steering-screen",
                                    "steering-select","steering-controls","lora-functional","synthesis","plots","judge-plan","judge-generate","judge-run",
                                    "judge-aggregate"))
    p.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"config/pipeline.yaml"))
    p.add_argument("--model",choices=("gemma2","qwen3","llama31","mistral"),default="gemma2")
    p.add_argument("--condition",default="canonical")
    p.add_argument("--feature",choices=FEATURES)
    p.add_argument("--task",choices=("linguistic","concept","behavior"),default="concept")
    p.add_argument("--checkpoint",choices=("midpoint","final"),default="final")
    p.add_argument("--policy",choices=("prompt_last","generation"),default="prompt_last")
    p.add_argument("--execute-judges",action="store_true")
    p.add_argument("--check-screening",action="store_true")
    args=p.parse_args()
    cfg=load_config(args.config)
    if args.stage=="validate":
        print("General prompts:",len(load_general(cfg)))
        print("Concept examples:",len(load_concepts(cfg)))
        print("Existing evaluation scenarios:",len(load_scenarios(cfg)))
        if args.check_screening:
            print("Steering examples:",len(load_screening(cfg)))
        print("Validation passed.")
        return
    if args.stage=="difference":
        from .experiments import difference
        result=difference(cfg,args.model,args.condition,args.checkpoint)
    elif args.stage=="probe":
        from .experiments import probing
        result=probing(cfg,args.model,args.condition,args.task,args.feature)
    elif args.stage=="lora":
        from .experiments import lora_analysis
        result=lora_analysis(cfg,args.model)
    elif args.stage=="lora-functional":
        from .functional import functional_analysis
        result=functional_analysis(cfg,args.model,args.checkpoint)
    elif args.stage=="plots":
        from .visualize import plots
        result=plots(cfg,args.model)
    elif args.stage=="synthesis":
        from .synthesis import summarize
        result=summarize(cfg)
    elif args.stage=="steering-screen":
        from .steering import screen
        if not args.feature: p.error("--feature required")
        result=screen(cfg,args.model,args.feature,args.condition,args.policy)
    elif args.stage=="steering-select":
        from .steering import selection
        result=selection(cfg,args.model,policy=args.policy)
    elif args.stage=="steering-controls":
        from .steering import selected_controls
        result=selected_controls(cfg,args.model)
    elif args.stage=="judge-plan":
        from .judged import write_workload
        result=write_workload(cfg,args.model,args.policy)
    elif args.stage=="judge-generate":
        from .judged import generate
        result=generate(cfg,args.model,args.policy)
    elif args.stage=="judge-run":
        if not args.execute_judges:p.error("Remote judge calls require --execute-judges")
        from .judged import judge
        result=asyncio.run(judge(cfg,args.model,args.policy,execute=True))
    elif args.stage=="judge-aggregate":
        from .judged import aggregate
        result=aggregate(cfg,args.model,args.policy)
    if args.stage not in ("judge-run",) and isinstance(result,(str,Path)) and Path(result).is_file():
        from .common import record_run
        record_run(cfg,args.model,args.stage,Path(result))
    print(result)

if __name__=="__main__":
    main()
