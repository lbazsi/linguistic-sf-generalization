#!/usr/bin/env python3
"""Export only completed, small mechanistic results into the repository.

Source: mechanistic-interpretability/outputs/<model>/
Destination: results/mechanistic-interpretability/<model>/
No temporary checkpoints, archived experiments, raw generations, judge-level
responses, cached artifacts, pilot experiments or heavy direction tensors.
Never deletes or overwrites a destination unless explicitly requested.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
STAGES=("model-difference-analysis","linear-probing","LoRA-analysis","activation-steering")
ALLOW_ROOT={
    "model-difference-analysis":("*.jsonl","*.json"),
    "linear-probing":("*.jsonl","*.json"),
    "LoRA-analysis":("*.jsonl","*.json"),
}
STEERING_FILES={
    "selected_all.json", "selected_validation_audit.json",
    "selected_controls.jsonl", "site_calibration_v2.json", "workload_all.json",
    "generation_quality.json",
}
JUDGMENT_FILES={"summary.json","api_usage.json"}


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def choose(source):
    selected=[]
    for stage in STAGES[:3]:
        folder=source/stage
        if not folder.exists():
            continue
        for pat in ALLOW_ROOT[stage]:
            selected.extend(p for p in folder.glob(pat) if p.is_file()
                            and not p.name.endswith(".partial"))
    steer=source/"activation-steering"
    if steer.exists():
        selected.extend(p for p in steer.iterdir() if p.is_file() and
                        (p.name.startswith("screen_") and p.suffix==".jsonl" or
                         p.name.startswith("directions_") and p.name.endswith("_metadata.json") or
                         p.name in STEERING_FILES or
                         p.name in {f"{Path(x).stem}_manifest.json" for x in STEERING_FILES} or
                         (p.name.startswith("screen_") and p.name.endswith("_manifest.json"))))
        jud=steer/"judgments"
        selected.extend(jud/p for p in JUDGMENT_FILES if (jud/p).is_file())
        # Final aggregated paired results, not 5,600 prompt/response source files.
        if (jud/"combined.jsonl").is_file():
            selected.append(jud/"combined.jsonl")
        if (jud/"combined_manifest.json").is_file():
            selected.append(jud/"combined_manifest.json")
    plots=source/"plots"
    if plots.exists():
        selected.extend(p for p in plots.glob("*.png") if p.is_file())
    return sorted(set(selected))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,choices=("llama31","gemma2","qwen3","mistral"))
    p.add_argument("--source",type=Path)
    p.add_argument("--destination",type=Path)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args()
    source=(args.source or ROOT/"outputs"/args.model).resolve()
    destination=(args.destination or REPO/"results"/"mechanistic-interpretability"/args.model).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    picks=choose(source)
    if not picks:raise RuntimeError("No completed artifacts selected")
    for item in picks:
        relative=item.relative_to(source)
        target=destination/relative
        if target.exists() and sha(target)!=sha(item) and not args.overwrite:
            raise FileExistsError(f"Conflicting result (use --overwrite explicitly): {target}")
    print(f"Exporting {len(picks)} final scalar and figure artifacts from {source} to {destination}")
    print("Excluded: raw judged generations, per-judge files, partial results, checkpoints, direction tensors and pilots")
    for item in picks:
        target=destination/item.relative_to(source)
        print("  "+str(item.relative_to(source)))
        if args.execute:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or sha(target)!=sha(item):
                shutil.copy2(item,target)
    print("Completed." if args.execute else "Dry run only. Add --execute to copy.")


if __name__=="__main__":
    main()
