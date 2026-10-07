from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from common import (
    batches,
    ensure_directories,
    load_config,
    read_jsonl,
    render_prompt,
    resolve_path,
    write_jsonl_atomic,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate continuations for all evaluation scenarios.")
    parser.add_argument(
        "--condition",
        action="append",
        help="Evaluate only the named condition (base, canonical, or feature name); repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing response files.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = resolve_path(config, "fine_tuning_outputs")
    seed = int(config["model_generation"]["training_seed"])

    canonical_dir = root / "canonical" / f"seed_{seed}"
    canonical_summary_path = canonical_dir / "training_summary.json"
    if not canonical_summary_path.exists():
        raise FileNotFoundError(
            f"Missing canonical training summary: {canonical_summary_path}"
        )
    canonical_summary = load_summary(canonical_summary_path)
    base = canonical_summary["base_model"]

    conditions = [
        {
            "key": "base",
            "condition": "base",
            "feature": None,
            "comparison_control": None,
            "adapter": None,
            "base_model": base,
        },
        {
            "key": "canonical",
            "condition": "canonical",
            "feature": None,
            "comparison_control": None,
            "adapter": canonical_dir / "adapter",
            "base_model": base,
        },
    ]

    controls_root = root / "controls"
    if controls_root.exists():
        for language_dir in sorted(path for path in controls_root.iterdir() if path.is_dir()):
            run_dir = language_dir / f"seed_{seed}"
            summary_path = run_dir / "training_summary.json"
            if not summary_path.exists():
                continue
            summary = load_summary(summary_path)
            if summary["base_model"]["resolved_revision"] != base["resolved_revision"]:
                raise RuntimeError(f"{summary_path}: base-model revision differs from canonical run.")
            conditions.append(
                {
                    "key": f"control_{language_dir.name}",
                    "condition": "control",
                    "feature": None,
                    "comparison_control": None,
                    "adapter": run_dir / "adapter",
                    "base_model": summary["base_model"],
                }
            )

    feature_root = root / "features"
    if feature_root.exists():
        for feature_dir in sorted(path for path in feature_root.iterdir() if path.is_dir()):
            run_dir = feature_dir / f"seed_{seed}"
            summary_path = run_dir / "training_summary.json"
            if not summary_path.exists():
                continue
            summary = load_summary(summary_path)
            if summary["base_model"]["resolved_revision"] != base["resolved_revision"]:
                raise RuntimeError(
                    f"{summary_path}: base-model revision differs from canonical run."
                )
            feature = summary.get("feature")
            if not feature:
                raise RuntimeError(f"{summary_path}: missing feature name.")
            conditions.append(
                {
                    "key": feature,
                    "condition": "feature",
                    "feature": feature,
                    "comparison_control": summary.get("comparison_control") or "canonical",
                    "adapter": run_dir / "adapter",
                    "base_model": summary["base_model"],
                }
            )
    return conditions


def generate_condition(
    *,
    config: dict[str, Any],
    scenarios: list[dict[str, Any]],
    spec: dict[str, Any],
    overwrite: bool,
) -> None:
    output_path = resolve_path(config, "responses") / f"{spec['key']}.jsonl"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    base = spec["base_model"]
    generation = config["model_generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        base["name"],
        revision=base["resolved_revision"],
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base["name"],
        revision=base["resolved_revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=str(generation["attn_implementation"]),
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if spec["adapter"] is not None:
        if not Path(spec["adapter"]).is_dir():
            raise FileNotFoundError(spec["adapter"])
        model = PeftModel.from_pretrained(model, spec["adapter"])
    model.eval()

    rows: list[dict[str, Any]] = []
    batch_size = int(generation["batch_size"])
    base_seed = int(config["seeds"]["response_generation"])

    for batch_index, batch in enumerate(batches(scenarios, batch_size)):
        prompts = [
            render_prompt("model_completion.txt", {"SCENARIO": row["scenario"]})
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        set_seed(base_seed + batch_index)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=bool(generation["do_sample"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_width = encoded["input_ids"].shape[1]
        continuations = tokenizer.batch_decode(
            output[:, input_width:],
            skip_special_tokens=True,
        )

        for scenario, prompt, continuation in zip(batch, prompts, continuations):
            rows.append(
                {
                    "scenario_id": scenario["id"],
                    "condition": spec["condition"],
                    "feature": spec["feature"],
                    "comparison_control": spec.get("comparison_control"),
                    "training_seed": int(generation["training_seed"]),
                    "base_model_name": base["name"],
                    "base_model_revision": base["resolved_revision"],
                    "prompt": prompt,
                    "continuation": continuation.strip(),
                }
            )

    write_jsonl_atomic(output_path, rows)
    print(f"[{spec['key']}] wrote {len(rows)} responses to {output_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)
    scenarios = read_jsonl(resolve_path(config, "final_scenarios"))
    conditions = discover_conditions(config)

    requested = set(args.condition or [])
    if requested:
        available = {spec["key"] for spec in conditions}
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(f"Unknown conditions: {', '.join(missing)}")
        conditions = [spec for spec in conditions if spec["key"] in requested]

    output_paths = {}
    for spec in conditions:
        generate_condition(
            config=config,
            scenarios=scenarios,
            spec=spec,
            overwrite=args.overwrite,
        )
        output_paths[spec["key"]] = resolve_path(config, "responses") / f"{spec['key']}.jsonl"

    write_manifest(
        config,
        stage="03_generate_responses",
        inputs={"final_scenarios": resolve_path(config, "final_scenarios")},
        outputs=output_paths,
        prompt_files=["model_completion.txt"],
        extra={
            "evaluated_conditions": [spec["key"] for spec in conditions],
            "local_generation": True,
        },
    )


if __name__ == "__main__":
    main()
