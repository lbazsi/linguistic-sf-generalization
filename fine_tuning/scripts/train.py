from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from huggingface_hub import model_info
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from common import (
    DEFAULT_CONFIG_PATH,
    choose_max_length,
    feature_files,
    feature_index,
    length_stats,
    load_canonical,
    load_config,
    load_feature,
    output_directory,
    resolve_data_root,
    sha256_file,
    split_rows,
    stratified_validation_ids,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train canonical or linguistic-feature bf16 LoRA adapters."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--canonical", action="store_true", help="Train the canonical adapter.")
    group.add_argument("--feature", help="Train one feature adapter by dataset filename stem.")
    group.add_argument(
        "--all",
        action="store_true",
        help="Train canonical and every final feature dataset sequentially.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--data-root",
        help="Override data.root from training.yaml; must contain canonical/ and final/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run directory for the same condition and seed.",
    )
    return parser.parse_args()


class MidpointAdapterCallback(TrainerCallback):
    def __init__(self, output_path: Path, enabled: bool) -> None:
        self.output_path = output_path
        self.enabled = enabled
        self.saved = False
        self.saved_step: int | None = None

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled or self.saved or state.max_steps <= 0:
            return control

        midpoint = math.ceil(state.max_steps / 2)
        if state.global_step >= midpoint:
            model = kwargs["model"]
            self.output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(self.output_path, safe_serialization=True)
            self.saved = True
            self.saved_step = int(state.global_step)
        return control


def resolve_base_revision(model_name: str, requested_revision: str | None) -> str:
    info = model_info(model_name, revision=requested_revision or "main")
    if not info.sha:
        raise RuntimeError(f"Could not resolve a commit SHA for {model_name}.")
    return info.sha


def token_lengths(tokenizer, texts: list[str]) -> list[int]:
    lengths: list[int] = []
    eos_id = tokenizer.eos_token_id
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids.append(eos_id)
        lengths.append(len(ids))
    return lengths


def encode_rows(
    tokenizer,
    rows: list[dict[str, Any]],
    text_field: str,
    max_length: int,
) -> tuple[Dataset, int]:
    encoded: list[dict[str, Any]] = []
    truncated = 0
    eos_id = tokenizer.eos_token_id

    for row in rows:
        ids = tokenizer.encode(row[text_field], add_special_tokens=True)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids.append(eos_id)
        if len(ids) > max_length:
            truncated += 1
            ids = ids[:max_length]
        encoded.append(
            {
                "id": int(row["id"]),
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
            }
        )

    return Dataset.from_list(encoded), truncated


def verification_snapshot(
    *,
    condition: str,
    canonical_count: int,
    dataset_count: int,
    train_count: int,
    validation_count: int,
    validation_size_expected: int,
    metadata_matches_canonical: bool,
) -> dict[str, Any]:
    checks = {
        "dataset_count_matches_canonical": dataset_count == canonical_count,
        "train_plus_validation_matches_dataset": train_count + validation_count == dataset_count,
        "validation_size_matches_config": validation_count == validation_size_expected,
        "metadata_matches_canonical": metadata_matches_canonical,
        "condition_is_known": condition in {"canonical", "feature"},
    }
    return {"passed": all(checks.values()), "checks": checks}


def train_one(
    *,
    config: dict[str, Any],
    data_root: Path,
    condition: str,
    feature_name: str | None,
    overwrite: bool,
    resolved_revision: str,
) -> Path:
    training_cfg = config["training"]
    model_cfg = config["model"]
    sequence_cfg = config["sequence"]
    lora_cfg = config["lora"]

    seed = int(training_cfg["seed"])
    set_seed(seed)

    canonical_path, canonical_rows, canonical_by_id = load_canonical(data_root)
    validation_ids = stratified_validation_ids(
        canonical_rows,
        int(config["data"]["validation_size"]),
        seed,
    )
    validation_id_set = set(validation_ids)

    if condition == "canonical":
        dataset_path = canonical_path
        rows = canonical_rows
        text_field = "canonical"
        feature_number = None
        metadata_matches = True
    else:
        if feature_name is None:
            raise ValueError("feature_name is required for feature training.")
        dataset_path, rows, _ = load_feature(data_root, feature_name, canonical_by_id)
        text_field = "feature_variant"
        feature_number = feature_index(data_root, feature_name)
        metadata_matches = True

    train_rows, validation_rows = split_rows(rows, validation_id_set)
    run_dir = output_directory(
        config,
        seed=seed,
        condition=condition,
        feature_name=feature_name,
        feature_number=feature_number,
    )

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{run_dir} already exists. Use --overwrite to replace this run."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        revision=resolved_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    all_texts = [row[text_field] for row in rows]
    lengths = token_lengths(tokenizer, all_texts)
    max_length = choose_max_length(lengths, sequence_cfg, tokenizer.model_max_length)
    num_truncated_total = sum(length > max_length for length in lengths)
    if num_truncated_total:
        warnings.warn(
            f"{num_truncated_total}/{len(lengths)} examples exceed max_length={max_length} "
            "and will be truncated. Training will continue.",
            stacklevel=2,
        )

    train_dataset, train_truncated = encode_rows(
        tokenizer, train_rows, text_field, max_length
    )
    validation_dataset, validation_truncated = encode_rows(
        tokenizer, validation_rows, text_field, max_length
    )

    model_kwargs: dict[str, Any] = {
        "revision": resolved_revision,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    attn_implementation = model_cfg.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **model_kwargs)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg["bias"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    if trainable_params <= 0:
        raise RuntimeError("LoRA configuration produced zero trainable parameters.")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=int(sequence_cfg.get("pad_to_multiple_of", 8)),
    )

    trainer_scratch = run_dir / ".trainer_tmp"
    args = TrainingArguments(
        output_dir=str(trainer_scratch),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=str(training_cfg["lr_scheduler_type"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        weight_decay=float(training_cfg["weight_decay"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        max_grad_norm=float(training_cfg["max_grad_norm"]),
        gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        tf32=bool(training_cfg.get("tf32", True)),
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=int(training_cfg["logging_steps"]),
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        remove_unused_columns=True,
    )

    midpoint_callback = MidpointAdapterCallback(
        run_dir / "midpoint_adapter",
        bool(config["output"].get("save_midpoint_adapter", True)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        callbacks=[midpoint_callback],
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_adapter = run_dir / "adapter"
    model.save_pretrained(final_adapter, safe_serialization=True)

    verification = verification_snapshot(
        condition=condition,
        canonical_count=len(canonical_rows),
        dataset_count=len(rows),
        train_count=len(train_rows),
        validation_count=len(validation_rows),
        validation_size_expected=int(config["data"]["validation_size"]),
        metadata_matches_canonical=metadata_matches,
    )
    if not verification["passed"]:
        raise RuntimeError(f"Post-training verification failed: {verification}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "feature": feature_name,
        "feature_index": feature_number,
        "seed": seed,
        "base_model": {
            "name": model_cfg["name"],
            "requested_revision": model_cfg.get("revision"),
            "resolved_revision": resolved_revision,
        },
        "dataset": {
            "canonical_path": str(canonical_path),
            "canonical_sha256": sha256_file(canonical_path),
            "training_path": str(dataset_path),
            "training_sha256": sha256_file(dataset_path),
            "text_field": text_field,
            "total_examples": len(rows),
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "validation_ids": validation_ids,
        },
        "sequence": {
            "packing": False,
            "max_length": max_length,
            "selection": "configured" if sequence_cfg.get("max_length") is not None else "automatic",
            "length_stats_before_truncation": length_stats(lengths),
            "truncated_total": num_truncated_total,
            "truncated_train": train_truncated,
            "truncated_validation": validation_truncated,
        },
        "lora": {
            "r": int(lora_cfg["r"]),
            "alpha": int(lora_cfg["alpha"]),
            "dropout": float(lora_cfg["dropout"]),
            "bias": str(lora_cfg["bias"]),
            "target_modules": list(lora_cfg["target_modules"]),
            "trainable_parameters": trainable_params,
            "total_base_plus_adapter_parameters": total_params,
        },
        "training": {
            "num_train_epochs": float(training_cfg["num_train_epochs"]),
            "learning_rate": float(training_cfg["learning_rate"]),
            "lr_scheduler_type": str(training_cfg["lr_scheduler_type"]),
            "warmup_ratio": float(training_cfg["warmup_ratio"]),
            "weight_decay": float(training_cfg["weight_decay"]),
            "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
            "effective_batch_size_per_gpu": (
                int(training_cfg["per_device_train_batch_size"])
                * int(training_cfg["gradient_accumulation_steps"])
            ),
            "max_grad_norm": float(training_cfg["max_grad_norm"]),
            "gradient_checkpointing": bool(training_cfg["gradient_checkpointing"]),
            "train_loss": train_result.metrics.get("train_loss"),
            "eval_loss": eval_metrics.get("eval_loss"),
            "optimizer_steps": int(trainer.state.global_step),
        },
        "artifacts": {
            "midpoint_adapter": (
                "midpoint_adapter" if midpoint_callback.saved else None
            ),
            "midpoint_step": midpoint_callback.saved_step,
            "final_adapter": "adapter",
        },
        "verification": verification,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu_name,
        },
    }
    write_json(run_dir / "training_summary.json", summary)

    if trainer_scratch.exists():
        shutil.rmtree(trainer_scratch)

    print(json.dumps({
        "run_dir": str(run_dir),
        "condition": condition,
        "feature": feature_name,
        "train_loss": summary["training"]["train_loss"],
        "eval_loss": summary["training"]["eval_loss"],
        "truncated_total": num_truncated_total,
        "max_length": max_length,
    }, indent=2))
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(config, args.data_root)

    model_name = config["model"]["name"]
    requested_revision = config["model"].get("revision")
    resolved_revision = resolve_base_revision(model_name, requested_revision)
    print(f"Resolved {model_name}@{requested_revision} -> {resolved_revision}")

    if args.all:
        train_one(
            config=config,
            data_root=data_root,
            condition="canonical",
            feature_name=None,
            overwrite=args.overwrite,
            resolved_revision=resolved_revision,
        )
        for path in feature_files(data_root):
            train_one(
                config=config,
                data_root=data_root,
                condition="feature",
                feature_name=path.stem,
                overwrite=args.overwrite,
                resolved_revision=resolved_revision,
            )
        return

    if args.canonical:
        train_one(
            config=config,
            data_root=data_root,
            condition="canonical",
            feature_name=None,
            overwrite=args.overwrite,
            resolved_revision=resolved_revision,
        )
    else:
        train_one(
            config=config,
            data_root=data_root,
            condition="feature",
            feature_name=args.feature,
            overwrite=args.overwrite,
            resolved_revision=resolved_revision,
        )


if __name__ == "__main__":
    main()
