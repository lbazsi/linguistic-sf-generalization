"""Configuration, data contracts, split assignment, and audit artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "conditionality", "epistemic_hedging", "formality", "genericity",
    "modal_strength", "narrative_framing", "nominalization",
    "pronoun_distribution", "subordination", "tense_aspect", "voice",
    "constituent_order", "inflectional_synthesis", "fusion",
)
CROSS_LANGUAGE = frozenset(("constituent_order", "inflectional_synthesis", "fusion"))
CATEGORIES = (
    "id", "animal", "value", "context", "animal_value",
    "animal_context", "value_context", "animal_value_context",
)


def control_for(feature: str) -> str:
    if feature not in FEATURES:
        raise ValueError(f"Unregistered linguistic intervention: {feature}")
    return "control_zh" if feature in ("inflectional_synthesis", "fusion") else "canonical"


def read_data(path: Path) -> list[dict]:
    """Read JSON-array files or JSON Lines, independent of filename extension."""
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Empty dataset: {path}")
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"Expected list of objects in {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
    os.replace(temp, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | Path = ROOT / "config/pipeline.yaml") -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = ("models", "paths", "runtime", "probing", "steering", "stats")
    if any(key not in cfg for key in required):
        raise ValueError("Incomplete pipeline configuration")
    cfg["_config_file"] = str(path)
    return cfg


def location(cfg: dict, name: str) -> Path:
    path = Path(cfg["paths"][name])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def adapter_root(cfg: dict, family: str) -> Path:
    path = Path(cfg["models"][family]["adapters"])
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def adapter_dir(cfg: dict, family: str, condition: str, checkpoint: str = "final") -> Path | None:
    if condition == "base":
        return None
    root = adapter_root(cfg, family)
    seed = int(cfg["seed"])
    if condition == "canonical":
        run = root / "canonical" / f"seed_{seed}"
    elif condition.startswith("control_"):
        run = root / "controls" / condition[8:] / f"seed_{seed}"
    else:
        if condition not in FEATURES:
            raise ValueError(f"Unknown condition: {condition}")
        run = root / "features" / condition / f"seed_{seed}"
    sub = "adapter" if checkpoint == "final" else "midpoint_adapter"
    target = run / sub
    if not (target / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing LoRA tensors/config at {target}. Restore the external adapter archive."
        )
    return target


def resolve_base(cfg: dict, family: str) -> tuple[str, str]:
    spec = cfg["models"][family]
    revision = spec.get("revision")
    if not revision:
        raise ValueError(f"Pin an immutable model revision in config for {family}")
    return spec["name"], revision


def verify_adapter_identity(cfg: dict, family: str, condition: str) -> None:
    if condition == "base":
        return
    path = adapter_dir(cfg, family, condition)
    summary = path.parent / "training_summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Missing training provenance: {summary}")
    obj = json.loads(summary.read_text(encoding="utf-8"))
    name, revision = resolve_base(cfg, family)
    base = obj["base_model"]
    if base["name"] != name or base["resolved_revision"] != revision:
        raise ValueError(f"Base-model provenance mismatch: {summary}")


def scenario_prompt(scenario: str) -> str:
    return f"Situation:\n{scenario}\n\nAssessment:\nThe most appropriate response is to"


def batches(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def split_groups(keys: Iterable[Any], seed: int, ratios=(0.70, 0.15)) -> dict[Any, str]:
    keys = sorted(set(keys), key=str)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    train_end = round(n * ratios[0])
    dev_end = round(n * (ratios[0] + ratios[1]))
    return {key: ("train" if i < train_end else "development" if i < dev_end else "test")
            for i, key in enumerate(keys)}


def validate_unique(rows: list[dict], key: str) -> None:
    ids = [str(row[key]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {key} detected")


def load_scenarios(cfg: dict) -> list[dict]:
    rows = read_data(location(cfg, "scenarios"))
    validate_unique(rows, "id")
    return rows


def load_general(cfg: dict) -> list[dict]:
    rows = read_data(location(cfg, "general"))
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 general-domain prompts, got {len(rows)}")
    validate_unique(rows, "id")
    if any(not isinstance(r.get("prompt"), str) or not r["prompt"].strip() for r in rows):
        raise ValueError("General-domain prompt text missing")
    return rows


def load_concepts(cfg: dict) -> list[dict]:
    rows = read_data(location(cfg, "concepts"))
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 concept examples, got {len(rows)}")
    validate_unique(rows, "id")
    if Counter(r.get("label") for r in rows) != {0: 500, 1: 500}:
        raise ValueError("Concept-probe labels must have 500 examples each")
    groups = defaultdict(set)
    for row in rows:
        if not isinstance(row.get("text"), str):
            raise ValueError("Invalid concept text")
        groups[str(row["pair_id"])].add(int(row["label"]))
    if len(groups) != 500 or any(labels != {0, 1} for labels in groups.values()):
        raise ValueError("Concept-probe pairs must contain one positive and one negative")
    return rows


def load_screening(cfg: dict) -> list[dict]:
    path = location(cfg, "steering")
    rows = read_data(path)
    if len(rows) != 200:
        raise ValueError(f"Expected 200 steering-option rows, got {len(rows)}")
    validate_unique(rows, "scenario_id")
    splits = Counter(r["split"] for r in rows)
    if splits != {"screen": 20, "validation": 80, "test": 100}:
        raise ValueError(f"Unexpected steering partitions: {splits}")
    original = {int(r["id"]): r for r in load_scenarios(cfg)}
    for r in rows:
        source = original.get(int(r["scenario_id"]))
        if source is None or r["scenario"] != source["scenario"]:
            raise ValueError("Steering scenario differs from existing benchmark")
        if any(r.get(k) != source.get(k) for k in ("category", "tradeoff")):
            raise ValueError("Steering category/tradeoff mismatch")
        if not all(str(r.get(k, "")).strip() for k in ("welfare_supporting", "welfare_opposing")):
            raise ValueError("Empty steering response option")
    cats = Counter(r["category"] for r in rows)
    if any(cats[cat] != 25 for cat in CATEGORIES):
        raise ValueError("Expected 25 steering-option rows per category")
    return rows


def feature_pairs(cfg: dict, feature: str) -> list[dict]:
    path = location(cfg, "features") / (feature + ".jsonl")
    rows = read_data(path)
    validate_unique(rows, "id")
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("canonical", "feature_variant")):
            raise ValueError(f"Missing texts for {feature}, ID {row.get('id')}")
    return rows


def model_conditions() -> list[str]:
    return ["base", "canonical", "control_zh", *FEATURES]


def run_dir(cfg: dict, family: str, stage: str) -> Path:
    out = location(cfg, "outputs") / family / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def provenance(cfg: dict, family: str, stage: str, inputs: list[Path]) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "family": family,
        "model": cfg["models"][family],
        "seed": cfg["seed"],
        "config_sha256": digest(Path(cfg["_config_file"])),
        "input_hashes": {str(p): digest(p) for p in inputs if p.is_file()},
    }

def record_run(cfg, family, stage, artifact: Path) -> Path:
    """Write a compact provenance record without saving activations or weights."""
    import platform
    artifact=Path(artifact)
    if not artifact.is_file():
        return artifact
    record=provenance(cfg,family,stage,[location(cfg,"general"),location(cfg,"concepts"),
                                       location(cfg,"scenarios"),location(cfg,"steering")])
    record["artifact"]={"path":str(artifact),"sha256":digest(artifact)}
    record["python"]=platform.python_version()
    try:
        import torch
        import transformers
        import peft
        record["runtime"]={"torch":torch.__version__,"cuda":torch.version.cuda,
                           "transformers":transformers.__version__,"peft":peft.__version__,
                           "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        pass
    target=artifact.with_name(artifact.stem+"_manifest.json")
    write_json(target,record)
    return target
