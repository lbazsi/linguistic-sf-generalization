#!/usr/bin/env python3
"""Resume matched held-out steering generation on any configured model family.

Generation is offline. Remote judging is opt-in, requires --execute-judges,
and only starts after strict coverage and token-limit validation. Existing
incompatible generations are archived locally, never silently overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from mechinterp.batched_generation import PROTOCOL
from mechinterp.common import FEATURES, control_for, load_config, load_screening, location, read_data, write_json
from mechinterp.judged import selected_conditions


MODEL_NAMES = ("gemma2", "qwen3", "llama31", "mistral")
ARMS = ("unsteered", "steered")


def _source_protocol(path: Path) -> str:
    """Read only the first JSONL record, including paired partial checkpoints."""
    with path.open(encoding="utf-8") as handle:
        line = next((line for line in handle if line.strip()), None)
    if line is None:
        return "empty"
    row = json.loads(line)
    if "unsteered" in row:
        row = row["unsteered"]
    return str(row.get("sampling_protocol", "legacy"))


def archive_incompatible(root: Path) -> Path | None:
    """Keep the existing v2 data, or atomically move old judged material aside.

    Selection, steering directions, screening artifacts and other analyses
    remain untouched. The archive is under local .work, not published results.
    """
    judged = root / "judged"
    judgments = root / "judgments"
    quality = root / "generation_quality.json"
    files = sorted(judged.glob("*.jsonl")) if judged.exists() else []
    protocols = {_source_protocol(path) for path in files}
    other_entries = [
        entry for entry in judged.iterdir() if entry not in files
    ] if judged.exists() else []
    # If no source responses exist, any leftover judgments are stale.
    stale_judgments = judgments.exists() and any(judgments.iterdir()) and not files
    incompatible = bool(protocols and protocols != {PROTOCOL}) or bool(other_entries) or stale_judgments
    if not incompatible:
        return None

    archive_base = root / ".work"
    archive_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = archive_base / ("archived_judged_" + timestamp)
    suffix = 1
    while archive.exists():
        archive = archive_base / ("archived_judged_" + timestamp + "_" + str(suffix))
        suffix += 1
    archive.mkdir(parents=True)
    for source in (judged, judgments, quality):
        if source.exists():
            shutil.move(str(source), str(archive / source.name))
    print("Archived incompatible judged data:", archive, flush=True)
    return archive


def validate_selection(cfg: dict, model: str) -> list[dict]:
    if model not in cfg["models"]:
        raise ValueError("Model missing from configuration: " + model)
    gen = cfg["steering"]["generation"]
    if int(gen["max_new_tokens"]) != 512 or int(gen.get("min_new_tokens", -1)) != 48:
        raise ValueError("Expected the registered 48-to-512 token generation protocol")
    size = int(gen["batch_size"])
    if size != 32:
        raise ValueError("Expected generation batch_size=32 for paired_batch_sentence_v2")
    selected = selected_conditions(cfg, model)
    expected = {(feature, arm) for feature in FEATURES for arm in (control_for(feature), feature)}
    actual = {(s["feature"], s["adapter"]) for s in selected}
    if len(selected) != len(expected) or actual != expected:
        raise ValueError("Selected configurations must cover all 28 feature/adapter pairs")
    return selected


def verify_generations(cfg: dict, model: str, max_fraction: float) -> dict:
    """Validate all 5,600 records before ANY paid judging is permitted."""
    selected = validate_selection(cfg, model)
    test = [r for r in load_screening(cfg) if r["split"] == cfg["steering"]["judged_split"]]
    test_ids = {int(r["scenario_id"]) for r in test}
    if len(test_ids) != 100:
        raise ValueError("Expected exactly 100 held-out scenario IDs")
    from mechinterp.common import scenario_prompt
    prompt_by_id = {int(r["scenario_id"]): scenario_prompt(r["scenario"]) for r in test}
    root = location(cfg, "outputs") / model / "activation-steering"
    judged = root / "judged"
    reasons = Counter()
    by_arm = {arm: Counter() for arm in ARMS}
    lengths = {arm: [] for arm in ARMS}
    for spec in selected:
        feature, adapter, policy = spec["feature"], spec["adapter"], spec["policy"]
        stem = f"{feature}__{adapter}__{policy}"
        paired = {}
        for arm in ARMS:
            path = judged / f"{stem}__{arm}.jsonl"
            rows = read_data(path)
            indexed = {int(r["scenario_id"]): r for r in rows}
            if len(rows) != 100 or set(indexed) != test_ids:
                raise ValueError(f"Missing, duplicated, or unexpected scenario IDs: {path}")
            paired[arm] = indexed
            for sid, row in indexed.items():
                n = int(row["generated_tokens"])
                reason = row.get("stop_reason")
                if (row.get("sampling_protocol") != PROTOCOL or
                    int(row.get("max_new_tokens", -1)) != 512 or
                    int(row.get("min_new_tokens", -1)) != 48 or
                    int(row.get("batch_size_requested", -1)) != 32 or
                    row["feature"] != feature or row["adapter"] != adapter or
                    row["policy"] != policy or row["arm"] != arm or
                    row["prompt"] != prompt_by_id[sid] or
                    not row["continuation"].strip() or not (0 < n <= 512)):
                    raise ValueError(f"Invalid generation metadata/content: {path}, scenario {sid}")
                if reason not in ("eos", "sentence_boundary", "token_limit"):
                    raise ValueError(f"Missing/unknown stop reason: {path}, scenario {sid}")
                if reason == "token_limit":
                    if n != 512 or not row.get("reached_token_limit") or row.get("ended_with_eos"):
                        raise ValueError(f"Incorrect token-limit classification: {path}, scenario {sid}")
                else:
                    if row.get("reached_token_limit"):
                        raise ValueError(f"Incorrect truncation flag: {path}, scenario {sid}")
                    if reason == "eos" and not row.get("ended_with_eos"):
                        raise ValueError(f"Incorrect EOS flag: {path}, scenario {sid}")
                    if reason == "sentence_boundary" and n < 48:
                        raise ValueError(f"Sentence stop before minimum: {path}, scenario {sid}")
                by_arm[arm][reason] += 1
                reasons[reason] += 1
                lengths[arm].append(n)
        for sid in test_ids:
            a, b = paired["unsteered"][sid], paired["steered"][sid]
            for key in ("prompt", "generation_seed", "batch_scenario_ids", "batch_size_requested"):
                if a[key] != b[key]:
                    raise ValueError(f"Unmatched pair {stem}, scenario {sid}: {key}")
            if abs(float(a["strength"])) > 1e-10:
                raise ValueError(f"Nonzero control strength: {stem}, scenario {sid}")
            if abs(float(b["strength"]) - float(spec["strength"])) > 1e-7:
                raise ValueError(f"Unexpected steered strength: {stem}, scenario {sid}")
    total = sum(reasons.values())
    if total != 5600:
        raise ValueError(f"Expected 5600 generated responses, found {total}")
    arm_fractions = {a: by_arm[a]["token_limit"] / sum(by_arm[a].values()) for a in ARMS}
    fraction = reasons["token_limit"] / total
    report = {
        "model": model,
        "sampling_protocol": PROTOCOL,
        "max_new_tokens": 512,
        "min_new_tokens": 48,
        "generation_batch_size": 32,
        "selected_configurations": len(selected),
        "generated_responses": total,
        "stop_reasons": dict(reasons),
        "stop_reasons_by_arm": {arm: dict(by_arm[arm]) for arm in ARMS},
        "truncation_fraction": fraction,
        "truncation_fraction_by_arm": arm_fractions,
        "mean_generated_tokens_by_arm": {arm: statistics.mean(lengths[arm]) for arm in ARMS},
        "median_generated_tokens_by_arm": {arm: statistics.median(lengths[arm]) for arm in ARMS},
        "quality_gate_threshold": max_fraction,
        "quality_gate_passed": fraction <= max_fraction and all(
            f <= max_fraction for f in arm_fractions.values()
        ),
        "manual_quality_review_required": True,
    }
    write_json(root / "generation_quality.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["quality_gate_passed"]:
        raise RuntimeError(
            f"Truncation exceeds {max_fraction:.1%} overall or in an arm; no paid judging performed"
        )
    return report


def run(*args: str) -> None:
    print("+", sys.executable, " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--stage", choices=("generate", "judge", "all"), default="generate")
    p.add_argument("--execute-judges", action="store_true",
                   help="Explicitly authorize charged judge API requests for judge/all stages")
    p.add_argument("--max-truncation", type=float, default=0.25)
    p.add_argument("--overwrite-export", action="store_true",
                   help="Replace conflicting previously published scalar results after judging")
    args = p.parse_args()

    if not 0 <= args.max_truncation <= 1:
        p.error("--max-truncation must be in [0, 1]")
    if args.stage in ("judge", "all") and not args.execute_judges:
        p.error("Paid judge stages require --execute-judges")
    if args.stage == "generate" and args.execute_judges:
        p.error("--execute-judges is not valid with generation-only mode")

    cfg = load_config()
    validate_selection(cfg, args.model)
    root = location(cfg, "outputs") / args.model / "activation-steering"
    work = root / ".work"
    work.mkdir(parents=True, exist_ok=True)

    # Prevent concurrent runs for the same model/output directory.
    import fcntl
    with (work / "judged_run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another judged run is active for {args.model}") from exc

        if args.stage in ("generate", "all"):
            archive_incompatible(root)
            run("run.py", "judge-plan", "--model", args.model)
            run("run.py", "judge-generate-batched", "--model", args.model)

        verify_generations(cfg, args.model, args.max_truncation)

        if args.stage in ("judge", "all"):
            from dotenv import dotenv_values
            key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(
                location(cfg, "evaluation_config").parent.parent / ".env"
            ).get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("Missing OPENROUTER_API_KEY in environment or evaluation/.env")

            run("run.py", "judge-run", "--model", args.model, "--execute-judges")
            run("run.py", "judge-aggregate", "--model", args.model)
            run("run.py", "plots", "--model", args.model)
            export_args = ["export_final_results.py", "--model", args.model, "--execute"]
            if args.overwrite_export:
                export_args.append("--overwrite")
            run(*export_args)
        print(f"Completed {args.model}: {args.stage}", flush=True)


if __name__ == "__main__":
    main()
