from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DEFAULT_CONFIG_PATH,
    feature_files,
    feature_index,
    language_control_files,
    load_canonical,
    load_config,
    load_feature,
    load_language_control,
    output_directory,
    resolve_data_root,
    stratified_validation_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify fine-tuning inputs and, when present, saved run artifacts."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--canonical", action="store_true")
    group.add_argument("--control")
    group.add_argument("--feature")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root")
    return parser.parse_args()


def verify_one(
    config,
    data_root: Path,
    condition: str,
    feature_name: str | None,
    control_language: str | None = None,
) -> dict:
    canonical_path, canonical_rows, canonical_by_id = load_canonical(data_root)
    seed = int(config["training"]["seed"])
    validation_ids = stratified_validation_ids(
        canonical_rows,
        int(config["data"]["validation_size"]),
        seed,
    )

    if condition == "canonical":
        dataset_path = canonical_path
        rows = canonical_rows
        number = None
    elif condition == "control":
        dataset_path, rows, _ = load_language_control(
            data_root, control_language, canonical_by_id
        )
        number = None
    else:
        dataset_path, rows, _ = load_feature(data_root, feature_name, canonical_by_id)
        number = feature_index(data_root, feature_name)

    run_dir = output_directory(
        config,
        seed=seed,
        condition=condition,
        feature_name=feature_name,
        feature_number=number,
        control_language=control_language,
    )
    summary_path = run_dir / "training_summary.json"
    artifact_checks = None
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        artifact_checks = {
            "summary_exists": True,
            "final_adapter_exists": (run_dir / "adapter").is_dir(),
            "midpoint_adapter_exists": (
                (run_dir / "midpoint_adapter").is_dir()
                if config["output"].get("save_midpoint_adapter", True)
                else True
            ),
            "summary_verification_passed": bool(
                summary.get("verification", {}).get("passed", False)
            ),
        }

    return {
        "condition": condition,
        "feature": feature_name,
        "control_language": control_language,
        "dataset": str(dataset_path),
        "examples": len(rows),
        "canonical_examples": len(canonical_rows),
        "validation_examples": len(validation_ids),
        "validation_ids": validation_ids,
        "input_checks_passed": len(rows) == len(canonical_rows),
        "run_dir": str(run_dir),
        "artifacts": artifact_checks,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = resolve_data_root(config, args.data_root)

    results = []
    if args.all:
        results.append(verify_one(config, data_root, "canonical", None))
        for path in language_control_files(data_root):
            results.append(verify_one(config, data_root, "control", None, path.stem))
        for path in feature_files(data_root):
            results.append(verify_one(config, data_root, "feature", path.stem))
    elif args.canonical:
        results.append(verify_one(config, data_root, "canonical", None))
    elif args.control:
        results.append(verify_one(config, data_root, "control", None, args.control))
    else:
        results.append(verify_one(config, data_root, "feature", args.feature))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
