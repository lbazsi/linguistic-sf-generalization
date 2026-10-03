from __future__ import annotations

import argparse

from common import (
    RunManifest,
    ensure_directories,
    load_config,
    load_canonical_corpus,
    resolve_path,
    select_features,
    write_jsonl_atomic,
)
from validation import deterministic_review_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recheck repaired datasets deterministically.")
    parser.add_argument("--feature", action="append", help="Run only the named feature; repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)
    canonical = load_canonical_corpus(config)
    selected = select_features(config, args.feature)
    feature_paths = [path for path, _ in selected]

    manifest = RunManifest(
        stage="04_deterministic_recheck",
        config=config,
        feature_paths=feature_paths,
    )

    try:
        stats = []
        for _, feature in selected:
            feature_name = feature["name"]
            source_path = resolve_path(config, "first_review") / f"{feature_name}.jsonl"
            if not source_path.exists():
                raise FileNotFoundError(f"Missing first-review dataset: {source_path}")

            _, issues = deterministic_review_file(
                source_path,
                feature_spec=feature,
                canonical=canonical,
                dataset_size=config["dataset_size"],
                stage="deterministic_recheck",
            )
            issue_path = (
                resolve_path(config, "deterministic_issues")
                / f"{feature_name}.recheck.jsonl"
            )
            write_jsonl_atomic(issue_path, issues)
            stats.append(
                {
                    "feature": feature_name,
                    "issues": len(issues),
                    "output": str(issue_path),
                }
            )
            print(f"[{feature_name}] deterministic recheck issues: {len(issues)}")

        manifest.finish({"features": stats})
    except BaseException as exc:
        manifest.fail(exc)
        raise


if __name__ == "__main__":
    main()
