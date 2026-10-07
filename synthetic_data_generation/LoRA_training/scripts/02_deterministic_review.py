from __future__ import annotations

import argparse

from common import (
    RunManifest,
    ensure_directories,
    load_config,
    load_canonical_corpus,
    resolve_path,
    select_features,
    resolve_max_id,
    write_jsonl_atomic,
)
from validation import deterministic_review_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic validation on raw datasets.")
    parser.add_argument("--feature", action="append", help="Run only the named feature; repeatable.")
    parser.add_argument("--max-id", type=int, help="Process only IDs 1..N.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)
    target_max_id = resolve_max_id(config, args.max_id)
    canonical = load_canonical_corpus(config, target_max_id)
    selected = select_features(config, args.feature)
    feature_paths = [path for path, _ in selected]

    manifest = RunManifest(
        stage="02_deterministic_review",
        config=config,
        feature_paths=feature_paths,
    )

    try:
        stats = []
        for _, feature in selected:
            feature_name = feature["name"]
            raw_path = resolve_path(config, "raw") / f"{feature_name}.jsonl"
            if not raw_path.exists():
                raise FileNotFoundError(f"Missing raw dataset: {raw_path}")

            _, issues = deterministic_review_file(
                raw_path,
                feature_spec=feature,
                canonical=canonical,
                dataset_size=target_max_id,
                stage="deterministic_review",
            )
            issue_path = resolve_path(config, "deterministic_issues") / f"{feature_name}.jsonl"
            write_jsonl_atomic(issue_path, issues)
            stats.append(
                {
                    "feature": feature_name,
                    "issues": len(issues),
                    "output": str(issue_path),
                }
            )
            print(f"[{feature_name}] deterministic issues: {len(issues)}")

        manifest.finish({"features": stats})
    except BaseException as exc:
        manifest.fail(exc)
        raise


if __name__ == "__main__":
    main()
