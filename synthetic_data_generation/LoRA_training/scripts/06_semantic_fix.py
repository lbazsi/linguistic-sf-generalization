from __future__ import annotations

import argparse
import asyncio
import os

from common import (
    OpenRouterClient,
    RunManifest,
    append_jsonl,
    batches_by_topic,
    ensure_directories,
    feature_yaml_text,
    index_items,
    issues_by_id,
    json_text,
    load_config,
    load_issues,
    load_topics,
    render_prompt,
    resolve_path,
    select_features,
    topic_plan,
    write_jsonl_atomic,
)
from schemas import pair_batch_schema
from validation import deterministic_review_file, validate_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair semantic issues and write final datasets.")
    parser.add_argument("--feature", action="append", help="Run only the named feature; repeatable.")
    return parser.parse_args()


async def fix_feature(
    *,
    client: OpenRouterClient,
    config: dict,
    feature: dict,
    topics: list[str],
) -> dict:
    feature_name = feature["name"]
    dataset_size = config["dataset_size"]
    plan = topic_plan(topics, dataset_size)

    source_path = resolve_path(config, "first_review") / f"{feature_name}.jsonl"
    issue_path = (
        resolve_path(config, "nondeterministic_issues")
        / f"{feature_name}.jsonl"
    )
    final_path = resolve_path(config, "final") / f"{feature_name}.jsonl"
    partial_path = final_path.with_name(f".{feature_name}.partial.jsonl")
    final_validation_path = (
        resolve_path(config, "nondeterministic_issues")
        / f"{feature_name}.final_validation.jsonl"
    )

    if not source_path.exists():
        raise FileNotFoundError(f"Missing first-review dataset: {source_path}")
    if not issue_path.exists():
        raise FileNotFoundError(f"Missing non-deterministic issue file: {issue_path}")

    source_candidates = index_items(source_path)
    issues = load_issues(issue_path)
    grouped_issues = issues_by_id(issues)
    flagged_ids = set(grouped_issues)

    existing_partial = index_items(partial_path)
    output: dict[int, dict] = {}
    resumed_fixed = 0

    for item_id in range(1, dataset_size + 1):
        if item_id not in flagged_ids:
            candidates = source_candidates.get(item_id, [])
            if candidates:
                output[item_id] = candidates[-1]
            continue

        for candidate in reversed(existing_partial.get(item_id, [])):
            if not validate_item(
                candidate,
                feature_spec=feature,
                plan=plan,
                dataset_size=dataset_size,
                stage="semantic_fix_resume",
            ):
                output[item_id] = candidate
                resumed_fixed += 1
                break

    write_jsonl_atomic(partial_path, [output[item_id] for item_id in sorted(output)])
    pending_ids = sorted(flagged_ids - set(output))
    batches = batches_by_topic(pending_ids, plan, config["batch_size"])
    write_lock = asyncio.Lock()

    async def run_batch(topic: str, ids: list[int]) -> int:
        problems = []
        for item_id in ids:
            candidates = source_candidates.get(item_id, [])
            problems.append(
                {
                    "id": item_id,
                    "topic": topic,
                    "item": candidates[-1] if candidates else None,
                    "issues": grouped_issues.get(item_id, []),
                }
            )

        prompt = render_prompt(
            config,
            "semantic_fix.txt",
            {
                "FEATURE_YAML": feature_yaml_text(feature),
                "TOPIC": topic,
                "PROBLEMS_JSON": json_text(problems),
            },
        )
        response = await client.request_json(
            role="judge",
            prompt=prompt,
            schema=pair_batch_schema(ids),
            schema_name="semantically_repaired_pairs",
        )
        rows = response["items"]
        async with write_lock:
            append_jsonl(partial_path, rows)
        return len(rows)

    fixed_counts = await asyncio.gather(*(run_batch(topic, ids) for topic, ids in batches))

    candidates = index_items(partial_path)
    canonicalized = [
        candidates[item_id][-1]
        for item_id in range(1, dataset_size + 1)
        if candidates.get(item_id)
    ]
    write_jsonl_atomic(partial_path, canonicalized)

    _, final_issues = deterministic_review_file(
        partial_path,
        feature_spec=feature,
        plan=plan,
        dataset_size=dataset_size,
        stage="final_validation",
    )
    write_jsonl_atomic(final_validation_path, final_issues)
    if final_issues:
        raise RuntimeError(
            f"{feature_name}: final candidate has {len(final_issues)} deterministic issues; "
            f"see {final_validation_path}"
        )

    os.replace(partial_path, final_path)

    return {
        "feature": feature_name,
        "flagged_ids": len(flagged_ids),
        "resumed_fixed_ids": resumed_fixed,
        "judge_fixed_ids": sum(fixed_counts),
        "non_id_residue_removed_by_rebuild": sum(
            1 for issue in issues if issue.get("id") is None
        ),
        "final_items": len(canonicalized),
        "final_validation_issues": len(final_issues),
        "output": str(final_path),
    }


async def async_main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)
    topics = load_topics(config)
    selected = select_features(config, args.feature)
    feature_paths = [path for path, _ in selected]

    manifest = RunManifest(
        stage="06_semantic_fix",
        config=config,
        feature_paths=feature_paths,
        prompt_filenames=["semantic_fix.txt"],
    )
    client: OpenRouterClient | None = None

    try:
        client = OpenRouterClient(config)
        feature_stats = []
        for _, feature in selected:
            stats = await fix_feature(
                client=client,
                config=config,
                feature=feature,
                topics=topics,
            )
            feature_stats.append(stats)
            print(
                f"[{feature['name']}] final items: {stats['final_items']}; "
                f"judge-fixed {stats['judge_fixed_ids']} IDs."
            )

        manifest.finish({"features": feature_stats, **client.stats()})
    except BaseException as exc:
        manifest.fail(exc, client.stats() if client else {})
        raise
    finally:
        if client is not None:
            await client.close()


if __name__ == "__main__":
    asyncio.run(async_main())
