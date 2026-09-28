from __future__ import annotations

import argparse
import asyncio

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
from validation import validate_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair deterministic validation failures.")
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

    raw_path = resolve_path(config, "raw") / f"{feature_name}.jsonl"
    issue_path = resolve_path(config, "deterministic_issues") / f"{feature_name}.jsonl"
    output_path = resolve_path(config, "first_review") / f"{feature_name}.jsonl"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {raw_path}")
    if not issue_path.exists():
        raise FileNotFoundError(f"Missing deterministic issue file: {issue_path}")

    raw_candidates = index_items(raw_path)
    issues = load_issues(issue_path)
    grouped_issues = issues_by_id(issues)
    flagged_ids = set(grouped_issues)

    existing_candidates = index_items(output_path)
    output: dict[int, dict] = {}
    resumed_fixed = 0

    for item_id in range(1, dataset_size + 1):
        if item_id not in flagged_ids:
            candidates = raw_candidates.get(item_id, [])
            if candidates:
                output[item_id] = candidates[-1]
            continue

        for candidate in reversed(existing_candidates.get(item_id, [])):
            if not validate_item(
                candidate,
                feature_spec=feature,
                plan=plan,
                dataset_size=dataset_size,
                stage="deterministic_fix_resume",
            ):
                output[item_id] = candidate
                resumed_fixed += 1
                break

    write_jsonl_atomic(output_path, [output[item_id] for item_id in sorted(output)])
    pending_ids = sorted(flagged_ids - set(output))
    batches = batches_by_topic(pending_ids, plan, config["batch_size"])
    write_lock = asyncio.Lock()

    async def run_batch(topic: str, ids: list[int]) -> int:
        problems = []
        for item_id in ids:
            problems.append(
                {
                    "id": item_id,
                    "topic": topic,
                    "candidates": raw_candidates.get(item_id, []),
                    "issues": grouped_issues.get(item_id, []),
                }
            )

        prompt = render_prompt(
            config,
            "deterministic_fix.txt",
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
            schema_name="deterministically_repaired_pairs",
        )
        rows = response["items"]
        async with write_lock:
            append_jsonl(output_path, rows)
        return len(rows)

    fixed_counts = await asyncio.gather(*(run_batch(topic, ids) for topic, ids in batches))

    final_candidates = index_items(output_path)
    canonicalized: list[dict] = []
    for item_id in range(1, dataset_size + 1):
        candidates = final_candidates.get(item_id, [])
        if candidates:
            canonicalized.append(candidates[-1])
    write_jsonl_atomic(output_path, canonicalized)

    return {
        "feature": feature_name,
        "flagged_ids": len(flagged_ids),
        "resumed_fixed_ids": resumed_fixed,
        "judge_fixed_ids": sum(fixed_counts),
        "unaddressed_non_id_issues": sum(1 for issue in issues if issue.get("id") is None),
        "output_items": len(canonicalized),
        "output": str(output_path),
    }


async def async_main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)
    topics = load_topics(config)
    selected = select_features(config, args.feature)
    feature_paths = [path for path, _ in selected]

    manifest = RunManifest(
        stage="03_deterministic_fix",
        config=config,
        feature_paths=feature_paths,
        prompt_filenames=["deterministic_fix.txt"],
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
                f"[{feature['name']}] judge-fixed {stats['judge_fixed_ids']} IDs; "
                f"resumed {stats['resumed_fixed_ids']}."
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
