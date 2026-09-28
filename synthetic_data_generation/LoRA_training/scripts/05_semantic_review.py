from __future__ import annotations

import argparse
import asyncio

from common import (
    OpenRouterClient,
    RunManifest,
    ensure_directories,
    feature_yaml_text,
    index_items,
    json_text,
    load_config,
    load_issues,
    render_prompt,
    resolve_path,
    select_features,
    write_jsonl_atomic,
)
from schemas import review_batch_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-deterministic semantic review.")
    parser.add_argument("--feature", action="append", help="Run only the named feature; repeatable.")
    return parser.parse_args()


def chunks(values: list[dict], size: int) -> list[list[dict]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


async def review_feature(
    *,
    client: OpenRouterClient,
    config: dict,
    feature: dict,
) -> dict:
    feature_name = feature["name"]
    source_path = resolve_path(config, "first_review") / f"{feature_name}.jsonl"
    residue_path = (
        resolve_path(config, "deterministic_issues")
        / f"{feature_name}.recheck.jsonl"
    )
    output_path = (
        resolve_path(config, "nondeterministic_issues")
        / f"{feature_name}.jsonl"
    )

    if not source_path.exists():
        raise FileNotFoundError(f"Missing first-review dataset: {source_path}")
    if not residue_path.exists():
        raise FileNotFoundError(f"Missing deterministic recheck issues: {residue_path}")

    residue = load_issues(residue_path)
    candidates = index_items(source_path)
    items = [candidates[item_id][-1] for item_id in sorted(candidates) if candidates[item_id]]
    item_batches = chunks(items, config["batch_size"])

    async def run_batch(batch: list[dict]) -> list[dict]:
        ids = [item["id"] for item in batch]
        prompt = render_prompt(
            config,
            "semantic_review.txt",
            {
                "FEATURE_YAML": feature_yaml_text(feature),
                "ITEMS_JSON": json_text(batch),
            },
        )
        response = await client.request_json(
            role="reviewer",
            prompt=prompt,
            schema=review_batch_schema(ids),
            schema_name="semantic_review_issues",
        )
        return [
            {
                "id": issue["id"],
                "feature": feature_name,
                "stage": "semantic_review",
                "issue_type": issue["issue_type"],
                "field": issue["field"],
                "message": issue["message"],
            }
            for issue in response["issues"]
        ]

    results = await asyncio.gather(*(run_batch(batch) for batch in item_batches))
    semantic_issues = [issue for batch in results for issue in batch]
    all_issues = residue + semantic_issues
    all_issues.sort(
        key=lambda issue: (
            issue.get("id") is None,
            issue.get("id") if issue.get("id") is not None else 10**18,
            issue.get("stage", ""),
            issue.get("issue_type", ""),
        )
    )
    write_jsonl_atomic(output_path, all_issues)

    return {
        "feature": feature_name,
        "reviewed_items": len(items),
        "batches": len(item_batches),
        "deterministic_residue": len(residue),
        "semantic_issues": len(semantic_issues),
        "total_issues": len(all_issues),
        "output": str(output_path),
    }


async def async_main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)
    selected = select_features(config, args.feature)
    feature_paths = [path for path, _ in selected]

    manifest = RunManifest(
        stage="05_semantic_review",
        config=config,
        feature_paths=feature_paths,
        prompt_filenames=["semantic_review.txt"],
    )
    client: OpenRouterClient | None = None

    try:
        client = OpenRouterClient(config)
        feature_stats = []
        for _, feature in selected:
            stats = await review_feature(client=client, config=config, feature=feature)
            feature_stats.append(stats)
            print(
                f"[{feature['name']}] semantic issues: {stats['semantic_issues']}; "
                f"deterministic residue: {stats['deterministic_residue']}."
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
