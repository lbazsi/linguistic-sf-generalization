from __future__ import annotations

import argparse
import asyncio

from jsonschema import Draft202012Validator

from common import (
    OpenRouterClient,
    RunManifest,
    append_jsonl,
    batches_by_topic,
    ensure_directories,
    feature_yaml_text,
    index_items,
    json_text,
    load_config,
    load_topics,
    render_prompt,
    resolve_path,
    select_features,
    topic_plan,
)
from schemas import PAIR_SCHEMA, pair_batch_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw synthetic contrast-pair datasets.")
    parser.add_argument("--feature", action="append", help="Run only the named feature; repeatable.")
    return parser.parse_args()


def resumable_ids(path, dataset_size: int) -> set[int]:
    validator = Draft202012Validator(PAIR_SCHEMA)
    completed: set[int] = set()
    for item_id, candidates in index_items(path).items():
        if not (1 <= item_id <= dataset_size):
            continue
        if any(not list(validator.iter_errors(candidate)) for candidate in candidates):
            completed.add(item_id)
    return completed


async def generate_feature(
    *,
    client: OpenRouterClient,
    config: dict,
    feature: dict,
    topics: list[str],
) -> dict:
    feature_name = feature["name"]
    dataset_size = config["dataset_size"]
    plan = topic_plan(topics, dataset_size)
    output_path = resolve_path(config, "raw") / f"{feature_name}.jsonl"

    completed = resumable_ids(output_path, dataset_size)
    missing_ids = sorted(set(range(1, dataset_size + 1)) - completed)
    batches = batches_by_topic(missing_ids, plan, config["batch_size"])
    write_lock = asyncio.Lock()

    async def run_batch(topic: str, ids: list[int]) -> int:
        requests = [{"id": item_id, "topic": topic} for item_id in ids]
        prompt = render_prompt(
            config,
            "generate.txt",
            {
                "FEATURE_YAML": feature_yaml_text(feature),
                "TOPIC": topic,
                "REQUESTS_JSON": json_text(requests),
            },
        )
        response = await client.request_json(
            role="generator",
            prompt=prompt,
            schema=pair_batch_schema(ids),
            schema_name="generated_contrast_pairs",
        )
        rows = response["items"]
        async with write_lock:
            append_jsonl(output_path, rows)
        return len(rows)

    counts = await asyncio.gather(*(run_batch(topic, ids) for topic, ids in batches))
    return {
        "feature": feature_name,
        "dataset_size": dataset_size,
        "resumed_items": len(completed),
        "generated_items": sum(counts),
        "batches": len(batches),
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
        stage="01_generate",
        config=config,
        feature_paths=feature_paths,
        prompt_filenames=["generate.txt"],
    )
    client: OpenRouterClient | None = None

    try:
        client = OpenRouterClient(config)
        feature_stats = []
        for _, feature in selected:
            stats = await generate_feature(
                client=client,
                config=config,
                feature=feature,
                topics=topics,
            )
            feature_stats.append(stats)
            print(
                f"[{feature['name']}] generated {stats['generated_items']} "
                f"items; resumed {stats['resumed_items']}."
            )

        manifest.finish(
            {
                "features": feature_stats,
                **client.stats(),
            }
        )
    except BaseException as exc:
        manifest.fail(exc, client.stats() if client else {})
        raise
    finally:
        if client is not None:
            await client.close()


if __name__ == "__main__":
    asyncio.run(async_main())
