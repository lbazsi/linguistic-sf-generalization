from __future__ import annotations

import argparse
import asyncio

from jsonschema import Draft202012Validator

from common import (
    OpenRouterClient,
    RunManifest,
    append_jsonl,
    batches,
    batches_by_topic,
    generated_canonical_path,
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
    write_jsonl_atomic,
)
from schemas import CANONICAL_SCHEMA, PAIR_SCHEMA, canonical_batch_schema, variant_batch_schema
from review_canonical import review_canonical_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, review, and transform the shared canonical corpus for each feature."
    )
    parser.add_argument("--feature", action="append", help="Run only the named feature; repeatable.")
    return parser.parse_args()


def resumable_ids(path, schema: dict, dataset_size: int) -> set[int]:
    validator = Draft202012Validator(schema)
    completed: set[int] = set()
    for item_id, candidates in index_items(path).items():
        if not (1 <= item_id <= dataset_size):
            continue
        if any(not list(validator.iter_errors(candidate)) for candidate in candidates):
            completed.add(item_id)
    return completed


async def generate_canonical_corpus(
    *,
    client: OpenRouterClient,
    config: dict,
    topics: list[str],
) -> tuple[dict[int, dict], dict]:
    dataset_size = config["dataset_size"]
    language = config["canonical_language"]
    plan = topic_plan(topics, dataset_size)
    output_path = generated_canonical_path(config)

    validator = Draft202012Validator(CANONICAL_SCHEMA)
    completed: set[int] = set()
    for item_id, candidates in index_items(output_path).items():
        if not (1 <= item_id <= dataset_size):
            continue
        if any(
            not list(validator.iter_errors(candidate))
            and candidate.get("topic") == plan[item_id]
            and candidate.get("language") == language
            for candidate in candidates
        ):
            completed.add(item_id)
    missing_ids = sorted(set(range(1, dataset_size + 1)) - completed)
    topic_batches = batches_by_topic(missing_ids, plan, config["batch_size"])
    write_lock = asyncio.Lock()

    async def run_batch(topic: str, ids: list[int]) -> int:
        requests = [{"id": item_id} for item_id in ids]
        prompt = render_prompt(
            config,
            "generate_canonical.txt",
            {
                "LANGUAGE": language,
                "TOPIC": topic,
                "REQUESTS_JSON": json_text(requests),
            },
        )
        response = await client.request_json(
            role="generator",
            prompt=prompt,
            schema=canonical_batch_schema(ids),
            schema_name="generated_canonical_corpus",
        )
        rows = response["items"]
        async with write_lock:
            append_jsonl(output_path, rows)
        return len(rows)

    counts = await asyncio.gather(*(run_batch(topic, ids) for topic, ids in topic_batches))

    candidates = index_items(output_path)
    corpus: dict[int, dict] = {}
    validator = Draft202012Validator(CANONICAL_SCHEMA)
    for item_id in range(1, dataset_size + 1):
        valid = [
            row for row in candidates.get(item_id, [])
            if not list(validator.iter_errors(row))
            and row.get("topic") == plan[item_id]
            and row.get("language") == language
        ]
        if not valid:
            raise RuntimeError(f"Canonical corpus has no valid row for ID {item_id}.")
        corpus[item_id] = valid[-1]

    write_jsonl_atomic(output_path, [corpus[item_id] for item_id in range(1, dataset_size + 1)])
    return corpus, {
        "dataset_size": dataset_size,
        "resumed_items": len(completed),
        "generated_items": sum(counts),
        "batches": len(topic_batches),
        "output": str(output_path),
    }


async def generate_feature(
    *,
    client: OpenRouterClient,
    config: dict,
    feature: dict,
    canonical: dict[int, dict],
) -> dict:
    feature_name = feature["name"]
    dataset_size = config["dataset_size"]
    output_path = resolve_path(config, "raw") / f"{feature_name}.jsonl"

    completed: set[int] = set()
    validator = Draft202012Validator(PAIR_SCHEMA)
    for item_id, candidates in index_items(output_path).items():
        if not (1 <= item_id <= dataset_size):
            continue
        source = canonical[item_id]
        for candidate in candidates:
            if (
                not list(validator.iter_errors(candidate))
                and candidate.get("feature") == feature_name
                and candidate.get("language") == source["language"]
                and candidate.get("topic") == source["topic"]
                and candidate.get("canonical") == source["canonical"]
            ):
                completed.add(item_id)
                break

    missing_ids = sorted(set(range(1, dataset_size + 1)) - completed)
    item_batches = batches(missing_ids, config["batch_size"])
    write_lock = asyncio.Lock()

    async def run_batch(ids: list[int]) -> int:
        canonical_items = [canonical[item_id] for item_id in ids]
        prompt = render_prompt(
            config,
            "generate.txt",
            {
                "FEATURE_YAML": feature_yaml_text(feature),
                "CANONICAL_ITEMS_JSON": json_text(canonical_items),
            },
        )
        response = await client.request_json(
            role="generator",
            prompt=prompt,
            schema=variant_batch_schema(ids),
            schema_name="generated_feature_transformations",
        )
        variants = {row["id"]: row["feature_variant"] for row in response["items"]}
        rows = [
            {
                "id": item_id,
                "feature": feature_name,
                "language": canonical[item_id]["language"],
                "topic": canonical[item_id]["topic"],
                "canonical": canonical[item_id]["canonical"],
                "feature_variant": variants[item_id],
            }
            for item_id in ids
        ]
        async with write_lock:
            append_jsonl(output_path, rows)
        return len(rows)

    counts = await asyncio.gather(*(run_batch(ids) for ids in item_batches))
    candidates = index_items(output_path)
    canonicalized = [candidates[item_id][-1] for item_id in range(1, dataset_size + 1)]
    write_jsonl_atomic(output_path, canonicalized)

    return {
        "feature": feature_name,
        "dataset_size": dataset_size,
        "resumed_items": len(completed),
        "generated_items": sum(counts),
        "batches": len(item_batches),
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
        prompt_filenames=["generate_canonical.txt", "canonical_review.txt", "generate.txt"],
    )
    client: OpenRouterClient | None = None

    try:
        client = OpenRouterClient(config)
        canonical, canonical_stats = await generate_canonical_corpus(
            client=client,
            config=config,
            topics=topics,
        )
        print(
            f"[canonical] generated {canonical_stats['generated_items']} "
            f"items; resumed {canonical_stats['resumed_items']}."
        )

        canonical, review_stats = await review_canonical_corpus(
            client=client,
            config=config,
        )
        print(
            f"[canonical-review] reviewed {review_stats['reviewed_items']} items; "
            f"changed {review_stats['changed_items']}."
        )

        feature_stats = []
        for _, feature in selected:
            if feature["language"] != config["canonical_language"]:
                raise RuntimeError(
                    f"{feature['name']}: feature language {feature['language']!r} does not match "
                    f"canonical_language {config['canonical_language']!r}."
                )
            stats = await generate_feature(
                client=client,
                config=config,
                feature=feature,
                canonical=canonical,
            )
            feature_stats.append(stats)
            print(
                f"[{feature['name']}] transformed {stats['generated_items']} "
                f"items; resumed {stats['resumed_items']}."
            )

        manifest.finish(
            {
                "canonical_generation": canonical_stats,
                "canonical_review": review_stats,
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
