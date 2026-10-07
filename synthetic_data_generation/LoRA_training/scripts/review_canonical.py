from __future__ import annotations

import argparse
import asyncio

from common import (
    OpenRouterClient,
    RunManifest,
    batches,
    canonical_corpus_path,
    ensure_directories,
    generated_canonical_path,
    index_items,
    json_text,
    load_config,
    render_prompt,
    resolve_max_id,
    write_jsonl_atomic,
)
from schemas import CANONICAL_SCHEMA, canonical_text_batch_schema
from jsonschema import Draft202012Validator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review and repair the generated canonical corpus non-deterministically."
    )
    parser.add_argument("--max-id", type=int, help="Process only IDs 1..N.")
    return parser.parse_args()


async def review_canonical_corpus(
    *,
    client: OpenRouterClient,
    config: dict,
    max_id: int | None = None,
) -> tuple[dict[int, dict], dict]:
    source_path = generated_canonical_path(config)
    output_path = canonical_corpus_path(config)
    dataset_size = config["dataset_size"]
    target_max_id = resolve_max_id(config, max_id)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing generated canonical corpus: {source_path}. Run scripts/01_generate.py first."
        )

    source_candidates = index_items(source_path)
    validator = Draft202012Validator(CANONICAL_SCHEMA)
    source: dict[int, dict] = {}
    for item_id in range(1, target_max_id + 1):
        candidates = source_candidates.get(item_id, [])
        valid = [
            row for row in candidates
            if not list(validator.iter_errors(row))
        ]
        if not valid:
            raise RuntimeError(
                f"Generated canonical corpus has no schema-valid row for ID {item_id}."
            )
        source[item_id] = valid[-1]

    existing = index_items(output_path)
    completed: dict[int, dict] = {}
    for item_id in range(1, target_max_id + 1):
        candidates = existing.get(item_id, [])
        for candidate in reversed(candidates):
            if (
                not list(validator.iter_errors(candidate))
                and candidate.get("id") == item_id
                and candidate.get("language") == source[item_id]["language"]
                and candidate.get("animal") == source[item_id]["animal"]
                and candidate.get("value") == source[item_id]["value"]
                and candidate.get("context") == source[item_id]["context"]
            ):
                completed[item_id] = candidate
                break

    pending_ids = sorted(set(range(1, target_max_id + 1)) - set(completed))
    item_batches = batches(pending_ids, config["batch_size"])

    async def run_batch(ids: list[int]) -> list[dict]:
        items = [source[item_id] for item_id in ids]
        prompt = render_prompt(
            config,
            "canonical_review.txt",
            {"CANONICAL_ITEMS_JSON": json_text(items)},
        )
        response = await client.request_json(
            role="judge",
            prompt=prompt,
            schema=canonical_text_batch_schema(ids),
            schema_name="reviewed_canonical_corpus",
        )
        reviewed = {row["id"]: row["canonical"] for row in response["items"]}
        return [
            {
                "id": item_id,
                "language": source[item_id]["language"],
                "animal": source[item_id]["animal"],
                "value": source[item_id]["value"],
                "context": source[item_id]["context"],
                "canonical": reviewed[item_id],
            }
            for item_id in ids
        ]

    results = await asyncio.gather(*(run_batch(ids) for ids in item_batches))
    for batch_rows in results:
        for row in batch_rows:
            completed[row["id"]] = row

    ordered = [completed[item_id] for item_id in range(1, target_max_id + 1)]
    retained = [
        existing[item_id][-1]
        for item_id in sorted(existing)
        if item_id > target_max_id and item_id <= dataset_size
    ]
    write_jsonl_atomic(output_path, ordered + retained)

    changed = sum(
        completed[item_id]["canonical"] != source[item_id]["canonical"]
        for item_id in range(1, target_max_id + 1)
    )
    corpus = {row["id"]: row for row in ordered}
    return corpus, {
        "dataset_size": dataset_size,
        "reviewed_items": len(pending_ids),
        "resumed_items": dataset_size - len(pending_ids),
        "changed_items": changed,
        "unchanged_items": dataset_size - changed,
        "source": str(source_path),
        "output": str(output_path),
    }


async def async_main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)
    manifest = RunManifest(
        stage="canonical_review",
        config=config,
        feature_paths=[],
        prompt_filenames=["canonical_review.txt"],
    )
    client: OpenRouterClient | None = None

    try:
        client = OpenRouterClient(config)
        _, stats = await review_canonical_corpus(client=client, config=config, max_id=args.max_id)
        print(
            f"[canonical-review] reviewed {stats['reviewed_items']} items; "
            f"changed {stats['changed_items']}."
        )
        manifest.finish({"canonical_review": stats, **client.stats()})
    except BaseException as exc:
        manifest.fail(exc, client.stats() if client else {})
        raise
    finally:
        if client is not None:
            await client.close()


if __name__ == "__main__":
    asyncio.run(async_main())
