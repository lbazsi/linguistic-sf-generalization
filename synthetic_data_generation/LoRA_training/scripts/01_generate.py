from __future__ import annotations

import argparse
import asyncio

from jsonschema import Draft202012Validator

from common import (
    OpenRouterClient,
    RunManifest,
    append_jsonl,
    batches,
    batches_by_domain,
    domain_plan,
    ensure_directories,
    feature_yaml_text,
    generated_canonical_path,
    index_items,
    json_text,
    language_control_path,
    load_config,
    load_domains,
    render_prompt,
    resolve_path,
    select_features,
    write_jsonl_atomic,
)
from schemas import (
    CANONICAL_SCHEMA,
    PAIR_SCHEMA,
    canonical_text_batch_schema,
    control_text_batch_schema,
    variant_batch_schema,
)
from review_canonical import review_canonical_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, review, and transform semantic-anchor and linguistic-feature datasets."
    )
    parser.add_argument("--feature", action="append", help="Run only the named feature; repeatable.")
    return parser.parse_args()


async def generate_canonical_corpus(
    *,
    client: OpenRouterClient,
    config: dict,
    domains: dict[str, list[str]],
) -> tuple[dict[int, dict], dict]:
    dataset_size = config["dataset_size"]
    language = config["canonical_language"]
    plan = domain_plan(domains, dataset_size, int(config["seeds"]["generator"]))
    output_path = generated_canonical_path(config)

    validator = Draft202012Validator(CANONICAL_SCHEMA)
    completed: set[int] = set()
    for item_id, candidates in index_items(output_path).items():
        if not (1 <= item_id <= dataset_size):
            continue
        if any(
            not list(validator.iter_errors(candidate))
            and candidate.get("language") == language
            and candidate.get("animal") == plan[item_id]["animal"]
            and candidate.get("value") == plan[item_id]["value"]
            and candidate.get("context") == plan[item_id]["context"]
            for candidate in candidates
        ):
            completed.add(item_id)

    missing_ids = sorted(set(range(1, dataset_size + 1)) - completed)
    domain_batches = batches_by_domain(missing_ids, plan, config["batch_size"])
    write_lock = asyncio.Lock()

    async def run_batch(domain: dict[str, str], ids: list[int]) -> int:
        requests = [{"id": item_id} for item_id in ids]
        prompt = render_prompt(
            config,
            "generate_canonical.txt",
            {
                "LANGUAGE": language,
                "ANIMAL": domain["animal"],
                "VALUE": domain["value"],
                "CONTEXT": domain["context"],
                "REQUESTS_JSON": json_text(requests),
            },
        )
        response = await client.request_json(
            role="generator",
            prompt=prompt,
            schema=canonical_text_batch_schema(ids),
            schema_name="generated_canonical_corpus",
        )
        generated = {row["id"]: row["canonical"] for row in response["items"]}
        rows = [
            {
                "id": item_id,
                "language": language,
                "animal": plan[item_id]["animal"],
                "value": plan[item_id]["value"],
                "context": plan[item_id]["context"],
                "canonical": generated[item_id],
            }
            for item_id in ids
        ]
        async with write_lock:
            append_jsonl(output_path, rows)
        return len(rows)

    counts = await asyncio.gather(*(run_batch(domain, ids) for domain, ids in domain_batches))

    candidates = index_items(output_path)
    corpus: dict[int, dict] = {}
    for item_id in range(1, dataset_size + 1):
        valid = [
            row for row in candidates.get(item_id, [])
            if not list(validator.iter_errors(row))
            and row.get("language") == language
            and row.get("animal") == plan[item_id]["animal"]
            and row.get("value") == plan[item_id]["value"]
            and row.get("context") == plan[item_id]["context"]
        ]
        if not valid:
            raise RuntimeError(f"Canonical corpus has no valid row for ID {item_id}.")
        corpus[item_id] = valid[-1]

    write_jsonl_atomic(output_path, [corpus[item_id] for item_id in range(1, dataset_size + 1)])
    return corpus, {
        "dataset_size": dataset_size,
        "resumed_items": len(completed),
        "generated_items": sum(counts),
        "batches": len(domain_batches),
        "output": str(output_path),
    }


async def generate_language_control(
    *,
    client: OpenRouterClient,
    config: dict,
    language: str,
    semantic_anchor: dict[int, dict],
) -> tuple[dict[int, dict], dict]:
    dataset_size = config["dataset_size"]
    output_path = language_control_path(config, language)
    generated_path = output_path.with_name(f"{language}.generated.jsonl")

    existing = index_items(generated_path)
    completed = {
        item_id
        for item_id, rows in existing.items()
        if rows and rows[-1].get("language") == language
        and rows[-1].get("semantic_anchor") == semantic_anchor[item_id]["canonical"]
    }
    missing_ids = sorted(set(range(1, dataset_size + 1)) - completed)
    write_lock = asyncio.Lock()

    async def generate_batch(ids: list[int]) -> int:
        anchors = [
            {
                "id": item_id,
                "animal": semantic_anchor[item_id]["animal"],
                "value": semantic_anchor[item_id]["value"],
                "context": semantic_anchor[item_id]["context"],
                "semantic_anchor": semantic_anchor[item_id]["canonical"],
            }
            for item_id in ids
        ]
        prompt = render_prompt(
            config,
            "generate_language_control.txt",
            {
                "TARGET_LANGUAGE": language,
                "CANONICAL_ITEMS_JSON": json_text(anchors),
            },
        )
        response = await client.request_json(
            role="generator",
            prompt=prompt,
            schema=control_text_batch_schema(ids),
            schema_name=f"generated_{language}_language_control",
        )
        translated = {row["id"]: row["canonical"] for row in response["items"]}
        rows = [
            {
                "id": item_id,
                "language": language,
                "animal": semantic_anchor[item_id]["animal"],
                "value": semantic_anchor[item_id]["value"],
                "context": semantic_anchor[item_id]["context"],
                "semantic_anchor": semantic_anchor[item_id]["canonical"],
                "canonical": translated[item_id],
            }
            for item_id in ids
        ]
        async with write_lock:
            append_jsonl(generated_path, rows)
        return len(rows)

    generated_counts = await asyncio.gather(
        *(generate_batch(ids) for ids in batches(missing_ids, config["batch_size"]))
    )

    candidates = index_items(generated_path)
    generated_rows = [candidates[item_id][-1] for item_id in range(1, dataset_size + 1)]
    write_jsonl_atomic(generated_path, generated_rows)

    review_batches = batches(range(1, dataset_size + 1), config["batch_size"])

    async def review_batch(ids: list[int]) -> list[dict]:
        items = [candidates[item_id][-1] for item_id in ids]
        prompt = render_prompt(
            config,
            "review_language_control.txt",
            {
                "TARGET_LANGUAGE": language,
                "ITEMS_JSON": json_text(items),
            },
        )
        response = await client.request_json(
            role="judge",
            prompt=prompt,
            schema=control_text_batch_schema(ids),
            schema_name=f"reviewed_{language}_language_control",
        )
        reviewed = {row["id"]: row["canonical"] for row in response["items"]}
        return [
            {
                **candidates[item_id][-1],
                "canonical": reviewed[item_id],
            }
            for item_id in ids
        ]

    reviewed_batches = await asyncio.gather(*(review_batch(ids) for ids in review_batches))
    rows = [row for batch in reviewed_batches for row in batch]
    rows.sort(key=lambda row: row["id"])
    write_jsonl_atomic(output_path, rows)

    return {row["id"]: row for row in rows}, {
        "language": language,
        "generated_items": sum(generated_counts),
        "reviewed_items": len(rows),
        "output": str(output_path),
    }


async def generate_feature(
    *,
    client: OpenRouterClient,
    config: dict,
    feature: dict,
    semantic_anchor: dict[int, dict],
    language_controls: dict[str, dict[int, dict]],
) -> dict:
    feature_name = feature["name"]
    dataset_size = config["dataset_size"]
    output_path = resolve_path(config, "raw") / f"{feature_name}.jsonl"
    manipulation = feature["manipulation_level"]

    if manipulation == "cross_linguistic":
        canonical_language = feature["canonical_language"]
        feature_variant_language = feature["feature_variant_language"]
        source_rows = (
            semantic_anchor
            if canonical_language == config["canonical_language"]
            else language_controls[canonical_language]
        )
        prompt_name = "generate_cross_linguistic.txt"
    else:
        canonical_language = feature["language"]
        feature_variant_language = feature["language"]
        source_rows = semantic_anchor
        prompt_name = "generate.txt"

    completed: set[int] = set()
    validator = Draft202012Validator(PAIR_SCHEMA)
    for item_id, candidates in index_items(output_path).items():
        if not (1 <= item_id <= dataset_size):
            continue
        anchor = semantic_anchor[item_id]
        source = source_rows[item_id]
        for candidate in candidates:
            if (
                not list(validator.iter_errors(candidate))
                and candidate.get("feature") == feature_name
                and candidate.get("manipulation_level") == manipulation
                and candidate.get("canonical_language") == canonical_language
                and candidate.get("feature_variant_language") == feature_variant_language
                and candidate.get("animal") == anchor["animal"]
                and candidate.get("value") == anchor["value"]
                and candidate.get("context") == anchor["context"]
                and candidate.get("semantic_anchor") == anchor["canonical"]
                and candidate.get("canonical") == source["canonical"]
            ):
                completed.add(item_id)
                break

    missing_ids = sorted(set(range(1, dataset_size + 1)) - completed)
    item_batches = batches(missing_ids, config["batch_size"])
    write_lock = asyncio.Lock()

    async def run_batch(ids: list[int]) -> int:
        source_items = [
            {
                "id": item_id,
                "canonical_language": canonical_language,
                "feature_variant_language": feature_variant_language,
                "animal": semantic_anchor[item_id]["animal"],
                "value": semantic_anchor[item_id]["value"],
                "context": semantic_anchor[item_id]["context"],
                "semantic_anchor": semantic_anchor[item_id]["canonical"],
                "canonical": source_rows[item_id]["canonical"],
            }
            for item_id in ids
        ]
        prompt = render_prompt(
            config,
            prompt_name,
            {
                "FEATURE_YAML": feature_yaml_text(feature),
                "CANONICAL_ITEMS_JSON": json_text(source_items),
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
                "manipulation_level": manipulation,
                "canonical_language": canonical_language,
                "feature_variant_language": feature_variant_language,
                "animal": semantic_anchor[item_id]["animal"],
                "value": semantic_anchor[item_id]["value"],
                "context": semantic_anchor[item_id]["context"],
                "semantic_anchor": semantic_anchor[item_id]["canonical"],
                "canonical": source_rows[item_id]["canonical"],
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
        "manipulation_level": manipulation,
        "canonical_language": canonical_language,
        "feature_variant_language": feature_variant_language,
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
    domains = load_domains(config)
    selected = select_features(config, args.feature)
    feature_paths = [path for path, _ in selected]

    manifest = RunManifest(
        stage="01_generate",
        config=config,
        feature_paths=feature_paths,
        prompt_filenames=[
            "generate_canonical.txt",
            "canonical_review.txt",
            "generate.txt",
            "generate_cross_linguistic.txt",
            "generate_language_control.txt",
            "review_language_control.txt",
        ],
    )
    client: OpenRouterClient | None = None

    try:
        client = OpenRouterClient(config)
        semantic_anchor, canonical_stats = await generate_canonical_corpus(
            client=client,
            config=config,
            domains=domains,
        )
        print(
            f"[canonical] generated {canonical_stats['generated_items']} "
            f"items; resumed {canonical_stats['resumed_items']}."
        )

        semantic_anchor, review_stats = await review_canonical_corpus(
            client=client,
            config=config,
        )
        print(
            f"[canonical-review] reviewed {review_stats['reviewed_items']} items; "
            f"changed {review_stats['changed_items']}."
        )

        source_languages = sorted({
            feature["canonical_language"]
            for _, feature in selected
            if feature["manipulation_level"] == "cross_linguistic"
            and feature["canonical_language"] != config["canonical_language"]
        })
        language_controls: dict[str, dict[int, dict]] = {}
        control_stats = []
        for language in source_languages:
            control, stats = await generate_language_control(
                client=client,
                config=config,
                language=language,
                semantic_anchor=semantic_anchor,
            )
            language_controls[language] = control
            control_stats.append(stats)
            print(f"[control:{language}] reviewed {stats['reviewed_items']} items.")

        feature_stats = []
        for _, feature in selected:
            stats = await generate_feature(
                client=client,
                config=config,
                feature=feature,
                semantic_anchor=semantic_anchor,
                language_controls=language_controls,
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
                "language_controls": control_stats,
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
