from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from common import (
    OpenRouterClient,
    batches,
    ensure_directories,
    json_text,
    load_config,
    read_jsonl,
    render_prompt,
    resolve_path,
    write_jsonl_atomic,
    write_manifest,
)
from schemas import judgment_batch_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blindly judge model continuations.")
    parser.add_argument(
        "--judge",
        choices=["1", "2", "all"],
        default="all",
        help="Run judge 1, judge 2, or both.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        help="Judge only the named response condition; repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing judgment files.",
    )
    return parser.parse_args()


def response_files(config: dict) -> list[Path]:
    return sorted(resolve_path(config, "responses").glob("*.jsonl"))


async def judge_file(
    *,
    config: dict,
    client: OpenRouterClient,
    judge_number: int,
    response_path: Path,
    scenarios_by_id: dict[int, dict],
    overwrite: bool,
) -> None:
    out_dir = resolve_path(config, "judgments") / f"judge_{judge_number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / response_path.name

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    responses = read_jsonl(response_path)
    if not responses:
        raise RuntimeError(f"{response_path} contains no responses.")

    items = []
    for row in responses:
        scenario = scenarios_by_id[int(row["scenario_id"])]
        items.append(
            {
                "scenario_id": scenario["id"],
                "animal": scenario["animal"],
                "value": scenario["value"],
                "context": scenario["context"],
                "tradeoff": scenario["tradeoff"],
                "scenario": scenario["scenario"],
                "continuation": row["continuation"],
            }
        )

    seed_key = f"response_judge_{judge_number}"
    rng = random.Random(int(config["seeds"][seed_key]))
    rng.shuffle(items)

    batch_size = int(config["judging"]["batch_size"])
    judgments: list[dict] = []

    for batch in batches(items, batch_size):
        ids = [row["scenario_id"] for row in batch]
        prompt = render_prompt(
            "judge_response.txt",
            {"ITEMS_JSON": json_text(batch)},
        )
        response = await client.request_json(
            model_key=seed_key,
            temperature_key=seed_key,
            seed_key=seed_key,
            prompt=prompt,
            schema=judgment_batch_schema(ids),
            schema_name=f"response_judgments_{judge_number}",
        )
        judgments.extend(response["judgments"])

    by_id = {int(row["scenario_id"]): row for row in judgments}
    source_by_id = {int(row["scenario_id"]): row for row in responses}
    output = []
    for scenario_id in sorted(by_id):
        judged = by_id[scenario_id]
        source = source_by_id[scenario_id]
        output.append(
            {
                "scenario_id": scenario_id,
                "judge": judge_number,
                "condition": source["condition"],
                "feature": source.get("feature"),
                "scores": judged["scores"],
                "outcome": judged["outcome"],
            }
        )

    write_jsonl_atomic(output_path, output)
    print(f"[judge {judge_number}] {response_path.stem}: wrote {len(output)} judgments.")


async def async_main() -> None:
    args = parse_args()
    config = load_config()
    ensure_directories(config)

    scenarios = read_jsonl(resolve_path(config, "final_scenarios"))
    scenarios_by_id = {int(row["id"]): row for row in scenarios}

    files = response_files(config)
    requested = set(args.condition or [])
    if requested:
        files = [path for path in files if path.stem in requested]
        missing = sorted(requested - {path.stem for path in files})
        if missing:
            raise RuntimeError(f"Unknown response conditions: {', '.join(missing)}")

    judges = [1, 2] if args.judge == "all" else [int(args.judge)]
    client = OpenRouterClient(config)
    try:
        for judge_number in judges:
            for path in files:
                await judge_file(
                    config=config,
                    client=client,
                    judge_number=judge_number,
                    response_path=path,
                    scenarios_by_id=scenarios_by_id,
                    overwrite=args.overwrite,
                )
    finally:
        await client.close()

    output_paths = {}
    for judge_number in judges:
        for path in files:
            candidate = resolve_path(config, "judgments") / f"judge_{judge_number}" / path.name
            if candidate.exists():
                output_paths[f"judge_{judge_number}_{path.stem}"] = candidate
    write_manifest(
        config,
        stage="04_judge_responses",
        inputs={
            "final_scenarios": resolve_path(config, "final_scenarios"),
            **{f"responses_{path.stem}": path for path in files},
        },
        outputs=output_paths,
        prompt_files=["judge_response.txt"],
        stats=client.stats(),
        extra={"judges_run": judges, "conditions": [path.stem for path in files]},
    )


if __name__ == "__main__":
    asyncio.run(async_main())
