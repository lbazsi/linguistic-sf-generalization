from __future__ import annotations

import asyncio

from common import (
    OpenRouterClient,
    batches,
    ensure_directories,
    json_text,
    load_config,
    render_prompt,
    resolve_path,
    scenario_plan,
    validate_scenarios,
    write_jsonl_atomic,
    write_manifest,
)
from schemas import CATEGORIES, scenario_text_batch_schema


async def main() -> None:
    config = load_config()
    ensure_directories(config)
    assignments = scenario_plan(config)
    batch_size = int(config["scenario_generation"]["batch_size"])
    client = OpenRouterClient(config)

    async def run_batch(batch: list[dict]) -> list[dict]:
        ids = [row["id"] for row in batch]
        prompt = render_prompt(
            "generate_scenarios.txt",
            {"ASSIGNMENTS_JSON": json_text(batch)},
        )
        response = await client.request_json(
            model_key="scenario_generator",
            temperature_key="scenario_generator",
            seed_key="scenario_generator",
            prompt=prompt,
            schema=scenario_text_batch_schema(ids),
            schema_name="generated_eval_scenarios",
        )
        text_by_id = {row["id"]: row["scenario"] for row in response["items"]}
        return [{**row, "scenario": text_by_id[row["id"]]} for row in batch]

    try:
        results = await asyncio.gather(
            *(run_batch(batch) for batch in batches(assignments, batch_size))
        )
    finally:
        await client.close()

    rows = [row for batch in results for row in batch]
    rows.sort(key=lambda row: row["id"])
    expected = len(CATEGORIES) * int(
        config["scenario_generation"]["scenarios_per_category"]
    )
    validate_scenarios(rows, expected)
    output_path = resolve_path(config, "raw_scenarios")
    write_jsonl_atomic(output_path, rows)
    write_manifest(
        config,
        stage="01_generate_scenarios",
        inputs={
            "training_domains": resolve_path(config, "training_domains"),
            "held_out_domains": resolve_path(config, "held_out_domains"),
        },
        outputs={"raw_scenarios": output_path},
        prompt_files=["generate_scenarios.txt"],
        stats={**client.stats(), "scenario_count": len(rows)},
    )
    print(f"Wrote {len(rows)} raw scenarios.")


if __name__ == "__main__":
    asyncio.run(main())
