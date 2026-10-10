from __future__ import annotations

import asyncio

from common import (
    OpenRouterClient,
    batches,
    ensure_directories,
    json_text,
    load_config,
    read_jsonl,
    render_prompt,
    resolve_path,
    validate_scenarios,
    write_jsonl_atomic,
    write_manifest,
)
from schemas import scenario_text_batch_schema


async def main() -> None:
    config = load_config()
    ensure_directories(config)
    rows = read_jsonl(resolve_path(config, "raw_scenarios"))
    batch_size = int(config["scenario_generation"]["batch_size"])
    client = OpenRouterClient(config)

    async def run_batch(batch: list[dict]) -> list[dict]:
        ids = [row["id"] for row in batch]
        prompt = render_prompt(
            "review_scenarios.txt",
            {"SCENARIOS_JSON": json_text(batch)},
        )
        response = await client.request_json(
            model_key="scenario_judge",
            temperature_key="scenario_judge",
            seed_key="scenario_judge",
            prompt=prompt,
            schema=scenario_text_batch_schema(len(batch)),
            schema_name="reviewed_eval_scenarios",
        )
        items = response["items"]
        if len(items) != len(batch):
            raise RuntimeError(
                f"scenario review batch: expected {len(batch)} items, got {len(items)}"
            )
        return [
            {**row, "scenario": item["scenario"]}
            for row, item in zip(batch, items, strict=True)
        ]

    try:
        results = await asyncio.gather(
            *(run_batch(batch) for batch in batches(rows, batch_size))
        )
    finally:
        await client.close()

    reviewed = [row for batch in results for row in batch]
    reviewed.sort(key=lambda row: row["id"])
    validate_scenarios(reviewed, len(rows))
    output_path = resolve_path(config, "final_scenarios")
    write_jsonl_atomic(output_path, reviewed)
    write_manifest(
        config,
        stage="02_review_scenarios",
        inputs={"raw_scenarios": resolve_path(config, "raw_scenarios")},
        outputs={"final_scenarios": output_path},
        prompt_files=["review_scenarios.txt"],
        stats={**client.stats(), "scenario_count": len(reviewed)},
    )
    print(f"Wrote {len(reviewed)} reviewed scenarios.")


if __name__ == "__main__":
    asyncio.run(main())
