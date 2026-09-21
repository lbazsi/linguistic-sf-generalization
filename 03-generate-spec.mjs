// Stage 3 — AI generation spec (the ONLY stage that calls an LLM)
//
// This is deliberately the model's entire job: read the schema once, and for every column
// that isn't a primary key or foreign key (those are handled mechanically in stage 4),
// choose a generator strategy from a fixed, machine-checkable vocabulary. The output is
// small (roughly one JSON object per column), bounded, and cached against a hash of the
// schema — so a run that touches a million rows still only ever makes one API call, and
// re-runs against an unchanged schema make zero.
//
// This is exactly the shape of task LLMs are reliable at (semantic judgment on a bounded
// input) and avoids the shape they are unreliable at (tracking thousands of emitted rows'
// worth of state across a long generation).

import "dotenv/config";
import fs from "node:fs/promises";
import crypto from "node:crypto";
import Anthropic from "@anthropic-ai/sdk";
import { SCHEMA_JSON_PATH, SPEC_JSON_PATH, ARTIFACTS_DIR } from "./lib/paths.mjs";

const MODEL = process.env.ANTHROPIC_MODEL || "claude-sonnet-5";

const STRATEGY_VOCAB = `
Every strategy object must have a "type" field, one of:

- {"type":"faker","method":"<dotted faker.js v9 method path>","args":[...]}
    e.g. {"type":"faker","method":"internet.email","args":[]}
         {"type":"faker","method":"commerce.price","args":[{"min":5,"max":500,"dec":2}]}
         {"type":"faker","method":"lorem.sentence","args":[]}
    "args" is an array of JSON-serializable arguments passed positionally to the faker method.

- {"type":"enum","values":[...],"weights":[...]}
    "weights" optional, must sum to ~1 and match values length if present.

- {"type":"boolean","trueRate":0.5}

- {"type":"number","min":0,"max":100,"decimals":0}

- {"type":"date","from":"2020-01-01","to":"2025-12-31"}
    Use for a plain date/timestamp column with no better semantic match.

- {"type":"timestampRecent","withinDays":30}
    Use for "created_at"/"updated_at"-style columns that should cluster near "now".

- {"type":"constant","value":<any JSON value>}
    Use for columns that should always hold one fixed value (e.g. a status default).

- {"type":"uuid"}
    Use only for non-PK columns that are semantically a UUID (PKs are handled separately).

Rules:
- Never emit a strategy for a column whose isPrimaryKey or isForeignKey is true — omit it
  entirely, the deterministic generator owns those.
- If a column is marked isUnique, prefer a naturally-unique faker method
  (internet.email, internet.username, string.uuid) so the generator rarely has to retry.
- Match strategy to column name AND data type — e.g. a "price" numeric column should use
  commerce.price or a number strategy with matching decimals, not lorem text.
- Respond with ONLY the JSON object. No markdown fences, no prose, no explanation.
`.trim();

function schemaHash(schema) {
  // Hash only the parts that affect generation decisions, so unrelated metadata changes
  // (e.g. re-running extraction at a different timestamp) don't invalidate the cache.
  const stable = schema.tables.map((t) => ({
    name: t.name,
    columns: t.columns.map((c) => ({
      name: c.name,
      dataType: c.dataType,
      isNullable: c.isNullable,
      isPrimaryKey: c.isPrimaryKey,
      isForeignKey: c.isForeignKey,
      isUnique: c.isUnique,
      maxLength: c.maxLength,
    })),
  }));
  return crypto.createHash("sha256").update(JSON.stringify(stable)).digest("hex");
}

function buildPrompt(schema) {
  const trimmed = schema.tables.map((t) => ({
    table: t.name,
    columns: t.columns
      .filter((c) => !c.isPrimaryKey && !c.isForeignKey)
      .map((c) => ({
        name: c.name,
        dataType: c.dataType,
        nullable: c.isNullable,
        unique: c.isUnique,
        maxLength: c.maxLength,
      })),
  }));

  return [
    "You are producing a JSON generation spec for a synthetic-data pipeline. Do not generate any",
    "actual data rows yourself — only decide, per column, which generator strategy to use.",
    "",
    "Schema (primary keys and foreign keys already excluded — they're handled separately):",
    JSON.stringify(trimmed, null, 2),
    "",
    "Strategy vocabulary:",
    STRATEGY_VOCAB,
    "",
    'Respond with a single JSON object: {"<table>": {"<column>": <strategy object>, ...}, ...}',
    "Include every non-PK, non-FK column listed above for every table.",
  ].join("\n");
}

async function loadCache() {
  try {
    return JSON.parse(await fs.readFile(SPEC_JSON_PATH, "utf8"));
  } catch {
    return null;
  }
}

async function callClaude(schema) {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY is not set — required on a spec cache miss.");
  }
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 8000,
    messages: [{ role: "user", content: buildPrompt(schema) }],
  });

  const text = response.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("");

  const cleaned = text.replace(/^```json\s*|```$/g, "").trim();
  let columnStrategies;
  try {
    columnStrategies = JSON.parse(cleaned);
  } catch (err) {
    throw new Error(`Model did not return valid JSON: ${err.message}\n---\n${text}`);
  }
  return columnStrategies;
}

async function main() {
  const force = process.argv.includes("--force");
  const schema = JSON.parse(await fs.readFile(SCHEMA_JSON_PATH, "utf8"));
  const hash = schemaHash(schema);

  const cached = await loadCache();
  if (!force && cached && cached.schemaHash === hash) {
    console.log(`Spec cache hit (schema hash ${hash.slice(0, 12)}...) — skipping Claude call.`);
    return;
  }

  console.log(
    cached
      ? "Schema changed since last cached spec — calling Claude for a fresh generation spec."
      : "No cached spec found — calling Claude for a generation spec."
  );

  const columnStrategies = await callClaude(schema);

  const spec = {
    schemaHash: hash,
    model: MODEL,
    generatedAt: new Date().toISOString(),
    tables: columnStrategies,
  };

  await fs.mkdir(ARTIFACTS_DIR, { recursive: true });
  await fs.writeFile(SPEC_JSON_PATH, JSON.stringify(spec, null, 2), "utf8");
  console.log(`Wrote ${SPEC_JSON_PATH}`);
}

main().catch((err) => {
  console.error("Spec generation failed:", err.message);
  process.exit(1);
});
