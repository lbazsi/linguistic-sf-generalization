// Stage 5 — Bulk loader
//
// Loads each table with a single COPY FROM STDIN instead of individual INSERT statements —
// this is the fix for the raw performance problem the original repo also hit, independent of
// the referential-integrity fix in stage 4. Runs in one transaction: a failure partway
// through rolls back the whole load rather than leaving the target database half-seeded.

import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { from as copyFrom } from "pg-copy-streams";
import { pipeline } from "node:stream/promises";
import { getTargetPool, closeAllPools } from "./lib/db.mjs";
import { GRAPH_JSON_PATH, DATA_DIR } from "./lib/paths.mjs";
import fsp from "node:fs/promises";

async function loadJson(filePath) {
  return JSON.parse(await fsp.readFile(filePath, "utf8"));
}

async function copyTable(client, tableName, manifestEntry) {
  const filePath = path.join(DATA_DIR, manifestEntry.file);
  const columnList = manifestEntry.columns.map((c) => `"${c}"`).join(", ");
  const copyStream = client.query(copyFrom(`COPY "${tableName}" (${columnList}) FROM STDIN`));
  const fileStream = fs.createReadStream(filePath);
  await pipeline(fileStream, copyStream);
}

async function resetSerialSequences(client, schema, tableName) {
  const table = schema.tables.find((t) => t.name === tableName);
  if (!table) return;
  for (const column of table.columns) {
    if (!column.isSerial) continue;
    await client.query(
      `SELECT setval(pg_get_serial_sequence($1, $2), COALESCE((SELECT MAX("${column.name}") FROM "${tableName}"), 1))`,
      [tableName, column.name]
    );
  }
}

async function main() {
  const graph = await loadJson(GRAPH_JSON_PATH);
  const manifest = await loadJson(path.join(DATA_DIR, "manifest.json"));
  const schema = await loadJson(path.join(path.dirname(GRAPH_JSON_PATH), "schema.json"));

  const pool = getTargetPool();
  const client = await pool.connect();

  try {
    await client.query("BEGIN");
    for (const tableName of graph.insertionOrder) {
      const entry = manifest[tableName];
      if (!entry) {
        console.warn(`No generated data for "${tableName}" — skipping.`);
        continue;
      }
      await copyTable(client, tableName, entry);
      console.log(`Loaded ${entry.rowCount} rows into "${tableName}"`);
    }

    for (const tableName of graph.insertionOrder) {
      await resetSerialSequences(client, schema, tableName);
    }

    await client.query("COMMIT");
    console.log("Load committed.");
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }

  await closeAllPools();
}

main().catch((err) => {
  console.error("Load failed, transaction rolled back:", err.message);
  process.exit(1);
});
