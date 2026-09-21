// Stage 4 — Deterministic generator
//
// This is where referential integrity is GUARANTEED rather than hoped for. Every foreign key
// value is sampled from an in-memory pool of primary keys this same process already assigned
// to the referenced table — it is array indexing, not model memory, so it scales identically
// at 10 rows or 10 million rows. Tables are processed in the dependency order computed in
// stage 2, so a referenced table's ID pool always exists before it's needed.
//
// Self-referential and cycle-breaking FK columns (flagged as "deferred" in stage 2) are
// always written as NULL here and filled in afterwards by stage 6, against the loaded target
// database, using a set-based UPDATE — see that file for why.

import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import { faker } from "@faker-js/faker";
import { SCHEMA_JSON_PATH, GRAPH_JSON_PATH, SPEC_JSON_PATH, CONFIG_PATH, DATA_DIR } from "./lib/paths.mjs";
import { formatCopyRow } from "./lib/copy-format.mjs";
import { defaultStrategyForColumn, valueForColumn } from "./lib/generators.mjs";

const DEFAULT_ROWS = Number(process.env.ROWS_PER_TABLE || 1000);
const NULLABLE_FILL_RATE = Number(process.env.NULLABLE_FILL_RATE ?? 0.15);

async function loadJson(filePath, fallback = null) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch (err) {
    if (fallback !== null) return fallback;
    throw new Error(`Could not read ${filePath}: ${err.message}`);
  }
}

function rowCountFor(tableName, config) {
  return config.rowsPerTable?.[tableName] ?? config.defaultRows ?? DEFAULT_ROWS;
}

function nextPrimaryKeyValue(column, counter) {
  const type = column.dataType.toLowerCase();
  if (type.includes("uuid")) return faker.string.uuid();
  if (type.includes("char") || type === "text") return `${column.name}_${counter}`;
  return counter; // integer-like PK (serial, bigint, int)
}

/** Non-key columns needing a value; PK/FK are handled by the caller. */
function plainColumns(table) {
  return table.columns.filter((c) => !c.isPrimaryKey && !c.isForeignKey);
}

async function main() {
  const schema = await loadJson(SCHEMA_JSON_PATH);
  const graph = await loadJson(GRAPH_JSON_PATH);
  const spec = await loadJson(SPEC_JSON_PATH);
  const config = await loadJson(CONFIG_PATH, {});

  const tableByName = new Map(schema.tables.map((t) => [t.name, t]));

  const deferredByTable = new Map(); // fromTable -> [{columns, toTable}, ...]
  for (const e of graph.deferredEdges) {
    if (!deferredByTable.has(e.fromTable)) deferredByTable.set(e.fromTable, []);
    deferredByTable.get(e.fromTable).push(e);
  }

  await fs.mkdir(DATA_DIR, { recursive: true });

  const idPools = new Map(); // table -> array of PK values (scalar, or array for composite PK)
  const manifest = {}; // table -> { file, rowCount, columns }

  for (const tableName of graph.insertionOrder) {
    const table = tableByName.get(tableName);
    if (!table) throw new Error(`Table "${tableName}" in dependency graph but not in schema.json`);

    const rowCount = rowCountFor(tableName, config);
    const deferredColumns = new Set((deferredByTable.get(tableName) || []).flatMap((e) => e.columns));
    const seenByColumn = new Map(); // columnName -> Set (uniqueness tracking)
    const pool = [];

    const outputColumns = table.columns.map((c) => c.name);
    const filePath = path.join(DATA_DIR, `${tableName}.copy`);
    const fh = await fs.open(filePath, "w");

    let pkCounter = 1;
    for (let i = 0; i < rowCount; i++) {
      const row = {};

      // Primary key(s) — always assigned by us, never delegated to a DB default, so the
      // value is known immediately and can be pooled for children.
      for (const pkCol of table.primaryKey) {
        const column = table.columns.find((c) => c.name === pkCol);
        row[pkCol] = nextPrimaryKeyValue(column, pkCounter);
      }
      pool.push(table.primaryKey.length === 1 ? row[table.primaryKey[0]] : table.primaryKey.map((c) => row[c]));
      pkCounter++;

      // Foreign keys (non-deferred): sample from the referenced table's already-built pool.
      for (const fk of table.foreignKeys) {
        if (fk.isSelfReference) {
          for (const col of fk.columns) row[col] = null; // patched in stage 6
          continue;
        }
        if (fk.columns.some((c) => deferredColumns.has(c))) {
          for (const col of fk.columns) row[col] = null; // patched in stage 6
          continue;
        }

        const refPool = idPools.get(fk.refTable);
        if (!refPool || refPool.length === 0) {
          throw new Error(
            `"${tableName}.${fk.columns.join(",")}" references "${fk.refTable}", which has no rows. ` +
              "Check the dependency graph's insertion order, or set rowsPerTable for that table > 0."
          );
        }

        const nullable = fk.columns.every((c) => table.columns.find((tc) => tc.name === c)?.isNullable);
        if (nullable && Math.random() < NULLABLE_FILL_RATE) {
          for (const col of fk.columns) row[col] = null;
          continue;
        }

        const picked = refPool[Math.floor(Math.random() * refPool.length)];
        if (fk.columns.length === 1) {
          row[fk.columns[0]] = picked;
        } else {
          fk.columns.forEach((col, idx) => (row[col] = picked[idx]));
        }
      }

      // Everything else: driven by the cached AI spec, falling back to a type-based default
      // if the spec predates this column (e.g. schema evolved after the spec was cached).
      for (const column of plainColumns(table)) {
        if (column.isNullable && Math.random() < NULLABLE_FILL_RATE) {
          row[column.name] = null;
          continue;
        }
        const strategy = spec.tables?.[tableName]?.[column.name] || defaultStrategyForColumn(column);
        if (!seenByColumn.has(column.name)) seenByColumn.set(column.name, new Set());
        row[column.name] = valueForColumn(column, strategy, seenByColumn.get(column.name));
      }

      await fh.write(formatCopyRow(outputColumns.map((col) => row[col])));
    }

    await fh.close();
    idPools.set(tableName, pool);
    manifest[tableName] = { file: path.relative(DATA_DIR, filePath), rowCount, columns: outputColumns };
    console.log(`Generated ${rowCount} rows for "${tableName}" -> ${filePath}`);
  }

  await fs.writeFile(path.join(DATA_DIR, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  console.log(`Wrote ${path.join(DATA_DIR, "manifest.json")}`);
}

main().catch((err) => {
  console.error("Data generation failed:", err.message);
  process.exit(1);
});
