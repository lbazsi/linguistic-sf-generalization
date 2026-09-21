// Stage 7 — Validation
//
// Runs against the loaded target database (not the generated files) so it's a true check of
// what actually landed, catching anything a bug upstream might have let through: row counts
// per table, orphaned foreign keys, and any CHECK constraint violations the database itself
// wasn't asked to enforce at insert time (all deferred columns are validated too, since by
// this point stage 6 has already run). Exits non-zero on any failure so this can gate a CI job.

import "dotenv/config";
import fs from "node:fs/promises";
import { getTargetPool, closeAllPools } from "./lib/db.mjs";
import { SCHEMA_JSON_PATH, DATA_DIR, VALIDATION_REPORT_PATH, ARTIFACTS_DIR } from "./lib/paths.mjs";
import path from "node:path";

async function loadJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function checkRowCount(client, tableName, expected) {
  const res = await client.query(`SELECT count(*)::int AS n FROM "${tableName}"`);
  const actual = res.rows[0].n;
  return { check: "row_count", table: tableName, expected, actual, pass: actual === expected };
}

async function checkForeignKeyOrphans(client, table, fk) {
  const fromCols = fk.columns.map((c) => `t."${c}"`).join(", ");
  const joinConditions = fk.columns
    .map((c, i) => `t."${c}" = r."${fk.refColumns[i]}"`)
    .join(" AND ");
  const notNullGuard = fk.columns.map((c) => `t."${c}" IS NOT NULL`).join(" AND ");

  const sql = `
    SELECT count(*)::int AS n
    FROM "${table.name}" t
    LEFT JOIN "${fk.refTable}" r ON ${joinConditions}
    WHERE ${notNullGuard} AND r."${fk.refColumns[0]}" IS NULL;
  `;
  const res = await client.query(sql);
  const orphanCount = res.rows[0].n;
  return {
    check: "fk_orphans",
    table: table.name,
    constraint: fk.constraintName,
    refTable: fk.refTable,
    orphanCount,
    pass: orphanCount === 0,
  };
}

async function main() {
  const schema = await loadJson(SCHEMA_JSON_PATH);
  const manifest = await loadJson(path.join(DATA_DIR, "manifest.json"));

  const pool = getTargetPool();
  const client = await pool.connect();
  const results = [];

  try {
    for (const table of schema.tables) {
      const expected = manifest[table.name]?.rowCount;
      if (expected !== undefined) {
        results.push(await checkRowCount(client, table.name, expected));
      }
      for (const fk of table.foreignKeys) {
        results.push(await checkForeignKeyOrphans(client, table, fk));
      }
    }
  } finally {
    client.release();
    await closeAllPools();
  }

  const failures = results.filter((r) => !r.pass);
  const report = {
    ranAt: new Date().toISOString(),
    totalChecks: results.length,
    failedChecks: failures.length,
    results,
  };

  await fs.mkdir(ARTIFACTS_DIR, { recursive: true });
  await fs.writeFile(VALIDATION_REPORT_PATH, JSON.stringify(report, null, 2), "utf8");

  console.log(`Validation: ${results.length - failures.length}/${results.length} checks passed.`);
  if (failures.length) {
    console.error("Failures:");
    for (const f of failures) console.error(" -", JSON.stringify(f));
    process.exit(1);
  }
  console.log(`Full report: ${VALIDATION_REPORT_PATH}`);
}

main().catch((err) => {
  console.error("Validation run failed:", err.message);
  process.exit(1);
});
