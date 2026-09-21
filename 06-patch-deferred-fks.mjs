// Stage 6 — Patch deferred FK columns
//
// Self-referential FKs (employees.manager_id -> employees.id) and any FK edge stage 2 had to
// remove to break a cycle were loaded as NULL in stage 5. Doing this as a set-based UPDATE
// against the already-loaded target database — rather than trying to fill in the "right"
// value during generation — sidesteps the ordering problem entirely: it doesn't matter which
// of the two tables in a cycle was inserted first, because by this point both are fully
// loaded and every candidate id genuinely exists.
//
// The mapping is a simple round-robin over a randomly ordered candidate list, which is
// enough to produce plausible, non-degenerate references without a per-row correlated
// subquery (slow at scale). A configurable fraction of eligible rows are left NULL to match
// NULLABLE_FILL_RATE used elsewhere, since these columns are nullable by construction (that's
// exactly why stage 2 was able to defer them).

import "dotenv/config";
import { getTargetPool, closeAllPools } from "./lib/db.mjs";
import { GRAPH_JSON_PATH } from "./lib/paths.mjs";
import fs from "node:fs/promises";

const NULLABLE_FILL_RATE = Number(process.env.NULLABLE_FILL_RATE ?? 0.15);

async function loadJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function patchEdge(client, edge) {
  if (edge.columns.length !== 1) {
    console.warn(
      `Skipping composite deferred FK ${edge.fromTable}.(${edge.columns.join(",")}) — ` +
        "patch this manually; composite self-referential/cyclic FKs aren't auto-handled."
    );
    return;
  }
  const [fromCol] = edge.columns;
  const [toCol] = edge.refColumns;
  const excludeSelf = edge.fromTable === edge.toTable;

  // Candidate pool, shuffled once.
  const candidatesRes = await client.query(
    `SELECT "${toCol}" AS id, row_number() OVER (ORDER BY random()) AS rn
     FROM "${edge.toTable}"`
  );
  if (candidatesRes.rows.length === 0) {
    console.warn(`No candidate rows in "${edge.toTable}" — leaving ${edge.fromTable}.${fromCol} NULL.`);
    return;
  }
  const candidateCount = candidatesRes.rows.length;

  // Update in one statement: assign each target row a round-robin candidate, skip a random
  // fraction to leave NULLs, and (for self-references) exclude a row referencing itself.
  const sql = `
    WITH targets AS (
      SELECT ctid, row_number() OVER (ORDER BY random()) AS rn
      FROM "${edge.fromTable}"
      WHERE random() >= $1
    ),
    candidates AS (
      SELECT "${toCol}" AS id, row_number() OVER (ORDER BY random()) AS rn, count(*) OVER () AS cnt
      FROM "${edge.toTable}"
    )
    UPDATE "${edge.fromTable}" t
    SET "${fromCol}" = c.id
    FROM targets tg
    JOIN candidates c ON c.rn = ((tg.rn - 1) % c.cnt) + 1
    WHERE t.ctid = tg.ctid
      ${excludeSelf ? `AND c.id <> t."${edge.refColumns[0]}"` : ""};
  `;
  const result = await client.query(sql, [NULLABLE_FILL_RATE]);
  console.log(
    `Patched ${result.rowCount} rows: "${edge.fromTable}.${fromCol}" -> "${edge.toTable}.${toCol}" ` +
      `(${candidateCount} candidates, ~${Math.round(NULLABLE_FILL_RATE * 100)}% left NULL)`
  );
}

async function main() {
  const graph = await loadJson(GRAPH_JSON_PATH);
  if (!graph.deferredEdges.length) {
    console.log("No deferred FK columns — nothing to patch.");
    return;
  }

  const pool = getTargetPool();
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    for (const edge of graph.deferredEdges) {
      await patchEdge(client, edge);
    }
    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
  await closeAllPools();
}

main().catch((err) => {
  console.error("Deferred FK patch failed, transaction rolled back:", err.message);
  process.exit(1);
});
