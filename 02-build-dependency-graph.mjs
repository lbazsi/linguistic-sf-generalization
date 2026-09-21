// Stage 2 — Dependency graph
//
// This is the piece the original AI-only and hybrid approaches were missing: instead of
// asking the model to remember "which tables must exist before which other tables" across a
// multi-thousand-row generation run, we compute it once, mechanically, with a topological
// sort. Cycles (self-referential FKs like employees.manager_id, or genuine cross-table
// cycles) are broken by deferring nullable FK edges to a post-load "patch" pass — those
// columns get NULL on first insert and a backward-pointing value assigned afterwards, which
// keeps the graph a DAG by construction (a row can only ever point at an earlier row).

import fs from "node:fs/promises";
import { SCHEMA_JSON_PATH, GRAPH_JSON_PATH, ARTIFACTS_DIR } from "./lib/paths.mjs";

function loadSchema() {
  return fs.readFile(SCHEMA_JSON_PATH, "utf8").then(JSON.parse);
}

/** Every FK column in the constraint is nullable => safe to null-then-patch. */
function isDeferrable(table, fk) {
  const colByName = new Map(table.columns.map((c) => [c.name, c]));
  return fk.columns.every((colName) => colByName.get(colName)?.isNullable);
}

function buildEdges(schema) {
  const edges = []; // { from, to, constraintName, columns, refColumns, deferred, reason }
  for (const table of schema.tables) {
    for (const fk of table.foreignKeys) {
      if (fk.isSelfReference) {
        edges.push({
          from: table.name,
          to: fk.refTable,
          constraintName: fk.constraintName,
          columns: fk.columns,
          refColumns: fk.refColumns,
          deferred: true,
          reason: "self-reference",
        });
        continue;
      }
      edges.push({
        from: table.name,
        to: fk.refTable,
        constraintName: fk.constraintName,
        columns: fk.columns,
        refColumns: fk.refColumns,
        deferred: false,
        reason: null,
      });
    }
  }
  return edges;
}

/**
 * Kahn's algorithm over the non-deferred edges. If nodes remain with no zero in-degree
 * (a genuine cycle), progressively defer the cycle's nullable edges and retry. Throws only
 * if a cycle survives with no nullable edge to break it — at that point the schema itself is
 * unsatisfiable without a manual insertion-order hint, and that's worth a human's attention
 * rather than a silent guess.
 */
function topologicalSort(tableNames, edges) {
  let workingEdges = edges.filter((e) => !e.deferred);
  const extraDeferred = [];

  for (let attempt = 0; attempt < tableNames.length + 1; attempt++) {
    const inDegree = new Map(tableNames.map((t) => [t, 0]));
    const dependents = new Map(tableNames.map((t) => [t, []])); // to -> [from,...]

    for (const e of workingEdges) {
      if (e.from === e.to) continue; // already-handled self-reference
      inDegree.set(e.from, (inDegree.get(e.from) || 0) + 1);
      dependents.get(e.to).push(e.from);
    }

    const queue = tableNames.filter((t) => inDegree.get(t) === 0).sort();
    const order = [];
    const inDegreeWorking = new Map(inDegree);

    while (queue.length) {
      const node = queue.shift();
      order.push(node);
      for (const dep of dependents.get(node)) {
        inDegreeWorking.set(dep, inDegreeWorking.get(dep) - 1);
        if (inDegreeWorking.get(dep) === 0) {
          queue.push(dep);
          queue.sort();
        }
      }
    }

    if (order.length === tableNames.length) {
      return { order, deferredEdges: [...edges.filter((e) => e.deferred), ...extraDeferred] };
    }

    // Cycle remains among tables not yet in `order`. Try to break it by deferring one
    // nullable edge that participates in the remaining subgraph.
    const stuck = new Set(tableNames.filter((t) => !order.includes(t)));
    const breakable = workingEdges.find(
      (e) => stuck.has(e.from) && stuck.has(e.to) && e.from !== e.to && edgeIsNullable(e)
    );

    if (!breakable) {
      const cycleTables = [...stuck].join(", ");
      throw new Error(
        `Unbreakable foreign-key cycle among tables with no nullable FK to defer: [${cycleTables}]. ` +
          "Resolve manually, e.g. by making one of the participating FK columns nullable, " +
          "or add an explicit insertion-order override in generation.config.json."
      );
    }

    breakable.deferred = true;
    extraDeferred.push(breakable);
    workingEdges = workingEdges.filter((e) => e !== breakable);
  }

  throw new Error("Topological sort did not converge — this should not happen.");
}

function edgeIsNullable(edge) {
  return edge.__nullable ?? edge.__nullableCache;
}

async function main() {
  const schema = await loadSchema();
  const tableNames = schema.tables.map((t) => t.name);
  const tableByName = new Map(schema.tables.map((t) => [t.name, t]));

  const edges = buildEdges(schema);
  // Precompute nullability per edge (needed by edgeIsNullable during cycle-breaking).
  for (const e of edges) {
    const table = tableByName.get(e.from);
    e.__nullable = isDeferrable(table, { columns: e.columns });
  }

  const { order, deferredEdges } = topologicalSort(tableNames, edges);

  const graph = {
    builtAt: new Date().toISOString(),
    insertionOrder: order,
    deferredEdges: deferredEdges.map((e) => ({
      fromTable: e.from,
      toTable: e.to,
      constraintName: e.constraintName,
      columns: e.columns,
      refColumns: e.refColumns,
      reason: e.reason || "cycle-break (nullable FK)",
    })),
  };

  await fs.mkdir(ARTIFACTS_DIR, { recursive: true });
  await fs.writeFile(GRAPH_JSON_PATH, JSON.stringify(graph, null, 2), "utf8");
  console.log(`Wrote ${GRAPH_JSON_PATH}`);
  console.log(`Insertion order: ${order.join(" -> ")}`);
  if (deferredEdges.length) {
    console.log(
      `Deferred FK columns (nulled on insert, patched after load): ` +
        deferredEdges.map((e) => `${e.from}.${e.columns.join(",")}`).join("; ")
    );
  }
}

main().catch((err) => {
  console.error("Dependency graph build failed:", err.message);
  process.exit(1);
});
