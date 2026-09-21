// Stage 1 — Schema extraction
//
// Produces two artifacts:
//   artifacts/schema.sql   a `pg_dump --schema-only` for humans / archival (same as the
//                          original neondatabase/vibe-coding-synthetic-data-part-1 approach)
//   artifacts/schema.json  a structured introspection via information_schema, which is what
//                          every later stage actually reads. Parsing pg_dump's SQL text is
//                          fragile; querying the catalog directly is not.

import "dotenv/config";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs/promises";
import { getSourcePool, closeAllPools } from "./lib/db.mjs";
import { ARTIFACTS_DIR, SCHEMA_SQL_PATH, SCHEMA_JSON_PATH } from "./lib/paths.mjs";

const execFileAsync = promisify(execFile);

async function dumpSchemaSql() {
  const url = process.env.SOURCE_DATABASE_URL;
  try {
    const { stdout } = await execFileAsync("pg_dump", ["--schema-only", "--no-owner", "--no-privileges", url], {
      maxBuffer: 1024 * 1024 * 64,
    });
    await fs.writeFile(SCHEMA_SQL_PATH, stdout, "utf8");
    console.log(`Wrote ${SCHEMA_SQL_PATH}`);
  } catch (err) {
    console.warn(
      `pg_dump unavailable or failed (${err.message}). Continuing with information_schema introspection only ` +
        "— schema.sql is archival and not required by later stages."
    );
  }
}

const COLUMNS_QUERY = `
  select
    c.table_name,
    c.column_name,
    c.data_type,
    c.udt_name,
    c.is_nullable = 'YES'            as is_nullable,
    c.column_default,
    c.character_maximum_length,
    c.numeric_precision,
    c.numeric_scale,
    (c.column_default is not null and c.column_default like 'nextval%') as is_serial
  from information_schema.columns c
  join information_schema.tables t
    on t.table_schema = c.table_schema and t.table_name = c.table_name
  where c.table_schema = 'public' and t.table_type = 'BASE TABLE'
  order by c.table_name, c.ordinal_position;
`;

const PRIMARY_KEYS_QUERY = `
  select tc.table_name, kcu.column_name, kcu.ordinal_position
  from information_schema.table_constraints tc
  join information_schema.key_column_usage kcu
    on tc.constraint_name = kcu.constraint_name and tc.table_schema = kcu.table_schema
  where tc.constraint_type = 'PRIMARY KEY' and tc.table_schema = 'public'
  order by tc.table_name, kcu.ordinal_position;
`;

const FOREIGN_KEYS_QUERY = `
  select
    tc.constraint_name,
    tc.table_name        as table_name,
    kcu.column_name       as column_name,
    kcu.ordinal_position   as ordinal_position,
    ccu.table_name        as ref_table_name,
    ccu.column_name       as ref_column_name,
    rc.delete_rule
  from information_schema.table_constraints tc
  join information_schema.key_column_usage kcu
    on tc.constraint_name = kcu.constraint_name and tc.table_schema = kcu.table_schema
  join information_schema.constraint_column_usage ccu
    on tc.constraint_name = ccu.constraint_name and tc.table_schema = ccu.table_schema
  join information_schema.referential_constraints rc
    on tc.constraint_name = rc.constraint_name and tc.table_schema = rc.constraint_schema
  where tc.constraint_type = 'FOREIGN KEY' and tc.table_schema = 'public'
  order by tc.table_name, tc.constraint_name, kcu.ordinal_position;
`;

const UNIQUE_QUERY = `
  select tc.table_name, tc.constraint_name, kcu.column_name, kcu.ordinal_position
  from information_schema.table_constraints tc
  join information_schema.key_column_usage kcu
    on tc.constraint_name = kcu.constraint_name and tc.table_schema = kcu.table_schema
  where tc.constraint_type = 'UNIQUE' and tc.table_schema = 'public'
  order by tc.table_name, tc.constraint_name, kcu.ordinal_position;
`;

const CHECK_QUERY = `
  select tc.table_name, cc.check_clause
  from information_schema.table_constraints tc
  join information_schema.check_constraints cc
    on tc.constraint_name = cc.constraint_name and tc.table_schema = cc.constraint_schema
  where tc.constraint_type = 'CHECK' and tc.table_schema = 'public';
`;

function groupBy(rows, key) {
  const map = new Map();
  for (const row of rows) {
    const k = row[key];
    if (!map.has(k)) map.set(k, []);
    map.get(k).push(row);
  }
  return map;
}

async function introspect() {
  const pool = getSourcePool();

  const [columnsRes, pkRes, fkRes, uniqueRes, checkRes] = await Promise.all([
    pool.query(COLUMNS_QUERY),
    pool.query(PRIMARY_KEYS_QUERY),
    pool.query(FOREIGN_KEYS_QUERY),
    pool.query(UNIQUE_QUERY),
    pool.query(CHECK_QUERY),
  ]);

  const columnsByTable = groupBy(columnsRes.rows, "table_name");
  const pkByTable = groupBy(pkRes.rows, "table_name");
  const fkByTable = groupBy(fkRes.rows, "table_name");
  const uniqueByTable = groupBy(uniqueRes.rows, "table_name");
  const checkByTable = groupBy(checkRes.rows, "table_name");

  const tables = [...columnsByTable.keys()].sort().map((tableName) => {
    const pkColumns = (pkByTable.get(tableName) || []).map((r) => r.column_name);

    const fkGroups = groupBy(fkByTable.get(tableName) || [], "constraint_name");
    const foreignKeys = [...fkGroups.values()].map((rows) => ({
      constraintName: rows[0].constraint_name,
      columns: rows.map((r) => r.column_name),
      refTable: rows[0].ref_table_name,
      refColumns: rows.map((r) => r.ref_column_name),
      deleteRule: rows[0].delete_rule,
      isSelfReference: rows[0].ref_table_name === tableName,
    }));

    const uniqueGroups = groupBy(uniqueByTable.get(tableName) || [], "constraint_name");
    const uniqueConstraints = [...uniqueGroups.values()].map((rows) => rows.map((r) => r.column_name));

    const fkColumnNames = new Set(foreignKeys.flatMap((fk) => fk.columns));

    const columns = (columnsByTable.get(tableName) || []).map((c) => ({
      name: c.column_name,
      dataType: c.data_type,
      udtName: c.udt_name,
      isNullable: c.is_nullable,
      columnDefault: c.column_default,
      maxLength: c.character_maximum_length,
      numericPrecision: c.numeric_precision,
      numericScale: c.numeric_scale,
      isSerial: c.is_serial,
      isPrimaryKey: pkColumns.includes(c.column_name),
      isForeignKey: fkColumnNames.has(c.column_name),
      isUnique: uniqueConstraints.some((cols) => cols.length === 1 && cols[0] === c.column_name),
    }));

    return {
      name: tableName,
      columns,
      primaryKey: pkColumns,
      foreignKeys,
      uniqueConstraints,
      checkConstraints: (checkByTable.get(tableName) || []).map((r) => r.check_clause),
    };
  });

  return { extractedAt: new Date().toISOString(), tables };
}

async function main() {
  await fs.mkdir(ARTIFACTS_DIR, { recursive: true });
  await dumpSchemaSql();
  const schema = await introspect();
  await fs.writeFile(SCHEMA_JSON_PATH, JSON.stringify(schema, null, 2), "utf8");
  console.log(`Wrote ${SCHEMA_JSON_PATH} (${schema.tables.length} tables)`);
  await closeAllPools();
}

main().catch((err) => {
  console.error("Schema extraction failed:", err);
  process.exit(1);
});
