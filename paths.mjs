import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(__dirname, "..", "..");
export const ARTIFACTS_DIR = path.join(ROOT, "artifacts");
export const DATA_DIR = path.join(ARTIFACTS_DIR, "data");

export const SCHEMA_SQL_PATH = path.join(ARTIFACTS_DIR, "schema.sql");
export const SCHEMA_JSON_PATH = path.join(ARTIFACTS_DIR, "schema.json");
export const GRAPH_JSON_PATH = path.join(ARTIFACTS_DIR, "dependency-graph.json");
export const SPEC_JSON_PATH = path.join(ARTIFACTS_DIR, "generation-spec.json");
export const VALIDATION_REPORT_PATH = path.join(ARTIFACTS_DIR, "validation-report.json");
export const CONFIG_PATH = path.join(ROOT, "generation.config.json");
