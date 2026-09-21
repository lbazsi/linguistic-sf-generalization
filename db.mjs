import pg from "pg";

const { Pool } = pg;

let sourcePool;
let targetPool;

export function getSourcePool() {
  if (!process.env.SOURCE_DATABASE_URL) {
    throw new Error("SOURCE_DATABASE_URL is not set (check your .env)");
  }
  if (!sourcePool) {
    sourcePool = new Pool({ connectionString: process.env.SOURCE_DATABASE_URL });
  }
  return sourcePool;
}

export function getTargetPool() {
  if (!process.env.TARGET_DATABASE_URL) {
    throw new Error("TARGET_DATABASE_URL is not set (check your .env)");
  }
  if (!targetPool) {
    targetPool = new Pool({ connectionString: process.env.TARGET_DATABASE_URL });
  }
  return targetPool;
}

export async function closeAllPools() {
  if (sourcePool) await sourcePool.end();
  if (targetPool) await targetPool.end();
}
