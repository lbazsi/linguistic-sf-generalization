// Orchestrator — runs all pipeline stages in order.
//
//   node run.mjs                run the full pipeline (spec generation only calls Claude on
//                                a schema-hash cache miss)
//   node run.mjs --force-spec   ignore the cached spec and force a fresh Claude call
//   node run.mjs --skip-load    generate files only, don't touch the target database
//                                (useful for inspecting artifacts/data/*.copy by hand)

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const forceSpec = args.includes("--force-spec");
const skipLoad = args.includes("--skip-load");

const STAGES = [
  { script: "src/01-extract-schema.mjs", args: [] },
  { script: "src/02-build-dependency-graph.mjs", args: [] },
  { script: "src/03-generate-spec.mjs", args: forceSpec ? ["--force"] : [] },
  { script: "src/04-generate-data.mjs", args: [] },
  ...(skipLoad
    ? []
    : [
        { script: "src/05-load-data.mjs", args: [] },
        { script: "src/06-patch-deferred-fks.mjs", args: [] },
        { script: "src/07-validate.mjs", args: [] },
      ]),
];

function run(script, scriptArgs) {
  return new Promise((resolve, reject) => {
    console.log(`\n=== ${script} ${scriptArgs.join(" ")} ===`);
    const child = spawn("node", [script, ...scriptArgs], { cwd: __dirname, stdio: "inherit" });
    child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${script} exited with code ${code}`))));
  });
}

async function main() {
  for (const stage of STAGES) {
    await run(stage.script, stage.args);
  }
  console.log("\nPipeline complete.");
}

main().catch((err) => {
  console.error(`\nPipeline stopped: ${err.message}`);
  process.exit(1);
});
