# Synthetic data pipeline

A schema-aware synthetic data generator for Postgres, built as a follow-up to
[neondatabase/vibe-coding-synthetic-data-part-1](https://github.com/neondatabase/vibe-coding-synthetic-data-part-1).

That project's own writeup found that asking Claude to generate rows directly — either as raw
SQL `INSERT`s, or as faker.js-backed JS objects — works fine at small scale and then loses
referential integrity: the AI-only approach failed past ~75-100 rows, the hybrid
(AI + faker.js) approach past ~3,000-5,000 rows. In both cases the model lost track of which
foreign keys it had already emitted.

This pipeline keeps the same overall idea (schema in, realistic `INSERT`-ready data out) but
splits the work along a different line:

- **Claude decides *how* to generate each column** — a one-time, cacheable, bounded task
  (roughly one JSON strategy object per column) that plays to what LLMs are actually good at:
  semantic judgment on a fixed-size input.
- **Deterministic code decides *how much* and enforces integrity** — every foreign key is
  sampled from an in-memory pool of primary keys this same process already generated, so
  referential integrity holds by construction at 10 rows or 10 million rows.

## Pipeline stages

| # | Script | What it does |
|---|--------|---------------|
| 1 | `src/01-extract-schema.mjs` | `pg_dump --schema-only` (archival) + `information_schema` introspection into `artifacts/schema.json` |
| 2 | `src/02-build-dependency-graph.mjs` | Topologically sorts tables by FK; defers self-referential / cyclic FK edges instead of failing |
| 3 | `src/03-generate-spec.mjs` | Calls Claude **once** for a per-column generator spec, cached by schema hash in `artifacts/generation-spec.json` |
| 4 | `src/04-generate-data.mjs` | Deterministically generates rows in dependency order, streaming each table to `artifacts/data/*.copy` |
| 5 | `src/05-load-data.mjs` | Bulk-loads via `COPY FROM STDIN` inside one transaction |
| 6 | `src/06-patch-deferred-fks.mjs` | Fills in deferred FK columns with a set-based `UPDATE` against the loaded target DB |
| 7 | `src/07-validate.mjs` | Row-count and FK-orphan checks against the target DB; exits non-zero on failure |

Run them all with `node run.mjs`, or individually via the npm scripts in `package.json`.

## Getting started

1. `npm install`
2. `cp .env.example .env` and fill in:
   - `SOURCE_DATABASE_URL` — a Postgres DB with the schema you want to clone (data optional).
     No schema handy? Load `schema.example.sql` into a scratch database and point it there.
   - `TARGET_DATABASE_URL` — an **empty** Postgres DB with the same schema already applied,
     same major version as the source.
   - `ANTHROPIC_API_KEY` — only needed the first time, or after a schema change (cache miss).
3. Optionally edit `generation.config.json` to set row counts per table.
4. `node run.mjs`

Useful flags:
- `node run.mjs --force-spec` — ignore the cached spec, force a fresh Claude call.
- `node run.mjs --skip-load` — generate `artifacts/data/*.copy` files only, don't touch the
  target database (handy for inspecting output by hand).

### Working with Actions

Same shape as the original repo: add the secrets from `.env.example` to GitHub Secrets, then
trigger `.github/workflows/generate-synthetic-data.yml` manually or on a schema-change trigger.
The spec is cached as a build artifact keyed on the schema hash, so routine re-runs (same
schema, new random data) never call the Anthropic API.

## Design notes

- **Why not have the AI write SQL `INSERT`s or faker.js code directly?** That was the original
  repo's approach and its own tests are the evidence against it — the model has to hold every
  previously-emitted primary key in its context to stay consistent, and that degrades with
  scale. Here the model never sees or produces row data at all, only a strategy per column.
- **Composite primary/foreign keys**: single-column keys are fully supported. Composite keys
  are pooled as tuples for FK sampling, but the deferred-edge patch in stage 6 currently only
  handles single-column deferred FKs — a composite self-referential FK will log a warning and
  needs a manual patch.
- **Scaling further**: stage 4 streams each table straight to disk and never holds full row
  data in memory (only the small ID pools), so row counts are limited by disk and Postgres
  COPY performance, not by anything in this pipeline.
- **Determinism vs. realism**: `NULLABLE_FILL_RATE` and the AI-chosen strategies are the two
  knobs most worth tuning per project — see `generation.config.json` for row counts and
  `.env` for the null rate.
