# Synthetic data generation for LoRA training

This directory builds paired synthetic JSONL datasets for controlled linguistic-feature LoRA experiments. The canonical corpus is generated over a structured animal-welfare domain space, reviewed for non-deterministic quality problems, and then transformed into one dataset per linguistic feature.

## Data contract

The initial canonical generation is stored in `data/canonical/generated.jsonl`. The reviewed authoritative corpus is stored in `data/canonical/corpus.jsonl` and contains `id`, `language`, `animal`, `value`, `context`, and `canonical`. `config/domains.yaml` defines the training animals, welfare values, and decision contexts. The generator uses the Cartesian product of these dimensions and distributes examples uniformly across domain cells before deterministically shuffling their IDs. With the default 5 × 4 × 5 domain design and 5,000 examples, each of the 100 cells receives exactly 50 examples. Each feature dataset inherits the exact domain metadata and canonical text from the authoritative corpus.

## Configuration

`config/configs.yaml` contains model IDs, temperatures, concurrency, batch size, seeds, retry limits, dataset size, API settings, versions, and paths. `config/domains.yaml` defines the animal, welfare-value, and context dimensions used for training. `config/held_out_domains.yaml` is a reference-only specification for future generalization evals and is never read by the training-data generator.

Each `features/<variable_name>.yaml` contains `name`, `description`, `language`, `manipulation_level`, `languages`, `definition`, `transformation`, `semantic_constraints`, `examples`, and `cross_lingual_notes`. The feature filename stem and `name` must match. `manipulation_level` is one of `within_language`, `cross_linguistic`, or `covariate`. The current English LoRA generation pipeline automatically runs only `within_language` features; cross-linguistic features are retained for a separate translation-based pipeline, and covariates are retained for measurement/control rather than direct generation.

OpenRouter calls use JSON-schema structured outputs. API credentials are read from `.env`. The default API roles are pinned to explicit upstream providers with provider fallback disabled:

- generator: `google/gemini-3.8-flash` through Google AI Studio;
- judge: `openai/gpt-5.6-sol` through OpenAI;
- reviewer: `anthropic/claude-sonnet-5.5` through Anthropic.

Reasoning effort is configured explicitly per role in `config/configs.yaml`.

## Domain design

The canonical training corpus is defined over three semantic dimensions: animal group, welfare value, and decision context. `config/domains.yaml` contains only domains used for training-data generation. `config/held_out_domains.yaml` records disjoint domains reserved for downstream generalization evaluation.

This separation supports evaluation along several axes: unseen animals with familiar values and contexts, familiar animals with unseen values, familiar animals and values in unseen contexts, and combinations in which multiple semantic dimensions are held out. The held-out file is documentation for the evaluation design and is never consumed by the synthetic training-data generator.

## Pipeline

1. `01_generate.py` builds a balanced shuffled animal × value × context assignment, generates/resumes `data/canonical/generated.jsonl`, invokes the canonical review stage, and then generates raw feature datasets from the reviewed canonical corpus.
2. `review_canonical.py` reviews every canonical item with the judge model. Suitable items are preserved; materially flawed items are repaired while preserving the assigned animal, welfare value, decision context, subject matter, semantic content, factual claims, quantities, entities, stance, and value content. The reviewed corpus is written to `data/canonical/corpus.jsonl`.
3. `02_deterministic_review.py` performs JSON/schema, ID, animal/value/context, canonical-reference, and basic pair validation.
4. `03_deterministic_fix.py` uses the judge model to repair or regenerate flagged feature variants and writes `data/first_review/`.
5. `04_deterministic_recheck.py` reruns deterministic validation on the repaired datasets.
6. `05_semantic_review.py` reviews all pairs for semantic, feature-isolation, and other non-deterministic issues while treating each canonical text as the authoritative reference that every variant must continue to follow.
7. `06_semantic_fix.py` uses the judge model to repair flagged feature variants, keeps repairs tied to the exact canonical example, performs final deterministic validation, and writes `data/final/`.

Every run records a manifest in `data/manifests/` with configuration and prompt versions, model settings, hashes, timestamps, and run statistics.

## Setup and execution

Create a Python environment, install `requirements.txt`, copy `.env.example` to `.env`, and configure the domain and feature YAML files. The default OpenRouter model/provider assignments are already specified in `config/configs.yaml`. Then run:

```bash
python scripts/01_generate.py
python scripts/02_deterministic_review.py
python scripts/03_deterministic_fix.py
python scripts/04_deterministic_recheck.py
python scripts/05_semantic_review.py
python scripts/06_semantic_fix.py
```

`01_generate.py` and the feature-processing scripts accept repeated `--feature <variable_name>` arguments where applicable. Explicit requests for `cross_linguistic` or `covariate` features fail with a clear configuration error rather than accidentally treating them as English paraphrase interventions. `review_canonical.py` can also be run independently to review the generated canonical artifact. Feature dataset files are named `<variable_name>.jsonl` at every feature-data stage. The reviewed source corpus is `data/canonical/corpus.jsonl`; downstream feature generation, validation, review, and repair use it as the source of truth. The held-out-domain file is not consumed by any generation script.

## Design reference

### Dataset JSONL schema

Each line in a feature dataset is a JSON object with the following fields:

```json
{
  "id": "<integer>",
  "feature": "<feature_name>",
  "language": "<language_code>",
  "animal": "<animal_domain>",
  "value": "<welfare_value>",
  "context": "<decision_context>",
  "canonical": "<canonical_text>",
  "feature_variant": "<feature_enhanced_text>"
}
```

### Directory layout

```text
config/
├── configs.yaml
├── domains.yaml
└── held_out_domains.yaml
features/
prompts/
scripts/

data/
├── canonical/
│   ├── generated.jsonl
│   └── corpus.jsonl
├── raw/
├── first_review/
├── final/
├── manifests/
└── issues/
    ├── deterministic/
    └── non-deterministic/
```

### Configuration

The central configuration controls model selection, sampling temperatures, concurrency, batching, seeds, retry behavior, dataset size, API limits, paths, and schema versioning.

```yaml
models:
  generator: "google/gemini-3.8-flash"
  judge: "openai/gpt-5.6-sol"
  reviewer: "anthropic/claude-sonnet-5.5"

providers:
  generator:
    only: ["google-ai-studio"]
    allow_fallbacks: false
  judge:
    only: ["openai"]
    allow_fallbacks: false
  reviewer:
    only: ["anthropic"]
    allow_fallbacks: false

reasoning_effort:
  generator: "low"
  judge: "medium"
  reviewer: "medium"

temperatures:
  generator: <float>
  judge: <float>
  reviewer: <float>

concurrency: <integer>
batch_size: <integer>

seeds:
  generator: <integer>
  judge: <integer>
  reviewer: <integer>

dataset_size: <integer>
canonical_language: "en"

retry_limits:
  max_attempts: <integer>
  backoff_seconds: <number>
  max_backoff_seconds: <number>

api:
  base_url: "https://openrouter.ai/api/v1"
  timeout_seconds: <integer>
  max_output_tokens: <integer>
  require_parameters: <boolean>

paths:
  domains: "config/domains.yaml"
  held_out_domains: "config/held_out_domains.yaml"
  features: "features"
  prompts: "prompts"
  canonical: "data/canonical"
  raw: "data/raw"
  first_review: "data/first_review"
  final: "data/final"
  deterministic_issues: "data/issues/deterministic"
  nondeterministic_issues: "data/issues/non-deterministic"
  manifests: "data/manifests"

schema_version: "1.4"
```

### Issue JSONL schema

Issue files use one JSON object per line:

```json
{
  "id": "<integer_or_null>",
  "feature": "<feature_name>",
  "stage": "<pipeline_stage>",
  "issue_type": "<issue_type>",
  "field": "<field_name_or_null>",
  "message": "<concise_issue_description>"
}
```

### Prompt files

```text
prompts/
├── generate_canonical.txt
├── canonical_review.txt
├── generate.txt
├── deterministic_fix.txt
├── semantic_review.txt
└── semantic_fix.txt
```

### Feature YAML template

Each linguistic variable is defined in its own YAML file:

```yaml
name: <feature_name>
description: >
  <description of the linguistic feature>

language: <language_code>
manipulation_level: <within_language|cross_linguistic|covariate>
languages:
  - <ISO_639-1_code>

definition:
  canonical: >
    <definition of the canonical realization>
  feature_variant: >
    <definition of the feature-enhanced realization>

transformation:
  instructions:
    - <transformation instruction>
    - <transformation instruction>

  preferred_patterns:
    - <preferred realization pattern>
    - <preferred realization pattern>

  avoid_patterns:
    - <confound or transformation to avoid>
    - <confound or transformation to avoid>

semantic_constraints:
  preserve:
    - entities
    - quantities
    - stance
    - factual_claims
    - value_content

examples:
  - canonical: "<canonical example>"
    feature_variant: "<feature-enhanced example>"

cross_lingual_notes: >
  <cross-linguistic motivation or notes>
```

