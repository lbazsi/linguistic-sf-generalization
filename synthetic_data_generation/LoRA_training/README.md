# Synthetic data generation for LoRA training

This directory builds paired synthetic JSONL datasets for controlled linguistic-feature LoRA experiments. It generates one shared canonical corpus, reviews that corpus for non-deterministic quality problems, and then derives every feature dataset by transforming the reviewed canonical texts.

## Data contract

The initial canonical generation is stored in `data/canonical/generated.jsonl`. The reviewed authoritative corpus is stored in `data/canonical/corpus.jsonl` and contains `id`, `language`, `topic`, and `canonical`. Topic allocation is determined during canonical generation: it is deterministic and as even as possible across `dataset_size`, with any remainder assigned to the earliest topics in `config/topics.yaml`. Each feature dataset contains `id`, `feature`, `language`, `topic`, `canonical`, and `feature_variant`, inheriting the ID, language, topic, and reviewed canonical text exactly from the authoritative corpus.

## Configuration

`config/configs.yaml` contains model IDs, temperatures, concurrency, batch size, seeds, retry limits, dataset size, API settings, versions, and paths. `config/topics.yaml` is a YAML list of topic names.

Each `features/<variable_name>.yaml` contains `name`, `description`, `language`, `definition`, `transformation`, `semantic_constraints`, and `examples`. The feature filename stem and `name` must match. The transformation contains `instructions`, `preferred_patterns`, and `avoid_patterns`; semantic constraints contain a `preserve` list; definitions and examples each contain `canonical` and `feature_variant`.

OpenRouter calls use JSON-schema structured outputs. API credentials are read from `.env`.

## Pipeline

1. `01_generate.py` generates/resumes `data/canonical/generated.jsonl`, invokes the canonical review stage, and then generates raw feature datasets from the reviewed canonical corpus.
2. `review_canonical.py` reviews every canonical item with the judge model. Suitable items are preserved; materially flawed items are repaired while preserving their subject matter, semantic content, factual claims, quantities, entities, stance, and value content. The reviewed corpus is written to `data/canonical/corpus.jsonl`.
3. `02_deterministic_review.py` performs JSON/schema, ID, topic, canonical-reference, and basic pair validation.
4. `03_deterministic_fix.py` uses the judge model to repair or regenerate flagged feature variants and writes `data/first_review/`.
5. `04_deterministic_recheck.py` reruns deterministic validation on the repaired datasets.
6. `05_semantic_review.py` reviews all pairs for semantic, feature-isolation, and other non-deterministic issues while treating each canonical text as the authoritative reference that every variant must continue to follow.
7. `06_semantic_fix.py` uses the judge model to repair flagged feature variants, keeps repairs tied to the exact canonical example, performs final deterministic validation, and writes `data/final/`.

Every run records a manifest in `data/manifests/` with configuration and prompt versions, model settings, hashes, timestamps, and run statistics.

## Setup and execution

Create a Python environment, install `requirements.txt`, copy `.env.example` to `.env`, and populate model IDs, topics, and feature YAML files. Then run:

```bash
python scripts/01_generate.py
python scripts/02_deterministic_review.py
python scripts/03_deterministic_fix.py
python scripts/04_deterministic_recheck.py
python scripts/05_semantic_review.py
python scripts/06_semantic_fix.py
```

`01_generate.py` and the feature-processing scripts accept repeated `--feature <variable_name>` arguments where applicable. `review_canonical.py` can also be run independently to review the generated canonical artifact. Feature dataset files are named `<variable_name>.jsonl` at every feature-data stage. The reviewed source corpus is `data/canonical/corpus.jsonl`; downstream feature generation, validation, review, and repair use it as the source of truth.

## Design reference

### Dataset JSONL schema

Each line in a feature dataset is a JSON object with the following fields:

```json
{
  "id": "<integer>",
  "feature": "<feature_name>",
  "language": "<language_code>",
  "topic": "<topic_name>",
  "canonical": "<canonical_text>",
  "feature_variant": "<feature_enhanced_text>"
}
```

### Directory layout

```text
config/
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
  generator: <openrouter_model>
  judge: <openrouter_model>
  reviewer: <openrouter_model>

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
  topics: "config/topics.yaml"
  features: "features"
  prompts: "prompts"
  canonical: "data/canonical"
  raw: "data/raw"
  first_review: "data/first_review"
  final: "data/final"
  deterministic_issues: "data/issues/deterministic"
  nondeterministic_issues: "data/issues/non-deterministic"
  manifests: "data/manifests"

schema_version: "1.2"
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
```

