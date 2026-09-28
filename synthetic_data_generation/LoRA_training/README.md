# Synthetic data generation for LoRA training

This directory builds paired synthetic JSONL datasets for controlled linguistic-feature LoRA experiments. Each dataset contains a canonical text and a feature-enhanced version with the same semantic and value content.

## Data contract

Each JSONL row contains `id`, `feature`, `language`, `topic`, `canonical`, and `feature_variant`. IDs are integers in `1..dataset_size`. The same numeric ID range is used independently for every feature dataset. Topic allocation is deterministic and as even as possible across the configured dataset size; any remainder is assigned to the earliest topics in `config/topics.yaml`.

## Configuration

`config/configs.yaml` contains model IDs, temperatures, concurrency, batch size, seeds, retry limits, dataset size, API settings, versions, and paths. `config/topics.yaml` is a YAML list of topic names.

Each `features/<variable_name>.yaml` contains `name`, `description`, `language`, `definition`, `transformation`, `semantic_constraints`, and `examples`. The feature filename stem and `name` must match. The transformation contains `instructions`, `preferred_patterns`, and `avoid_patterns`; semantic constraints contain a `preserve` list; definitions and examples each contain `canonical` and `feature_variant`.

OpenRouter calls use JSON-schema structured outputs. API credentials are read from `.env`.

## Pipeline

1. `01_generate.py` generates resumable raw paired datasets.
2. `02_deterministic_review.py` performs JSON/schema, ID, topic, and basic pair validation.
3. `03_deterministic_fix.py` uses the judge model to repair or regenerate flagged IDs and writes `data/first_review/`.
4. `04_deterministic_recheck.py` reruns deterministic validation on the repaired datasets.
5. `05_semantic_review.py` reviews all pairs for semantic, feature-isolation, and other non-deterministic issues while carrying forward deterministic residue.
6. `06_semantic_fix.py` uses the judge model to repair flagged pairs, performs a final deterministic validation, and writes `data/final/`.

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

Each script accepts repeated `--feature <variable_name>` arguments. Dataset files are named `<variable_name>.jsonl` at every data stage.
