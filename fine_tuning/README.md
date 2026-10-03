# Fine-tuning

This directory trains one canonical LoRA adapter and one LoRA adapter for each linguistic-feature dataset produced by the synthetic data pipeline.

All runs start from the same pretrained base-model family and use the same training configuration and seed. The canonical adapter is trained on the shared canonical corpus. Each feature adapter is trained on the corresponding transformed text from `data/final/<feature>.jsonl`.

## Input data contract

The fine-tuning code consumes the synthetic-data directory directly. No conversion step or duplicated training-data directory is required.

Expected input structure:

```text
data/
├── canonical/
│   └── corpus.jsonl
└── final/
    ├── <feature_a>.jsonl
    ├── <feature_b>.jsonl
    └── ...
```

The canonical corpus must contain:

```json
{
  "id": 1,
  "language": "en",
  "topic": "<topic>",
  "canonical": "<text>"
}
```

Each final feature dataset must contain the same IDs and must preserve the canonical `language`, `topic`, and `canonical` fields exactly:

```json
{
  "id": 1,
  "feature": "<feature_name>",
  "language": "en",
  "topic": "<topic>",
  "canonical": "<same canonical text>",
  "feature_variant": "<transformed text>"
}
```

Canonical training uses the `canonical` field. Feature training uses only the `feature_variant` field.

## Model and objective

The default base model is `google/gemma-2-9b`.

The model is loaded in bfloat16 and fine-tuned using LoRA rather than QLoRA. This avoids introducing quantization as an additional intervention while keeping GPU memory use far below full-parameter fine-tuning.

Training uses the ordinary causal-language-model objective over each complete text. No chat template, instruction wrapper, preference objective, or contrastive objective is added. The training intervention is therefore the text distribution itself rather than an additional supervised task format.

The configured Hugging Face revision is resolved to an exact repository commit SHA before the model is loaded. That resolved revision is written to each training summary.

## LoRA configuration

The default LoRA configuration is:

```yaml
r: 16
alpha: 32
dropout: 0.0
bias: "none"
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

LoRA is applied to both attention projections and MLP projections. The configuration is held constant across canonical and feature conditions so that the linguistic dataset is the intended changing factor.

## Validation split

Exactly 50 canonical IDs are held out by default.

The validation IDs are chosen once from the canonical corpus using the configured training seed and are stratified by topic. The same IDs are then held out from the canonical run and every feature run.

Topic stratification avoids a tail split accidentally over-representing the final topic when the canonical corpus was generated in deterministic topic blocks.

Validation data is used only as a diagnostic held-out set. It does not select checkpoints and there is no early stopping.

## Sequence handling

Sequence packing is disabled.

Without packing, one dataset row is tokenized as one model sequence. With packing, several short rows can be concatenated into the same context window to reduce padding. Packing is efficient, but the packing pattern depends on sequence lengths. Because linguistic transformations may themselves change text length, packing could make canonical and feature conditions differ in which otherwise unrelated examples share a context window. Disabling it removes that extra source of variation.

`sequence.max_length` is `null` by default because the final token-length distribution is not known before data generation. When it is null, the trainer tokenizes the selected dataset, finds the configured empirical quantile, and chooses the smallest configured candidate length that covers that quantile.

Any examples longer than the selected maximum are truncated. Truncation never fails the run: it emits a warning and the counts are recorded in `training_summary.json`.

## Training configuration

The default training schedule is:

```yaml
seed: 42
num_train_epochs: 3
learning_rate: 0.0001
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
max_grad_norm: 1.0
gradient_checkpointing: true
```

With one GPU, the configured micro-batch size and gradient accumulation give an effective batch size of 32 examples per optimizer update.

The same seed controls model-training randomness and data shuffling for all conditions.

## Checkpoints and outputs

The base model is never copied into the output tree. Each run stores only LoRA adapters and a compact JSON summary.

```text
outputs/
├── canonical/
│   └── seed_42/
│       ├── midpoint_adapter/
│       ├── adapter/
│       └── training_summary.json
└── features/
    ├── <feature_name>_01/
    │   └── seed_42/
    │       ├── midpoint_adapter/
    │       ├── adapter/
    │       └── training_summary.json
    ├── <feature_name>_02/
    │   └── seed_42/
    └── ...
```

Feature numbers are assigned deterministically from the alphabetically sorted filenames in `data/final/`.

The midpoint adapter is saved at approximately 50% of optimizer updates. It provides one intermediate point for later mechanistic comparison without saving full Trainer checkpoints throughout training. The final adapter is stored in `adapter/`.

`training_summary.json` records the information needed to identify and inspect a run without retaining optimizer state or full-model checkpoints:

- condition, feature, feature index, and seed;
- exact resolved base-model revision;
- input file hashes and validation IDs;
- train and validation example counts;
- token-length statistics and truncation counts;
- selected maximum sequence length;
- LoRA configuration and parameter counts;
- core training hyperparameters;
- final training loss and validation loss;
- optimizer-step count;
- midpoint step and artifact locations;
- basic structural verification results;
- minimal runtime information.

## Verification

Input validation is strict for structural problems. Training stops if IDs are duplicated or missing, if a feature dataset has different IDs from the canonical corpus, or if inherited canonical metadata differs.

Truncation is intentionally not a structural failure and therefore produces only a warning.

`scripts/verify.py` can check input data before training. If a training output already exists, it also checks that the expected adapter directories and summary are present.

## Setup

Install dependencies:

```bash
cd fine_tuning
pip install -r requirements.txt
```

The Gemma model is gated on Hugging Face, so the GPU environment must have access to the model under the account/token used by `huggingface_hub`.

By default, `config/training.yaml` points to the repository's synthetic-data output:

```text
../synthetic_data_generation/LoRA_training/data
```

If the same `data/` directory is copied elsewhere on the GPU, pass its path with `--data-root`.

## Commands

Verify the canonical input:

```bash
python scripts/verify.py --canonical
```

Verify one feature:

```bash
python scripts/verify.py --feature assertive_language
```

Train the canonical model:

```bash
python scripts/train.py --canonical
```

Train one feature model:

```bash
python scripts/train.py --feature assertive_language
```

Train canonical followed by every feature dataset:

```bash
python scripts/train.py --all
```

Use a copied data directory:

```bash
python scripts/train.py --canonical --data-root /path/to/data
```

An existing run directory is not overwritten by default. Pass `--overwrite` only when replacing that exact condition/seed output is intentional.

## Directory layout

```text
fine_tuning/
├── README.md
├── requirements.txt
├── config/
│   └── training.yaml
├── scripts/
│   ├── common.py
│   ├── train.py
│   └── verify.py
└── outputs/
    ├── canonical/
    │   └── .gitkeep
    └── features/
        └── .gitkeep
```
