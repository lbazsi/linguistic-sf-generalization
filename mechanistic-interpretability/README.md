# Mechanistic Interpretability

## Research question

How does holding animal-welfare training content approximately constant while changing its linguistic structure alter internal representations, LoRA updates, and expressed welfare-related decisions? We compare descriptive representation measurements with exploratory causal interventions, without equating decodability, association, or steering sensitivity with a unique mechanism.

## Models and comparisons

| Family | Pinned pretrained base | Size |
| --- | --- | --- |
| Gemma 2 | `google/gemma-2-9b` | 9B |
| Qwen 3 | `Qwen/Qwen3-8B-Base` | 8B |
| Llama 3.1 | `meta-llama/Llama-3.1-8B` | 8B |
| Mistral | `mistralai/Mistral-7B-v0.3` | 7B |

Immutable checkpoint revisions are specified in [the pipeline configuration](config/pipeline.yaml). Within each family, independently trained LoRA adapters comprise English canonical, Mandarin control and 14 linguistic-feature conditions, with the unadapted base as reference. English interventions and English→Japanese constituent order use English canonical as their control; Mandarin→Korean inflectional synthesis and Mandarin→Latin fusion use Mandarin control. Cross-language differences confound typology with language identity. Lexical diversity is measured, not separately fine-tuned.

Each adapter is trained from the same family-specific base with LoRA rank 16, alpha 32, zero dropout, a three-epoch schedule and seed 42, subject to architecture-compatible projection mappings. Midpoint and final adapter checkpoints are separate observations from the same training trajectory. Only one seed is used per condition.

## Datasets and representation protocol

The study uses reviewed paired linguistic training data, the established 800-scenario behavioral evaluation with eight generalization categories, and [three supporting datasets](data/README.md): 1,000 general-domain prompts, 1,000 welfare-concept probe records and 200 steering scenarios divided into screen (20), validation (80) and test (100). Scenario and semantic-ID partitions are held fixed across within-model conditions.

Representations are measured at every transformer layer, at residual-stream and MLP-output sites, using final non-padding token and masked mean pooling. Base and adapter models are loaded sequentially. Inference uses BF16 where supported; statistical calculations use adequate higher precision. Token activations and pooled matrices remain transient rather than being committed to the repository.

## Analyses

**[Model differences](Model-difference-analysis/README.md).** On identical welfare and general-domain prompts, measure layerwise linear CKA, descriptive SVCCA and normalized representation displacement between each feature adapter and its registered control. Cross-family comparison uses layer-normalized profiles and summary statistics rather than raw coordinate alignment.

**[Linear probing](linear-probing/README.md).** Fit separate linear classifiers for linguistic contrast and welfare-concept presence, and linear regressors for model-specific judged welfare support. Grouped 70/15/15 train/development/test splits prevent paired-item leakage. Behavioral regression uses `dual_ridge_v2`, regularization chosen only on development data, a constant-target baseline and separate held-out predictions. Lexical and length controls probe shortcut dependence. Classification accuracy establishes decodability, not causal utilization.

**[LoRA analysis](LoRA-analysis/README.md).** Compare effective low-rank parameter updates and module/subspace similarities across controls, features and training checkpoints. The functional analysis measures on-input update effects on a fixed stratified sample: 160 welfare prompts (category × tradeoff) and 200 general prompts (topic strata), separately for midpoint and final adapters. Report global RMS with element-count normalization, not an average of batchwise RMS values.

**[Activation steering](activation-steering/README.md).** Extract paired feature-minus-control directions from base-model representations, calibrate direction norms at each site and screen 14 features × two adapters × two token policies = 56 configurations across all layers/sites and signed doses. The active strength convention is `relative_activation_norm_v2`. A validation-only proxy selects 28 feature–adapter configurations, one each; random and label-shuffled directions provide diagnostic controls. The selected configurations undergo matched steered/unsteered held-out generation on 100 unseen scenarios, followed by two-judge scoring and paired statistical analysis with bootstrap intervals and FDR correction. A significant likelihood-proxy shift does not establish successful linguistic manipulation.

## Quality control and interpretation

Keep direction-fitting training IDs, steering validation and final test scenarios separate. Probe development selection must not use test labels; subsequent held-out audits are conditional on the selected estimator and do not measure training-seed uncertainty. Steering pilots on development texts are exploratory; linguistic test pairs exposed by earlier pilots cannot be treated retrospectively as untouched confirmation. Judge item ordering, arm matching, response validity, coherence and language switching require explicit checks. Automated generation coverage/truncation gates are necessary but not sufficient: inspect stratified matched continuations, particularly cross-language conditions, before interpreting behavioral differences.

The screening proxy, judged welfare support and linguistic feature realization are distinct outcomes. Steered-versus-unsteered differences demonstrate intervention sensitivity, not proof that a linguistic direction mediates a fine-tuning effect. Negative outcomes remain part of the registered analysis. Comparisons across families remain descriptive because architectural, tokenizer and pretraining differences are not experimentally controlled.

## Execution and results

The pipeline is configured in [`config/pipeline.yaml`](config/pipeline.yaml). `run_all.py` orchestrates local difference, probe, functional LoRA and steering-screen tasks; `run.py` exposes individual analyses, screening selection and controls. `run_judged_remainder.py` runs the matched 512-token maximum, 48-token minimum generation protocol (generation batch 32), checks coverage and truncation and, only with explicit `--execute-judges`, invokes batched remote judging (batch 20), aggregation and plots. Base-model access, adapter archives, configuration and API credentials are external prerequisites; the pipeline does not create them.

Temporary GPU outputs are kept in `outputs/<model>/`; accepted compact results are organized under `../results/mechanistic-interpretability/final/<model>/`, with stage directories `model-difference-analysis/`, `linear-probing/`, `LoRA-analysis/`, `activation-steering/`, and `plots/`. Earlier analysis versions are retained separately and must not be combined with corrected estimators or strength conventions. Models' raw direction tensors, adapter weights and transitory representations are not publication artifacts in Git. The independent [cross-model analysis](../cross-model-analysis/README.md) examines replication and heterogeneity.
