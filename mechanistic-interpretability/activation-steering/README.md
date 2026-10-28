# Activation Steering Along Linguistic Directions

## Question

Can strengthening or reversing internal directions associated with linguistic structure alter expressed linguistic properties and animal-welfare value generalization? Distinguish direct manipulation of surface linguistic form from reproduction of a behavioral effect induced by linguistically different training data.

## Features and source data

Run the primary protocol for **all 11 English linguistic interventions**. Derive directions independently for the three cross-language interventions and report these separately, as their contrasts also encode a change in language identity. For each intervention, use the reviewed matched `canonical`/`feature_variant` texts (for Mandarin-source features, use the actual Mandarin control text). Both sides of a pair belong to one semantic ID and one data split. Estimate directions using the corresponding **pretrained base model's** activations before any animal-welfare LoRA adaptation; this reduces entanglement with fine-tuning. Do not extract a direction on the evaluation test set.

## Contrastive direction estimation

For each layer L, injection site S (residual stream after block or MLP output before residual addition), and pooling P (final non-padding prompt token or masked prompt-token mean), compute a paired mean difference:

```text
d[L,S,P,feature] = mean_over_training_IDs(
    h(L,S,P,feature_variant) - h(L,S,P,canonical_control)
)
unit_d = d / ||d||_2
```

Verify the direction norm is nontrivial relative to a shuffled-label/pairing null. Determine sign from the fixed variant−control definition; do not flip direction to maximize evaluation effects. Report the top-k lexical or functional associations only as descriptive checks.

## Activation normalization and intervention

Calibrate each site on a separate fixed, non-welfare development corpus, using its baseline activation distribution. Define `s[L,S] = mean(||h||_2 / sqrt(D))`, with D the activation width; specify the pooling/token context used to estimate s and keep it fixed across feature directions at that site. Inject

```text
h_steered = h + alpha * s[L,S] * unit_d
alpha = [-2, -1, 0, 1, 2]
```

Use the same convention for MLP-output and residual-stream sites, calibrating their scales separately. Log realized perturbation norm relative to baseline RMS and flag saturation/degradation at extreme alpha. Keep all settings including sign and strength identical within each model family; do not interpret equal alpha as an identical absolute intervention across architectures.

**Token policy:** first inject at the final non-padding prompt token in the prompt-processing forward pass. In a separately reported condition, inject at every autoregressive generation step at the current position; do not apply a prompt-token intervention indiscriminately across all prompt tokens. Make cached generation/past-key-value hooks explicit so neither the intervention nor its application count silently changes between models.

## Layer/site screening

Screen **all layers and both activation sites**, for every feature and both its registered control and feature adapter. Use the same 200 existing benchmark scenarios with paired welfare-supporting/opposing options: 20 initial screening scenarios, 80 independent validation scenarios and 100 held-out final-evaluation scenarios. Compute the length-normalized log-probability difference between the options, and compare each steered score to its same-scenario zero-strength baseline. Retain all screening results. On validation, select one nonzero layer/site/strength combination with the largest absolute mean proxy change **for every feature–adapter pair**, including null features; break ties deterministically. The selection does not restrict or remove full-layer screening results. The final held-out judged evaluation is a separate explicit command, using the original scoring and batching system. Report prompt-last-only and generation-step steering conditions separately. Configuration selection currently uses the proxy outcome and requires the manipulation/coherence diagnostics to be interpreted separately.

## Evaluation and controls

Use the existing scenario prompt, continuation, two-judge rubric, aggregation, and statistics pipeline for **welfare outcome scoring**, including overall `target_value_support`, generalization categories, tradeoff effects and coherence. On a separate fixed sample, additionally assess **linguistic manipulation success** with feature-appropriate diagnostics; this is a manipulation check, not a substitute for welfare evaluation. For cross-language features, measure whether the intervention changes output language, since language switching would confound welfare results.

For each feature/layer/site in screening, include alpha=0 and both positive and negative alpha. Norm-matched random directions and randomly swapped-label directions are planned negative-control diagnostics that must be implemented before drawing strong steering-specific conclusions. Include an unsteered canonical adapter and the matching feature adapter: does steering canonical toward the feature direction shift behavior toward the feature adapter's observed result? Also run steering on the pretrained base model to separate adapter-specific effects. Random-direction controls should be repeated as computational budget allows; one sampled random vector is a weak null distribution. Track refusal/invalid responses and incoherence. Test steering on the remaining welfare scenarios only **after** layer/strength selection, with scenario-ID and semantic-cell separation from development.

## Statistics and outputs

Report signed paired change in target-value support versus unsteered, interval estimates from semantic-cell cluster bootstrap, and FDR-corrected p-values within predeclared steering outcome families. Plot alpha dose-response for linguistic-feature realization, welfare score, coherence, and language switching where relevant; show all tested features including nulls. Retain compact direction vectors for replay if acceptable (these are **not** raw token activations), or direction-generation seeds and training IDs if even vector artifacts are excluded. Persist sweep configuration, layer selection criteria, calibration scales, scalar scores and generated continuations in the existing evaluation result format. Do not save token activations or additional model checkpoints.

## Interpretation

A contrastive activation direction need not correspond to a single linguistic concept. Steering can influence unrelated factors, and large alpha may cause off-distribution behavior. An observed welfare shift is evidence of intervention sensitivity, not proof that the fine-tuning-induced welfare difference was caused by the same direction. Null welfare effects despite successful linguistic manipulation are informative.


## Implementation and separation of screening from judging

Contrastive directions for all layers/sites are estimated together from the model's base checkpoint using matched semantic-ID training pairs. The only cached objects are small unit direction vectors, shuffled-label control directions, random unit control directions and per-site activation RMS calibration scales. Raw prompt/token activations are not saved. The screen uses the existing 200-scenario matched-option dataset: 20 `screen`, 80 `validation`, and 100 untouched `test` examples. Every feature is screened on its own feature adapter and its registered control, with residual/MLP intervention sites, every layer, both `prompt_last` and `generation` policies, and alpha in `[-2,-1,0,1,2]`. Scoring uses teacher-forced, length-normalized log-probability differences, without generating judge-bound responses.

The selector retains full-layer screening outputs and chooses **one** layer/site/strength/token-policy combination per feature–adapter pair from the 80 validation scenarios. Selected configurations are checked against norm-matched random and shuffled-label direction controls. The full judged evaluation generates 100 held-out continuations per selection and uses the two original blind judges in batches of ten. Preview the number of continuations, judgments and requests with `python run.py judge-plan --model gemma2`. The separate `judge-run --execute-judges` command is the only stage that contacts the remote judge APIs; the program cannot silently advance from screening to judging. Original unsteered condition scores are used for paired evaluation differences.

## Optimized screening implementation

The optional `steering-screen-fast` entry point preserves the existing teacher-forced supporting-minus-opposing log-probability scoring, full layer/site/strength grid, and `screen_<feature>_<adapter>_<policy>.jsonl` output schema. `run_all.py --stage steering --execute --resume` now uses this implementation. The original `steering-screen` command remains available as the reference scorer.

The optimized path pre-tokenizes the 100 screen/validation scenarios and both continuations once per invocation, uses `steering.screening.batch_size` scenarios per forward batch (two candidates per scenario), and caches the unsteered zero-strength score once per adapter, shared across features, layers, sites and policies. Each completed layer/site grid is written atomically under `activation-steering/screen_checkpoints/`; incomplete layers are recomputed on restart. Compatible cached scores are stored under `activation-steering/screen_cache/`. Manifests record model/adapter weight identity, data, attention/dtype, batch size, direction artifact and calibration fingerprints. Incompatible resumes are rejected instead of silently mixing screening methods.

Begin with batch size 8, then evaluate 16 only after confirming memory headroom and agreement with the reference scorer. This is independent of `steering.generation.batch_size`. GPU memory consumption can be higher than batched text generation because scoring materializes token-level vocabulary logits. For a clean numerical validation on an actual GPU, compare optimized and reference scores using the same batch size for selected scenarios, sites, strengths and both policies, then check the selected configurations. Changing batch size or data identity requires archiving any existing fast-screening checkpoints and relevant final files first.

```bash
python -m pytest tests/test_screening_fast.py -q
python run.py steering-screen-fast --model qwen3 --feature conditionality --condition canonical --policy prompt_last
python run.py steering-screen-fast --model qwen3 --feature conditionality --condition canonical --policy generation
python run_all.py --model qwen3 --stage steering --execute --resume
```

The optimization does not change Llama's already completed screening artifacts and never invokes paid judge calls.
