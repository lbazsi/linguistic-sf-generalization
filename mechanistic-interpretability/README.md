# Mechanistic Interpretability of Linguistic Fine-Tuning Effects

## Research objective

This study characterizes how semantically matched linguistic variants of animal-welfare fine-tuning data affect model representations, learned LoRA updates, and downstream animal-welfare value generalization. The analyses are observational except for exploratory activation steering. Results are compared across four pretrained model families to evaluate replication.

The four stages are (1) model-difference analysis, (2) linear probing, (3) LoRA analysis with midpoint-to-final comparison, and (4) linguistic-direction activation steering. The stage-specific methods are in [Model-difference-analysis](Model-difference-analysis/README.md), [linear-probing](linear-probing/README.md), [LoRA-analysis](LoRA-analysis/README.md), and [activation-steering](activation-steering/README.md). Cross-model replication and synthesis are specified separately in [cross-model-analysis](../cross-model-analysis/README.md).

## Models and conditions

| Family | Pretrained base checkpoint | Approximate parameters |
| --- | --- | --- |
| Gemma 2 | `google/gemma-2-9b` | 9B |
| Qwen 3 | `Qwen/Qwen3-8B-Base` | 8B |
| Llama 3.1 | `meta-llama/Llama-3.1-8B` | 8B |
| Mistral | `mistralai/Mistral-7B-v0.3` | 7B |

Resolve and record an immutable checkpoint revision before training. For Qwen, explicitly verify availability and base (rather than instruct) status; do not substitute a reasoning/instruct checkpoint. For Mistral, use the v0.3 *base* model, not an Instruct checkpoint. Record access/licensing and architecture compatibility.

For every family, evaluate the pretrained base model, English canonical adapter, Mandarin control adapter, and 14 feature adapters. The 11 English-to-English features are compared with English canonical. `constituent_order` (English→Japanese) uses English canonical as control; `inflectional_synthesis` (Mandarin→Korean) and `fusion` (Mandarin→Latin) use Mandarin control. Analyze cross-language contrasts separately because language identity and typology are confounded. `lexical_diversity` is a measured covariate, not a distinct adapter.

## Shared training and evaluation contracts

Reuse the reviewed paired training corpus and the existing 800-scenario evaluation benchmark. Train independent adapters from each model's identical base checkpoint: LoRA r=16, alpha=32, dropout=0, bias=none; target `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` when architecturally supported. Three epochs, learning rate 1e-4, cosine scheduler, warmup ratio 0.05, effective batch size 32, weight decay zero, seed 42; preserve semantic-ID validation splits and record unavoidable architecture-specific changes. Verify target coverage and trainable parameter counts before each run. Train and save midpoint and final adapters; the midpoint corresponds to half the scheduled optimizer steps, not a second independent run. Replication uses one training seed per condition, so variability from training seeds remains unmeasured.

The evaluation contract is the existing pipeline: identical scenario IDs, prompt template, generation settings, two judges, 0–4 `target_value_support` primary outcome, and existing category/tradeoff stratification. The primary behavioral contrast is feature minus its registered control, with canonical minus base as a separate measure. Report judge reliability, invalid responses, uncertainty, and the original multiple-comparison families.

For representation analysis add **1,000 general-domain, non-animal-welfare English prompts**, independently generated in five batches of 200, assigned to predefined broad topical categories. Deduplicate exact and near-duplicate prompts and screen for welfare leakage. Preserve prompt texts, category, source batch, IDs and generation metadata in a small dataset artifact. Do not reuse probe training prompts for held-out steering evaluation. For concept probes, generate or derive balanced positive/negative statements, matched on animal mention, context, length and topic; independently label the presence of welfare concepts rather than using model response scores.

## Common measurement protocol

- Extract representations at **every transformer layer**, including residual stream and MLP block outputs. Define hooks with respect to the forward computation (for example, post-block residual and pre-residual-add MLP output) and record exact module paths per architecture.
- Collect both final *non-padding prompt token* and masked mean over prompt tokens, with a uniform tokenizer/padding policy. Keep tokenization-specific differences visible in the metadata.
- Use the same semantic-ID-disjoint split assignments across matched canonical/feature examples. Use non-overlapping development and test subsets for layer/strength selection and final inference.
- Use BF16 inference where supported, no dropout, `eval()`, no gradient tracking, fixed prompt formatting and generation seeds. Fit numerically sensitive statistics/probes in FP32/FP64.
- Process one layer/batch/condition at a time. **Token activations are GPU-resident, transient and not persisted**: compute and accumulate sufficient statistics or probe batches online, then release buffers. Do not persist raw token activations, full activation matrices, or merged model checkpoints; pooled matrices may exist transiently on the GPU while CKA/SVCCA are calculated. The originally required midpoint/final adapter tensors remain necessary input artifacts, stored outside the repository; this project does not create additional heavyweight checkpoints.
- Persist small artifacts sufficient to audit conclusions: run manifests, data/split hashes, model/adapter identifiers, hook maps, aggregate covariances or similarity summaries where practical, probe coefficients and metrics, small normalized steering directions if needed for rerunning steering (not token activations), scalar evaluation outputs, and plots/tables. Omit any summary that would reveal private prompt text if inputs are sensitive.
- Device memory use depends on layer width, batch size, prompt length and whether modules emit logits or full token matrices; perform an A100 memory pilot rather than assuming unrestricted all-layer retention.

## Sequence and outputs

1. Verify base revisions, adapter archives, exact datasets, condition-to-control mappings and GPU compatibility. Generate and validate the 1,000 general prompts and concept-probe dataset.
2. Run [model-difference analysis](Model-difference-analysis/README.md) for welfare and general prompts; summarize layer/module sensitivity.
3. Run [linear probes](linear-probing/README.md) for linguistic identification, welfare-concept detection and welfare-behavior prediction using a fixed estimator protocol.
4. Run [LoRA analysis](LoRA-analysis/README.md) over all adapters, modules and midpoint/final steps.
5. Run [activation steering](activation-steering/README.md) for all 14 features, both associated adapter conditions, every layer and both activation sites. The complete layerwise screening results are retained; validation selects one configuration per feature–adapter pair for a separately authorized held-out judged evaluation.
6. Export the four families' scalar analysis artifacts for separate comparison in [cross-model-analysis](../cross-model-analysis/README.md).

## Inference and limitations

Report paired scenario-level effects with 95% semantic-cell-cluster bootstrap confidence intervals; use cluster-aware sign-flip/permutation tests where appropriate and Benjamini–Hochberg FDR within predeclared outcome families. Separately show feature-level and held-out-domain effects. With four distinct model families, consistent signs and patterns are strong **cross-model replication evidence within this sample**, not a universal claim; seed-level uncertainty cannot be assessed with one seed. Probes demonstrate decodability, and differences in representation or LoRA parameters are descriptive. Steering supports limited intervention-based evidence but can move activations off-distribution.

## Reproducibility contract

Record resolved Hugging Face base-model revision, adapter hash and checkpoint step, package/CUDA/PyTorch/Transformers/PEFT versions, hardware, random seeds, full evaluation configuration, corpus and prompt hashes, split ID lists, hook definition, pooling rule, regularization choices, normalization calibration, predeclared search space, and scoring revisions. Transient activations and model weights are deliberately excluded from committed artifacts; reproducing results requires regenerating the activations from the original inputs and external base/adapter checkpoints. Documentation here specifies planned analyses; it does not claim they have already been executed.


## Implementation and commands

The Python implementation lives in `mechinterp/`, with a single command entry point, `run.py`, configuration in `config/pipeline.yaml` and smoke tests in `tests/`. Install with `pip install -r requirements.txt` and run `python -m pytest tests -q`. The repository provides method specifications and implementation; the GPU analyses have not been executed by this documentation commit.

On GPU images with an existing CUDA-enabled PyTorch installation, an environment created with `python3 -m venv --system-site-packages .venv` can reuse that PyTorch build. Install the full requirements file **inside** the virtual environment: its compatible NumPy, Pillow, scikit-learn, and Transformers versions must take precedence over older system-level packages. Run `python -m pip check` and the smoke tests before GPU inference. The dependency bounds exclude NumPy 2 for legacy binary compatibility and Transformers 5 pending validation of the model-loading interface. When restoring feature adapters, the loader accepts both their original numbered paths (for example, `features/conditionality_01/seed_42`) and unnumbered paths. It verifies the presence of both `adapter_config.json` and `adapter_model.safetensors`; use the original checksum manifest separately for byte-integrity verification.

```bash
cd mechanistic-interpretability
python run.py validate
python run.py validate --check-screening   # requires steering_screening.jsonl
python run.py difference --model gemma2 --condition epistemic_hedging
python run.py probe --model gemma2 --condition canonical --task concept
python run.py probe --model gemma2 --condition canonical --task linguistic --feature epistemic_hedging
python run.py probe --model gemma2 --condition canonical --task behavior
python run.py lora --model gemma2
python run.py lora-functional --model gemma2
python run.py steering-screen --model gemma2 --feature epistemic_hedging --condition canonical
python run.py steering-screen --model gemma2 --feature epistemic_hedging --condition epistemic_hedging
python run.py steering-select --model gemma2
python run.py judge-plan --model gemma2
python run.py judge-generate --model gemma2
python run.py judge-run --model gemma2 --execute-judges  # explicit API charges
python run.py judge-aggregate --model gemma2
python ../cross-model-analysis/run.py --stage all
```

Screen all registered features and both matching adapter conditions before invoking `steering-select`. Repeat separately with `--policy generation` for generation-step interventions. The judged-evaluation process uses only the held-out 100 scenario IDs, and its batch size follows the original evaluation configuration. `judge-plan` prints/writes workload counts; actual monetary pricing requires observed or independently supplied prompt/completion token estimates and current per-judge prices.

The model configuration expects existing independently trained checkpoint trees, including midpoint/final adapters and their training summaries. Adapter bytes are not stored in this repository. Each additional model must have its revision pinned and adapters generated via the common fine-tuning contract. GPU analyses write locally under `outputs/<model>/<stage>/`; the retained scalar results are transferred to `results/mechanistic-interpretability/<model>/` in this repository. These results are not input checkpoints.

### Verification and current boundaries

Unit smoke tests verify data parsing, grouped splits, key statistics and hook behavior without downloading large models. Full GPU validation must separately verify architecture hooks, PEFT adapter reassembly, CUDA memory use, Qwen/Mistral/Llama compatibility and agreement of generated continuations with the original evaluator. Probe metrics, model differences and LoRA geometry are descriptive; observed intervention effects are not an identification of training-time causation. The current steering screen ranks by welfare log-probability change; linguistic-manipulation and coherence diagnostics and matched random-direction controls require additional validation before strong causal interpretation. No judge stage is called automatically from analysis commands.


## Complete local execution sequence

`run_all.py` lists local experiments by default; `--execute` is required to launch them and `--resume` skips existing completed result files. This orchestration never invokes API judges.

```bash
cd mechanistic-interpretability
pip install -r requirements.txt
python -m pytest tests -q
python run.py validate --check-screening

# Preview each local workload before execution.
python run_all.py --model gemma2 --stage difference
python run_all.py --model gemma2 --stage probes
python run_all.py --model gemma2 --stage lora
python run_all.py --model gemma2 --stage steering

# Run individual stages as GPU capacity permits:
python run_all.py --model gemma2 --stage difference --execute --resume
python run_all.py --model gemma2 --stage probes --execute --resume
python run_all.py --model gemma2 --stage lora --execute --resume
python run_all.py --model gemma2 --stage steering --execute --resume

# One selected configuration per feature-adapter pair, across both token policies.
python run.py steering-select --model gemma2
python run.py steering-controls --model gemma2
python run.py plots --model gemma2

# Deliberately separate expensive judged evaluation.
python run.py judge-plan --model gemma2
python run.py judge-generate --model gemma2
python run.py judge-run --model gemma2 --execute-judges
python run.py judge-aggregate --model gemma2

# Repeat on qwen3, llama31, and mistral after training/evaluation.
python ../cross-model-analysis/run.py --stage all
```

The standard matched judged budget is 14 features × two matched adapter conditions × 100 held-out scenarios = 2,800 **paired scenarios**, with two generation arms per pair: **5,600 continuations per model**. At two judges and batches of 20 responses per request, this corresponds to **560 judging requests per model**, excluding retries; token usage and actual prices determine monetary cost. Generation and judging are separately invoked. Direction extraction saves only small unit vectors and calibration scales, never token activations. Cross-model replication and synthesis use the standalone cross-model-analysis package.

The implementation is intended for a single GPU with sequential base/adapter loading. Its numerical correctness, model-specific compatibility, observed memory use and full evaluation integration still require an end-to-end GPU pilot before interpreting research findings.


## Clean output layout and final-only publication

New runs write to `mechanistic-interpretability/outputs/<model>/<stage>/`, with no revision, iteration or batch directories in the public results tree. The optimized screening implementation keeps resumable intermediate artifacts in `activation-steering/.work/`, excluded from final exports. Older experimental output directories are historical and need not be migrated or deleted. Generation batch size and protocol are recorded in each generated row for auditability, not in its directory name.

To publish **only final results**, run the dry-run exporter followed by an explicitly authorized export:

```bash
cd mechanistic-interpretability
python export_final_results.py --model llama31
python export_final_results.py --model llama31 --execute
```

The exporter copies finished model-difference, probing and LoRA scalar files, complete steering screens and selections, judgment summary and aggregated matched outcomes, and final plots into `results/mechanistic-interpretability/<model>/`. It deliberately excludes raw `judged/` generations, judge-specific files, partial checkpoints, caches, pilots, temporary archives, and large direction tensors. It does not delete anything. Existing different results are protected unless `--overwrite` is specified. Before claiming that an evaluation is complete, separately validate generation/judge coverage, truncation rates and output quality. The Llama matched evaluation produced 5,600 nonempty generations; 3,545 (63.3%) reached the 256-token cap, a consequential limitation requiring disclosure and a representative completion-quality review.

A model with an older `outputs/revision_2/<model>/` location can be exported without rearranging that GPU instance by specifying `--source outputs/revision_2/<model>`. For copying directly to a fresh Codespace, transfer the **model's finished GPU `outputs/<model>/` directory** separately from this repository and run the exporter there; if a transfer should contain only publishable artifacts, export into a staging directory first. Existing `results/mechanistic-interpretability/v0/` historical outputs should remain intact.


## Portable judged continuation on GPU instances

Use the same checked-in code on each GPU instance after restoring its model-specific adapter tensors and its existing `outputs/<model>/activation-steering/` selection and direction artifacts. The new `run_judged_remainder.py` is independent of instance mount paths, uses `config/pipeline.yaml`, and never changes the model family or the held-out scenario prompt. Check required direction tensors, base access and adapter identity before an expensive full run. A fresh Gemma instance also needs its original checkpoint tensors and provenance summaries; scalar results alone cannot recreate them.

Run from `mechanistic-interpretability/`:

```bash
# Free local generation only; safe to resume after interruption.
python run_judged_remainder.py --model qwen3 --stage generate
python run_judged_remainder.py --model mistral --stage generate

# Separately authorize paid judging after reviewing the generated continuations.
python run_judged_remainder.py --model qwen3 --stage judge --execute-judges
python run_judged_remainder.py --model mistral --stage judge --execute-judges

# Alternatively, generate, verify, judge, aggregate, plot, export in one run:
python run_judged_remainder.py --model gemma2 --stage all --execute-judges
```

Only run one model command per GPU and use the model family matching the instance. The generator saves exactly the same directory layout for all families. It archives legacy/incompatible raw `judged/` and `judgments/` outputs under the *local*, excluded `activation-steering/.work/archived_judged_<timestamp>/`. It does not touch completed layer screens, selected configurations, direction tensors, LoRA summaries or other families. V2 checkpoints resume in place. A strict coverage/pair-consistency gate and a 25% overall-and-per-arm token-limit threshold prevent paid judge invocation if generation is incomplete or too truncated. Surface sentence stopping is heuristic, so inspect a stratified sample of actual continuations and output-language shifts before relying on the outcome. The default `generate` stage never invokes remote APIs. `judge` and `all` require explicit `--execute-judges` authorization; paid judgments are not initiated by a repository pull.

The final exporter retains only small publishable scalar/plot outputs, under `results/mechanistic-interpretability/<model>/`, while GPU-private raw responses and judge files remain in `mechanistic-interpretability/outputs/<model>/`. On a different machine, copy required raw inputs and adapter artifacts before resuming; do not mistake the exported results for model checkpoints. Conflicting previously published results require explicit `--overwrite-export` after reviewing the intended replacement. Cross-model analysis is run only after all four families' final results are present.
