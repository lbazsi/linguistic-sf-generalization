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

The model configuration expects existing independently trained checkpoint trees, including midpoint/final adapters and their training summaries. Adapter bytes are not stored in this repository. Each additional model must have its revision pinned and adapters generated via the common fine-tuning contract. Results are written under `outputs/<model>/<stage>/` and are not input checkpoints.

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

The standard judged budget is 14 features × two matched adapter conditions × 100 held-out scenarios = 2,800 continuations per model. At two judges and ten items per API request, this corresponds to 560 judging requests per model, excluding retries; token usage and actual prices determine monetary cost. Generation and judging are separately invoked. Direction extraction saves only small unit vectors and calibration scales, never token activations. Cross-model replication and synthesis use the standalone cross-model-analysis package.

The implementation is intended for a single GPU with sequential base/adapter loading. Its numerical correctness, model-specific compatibility, observed memory use and full evaluation integration still require an end-to-end GPU pilot before interpreting research findings.
