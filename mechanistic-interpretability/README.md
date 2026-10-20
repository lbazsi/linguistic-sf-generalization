# Mechanistic Interpretability of Linguistic Fine-Tuning Effects

## Research objective

This study characterizes how semantically matched linguistic variants of animal-welfare fine-tuning data affect model representations, learned LoRA updates, and downstream animal-welfare value generalization. The analyses are observational except for exploratory activation steering. Results are compared across four pretrained model families to evaluate replication.

The five stages are (1) model-difference analysis, (2) linear probing, (3) LoRA analysis with midpoint-to-final comparison, (4) linguistic-direction activation steering, and (5) replication and synthesis across models. The stage-specific methods are in [Model-difference-analysis](Model-difference-analysis/README.md), [linear-probing](linear-probing/README.md), [LoRA-analysis](LoRA-analysis/README.md), and [activation-steering](activation-steering/README.md). Replication is specified here because the same four stages are applied across architectures.

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
- Process one layer/batch/condition at a time. **Token activations are GPU-resident, transient and not persisted**: compute and accumulate sufficient statistics or probe batches online, then release buffers. Do not cache raw token activations, full activation matrices, or merged model checkpoints. The originally required midpoint/final adapter tensors remain necessary input artifacts, stored outside the repository; this project does not create additional heavyweight checkpoints.
- Persist small artifacts sufficient to audit conclusions: run manifests, data/split hashes, model/adapter identifiers, hook maps, aggregate covariances or similarity summaries where practical, probe coefficients and metrics, small normalized steering directions if needed for rerunning steering (not token activations), scalar evaluation outputs, and plots/tables. Omit any summary that would reveal private prompt text if inputs are sensitive.
- Device memory use depends on layer width, batch size, prompt length and whether modules emit logits or full token matrices; perform an A100 memory pilot rather than assuming unrestricted all-layer retention.

## Sequence and outputs

1. Verify base revisions, adapter archives, exact datasets, condition-to-control mappings and GPU compatibility. Generate and validate the 1,000 general prompts and concept-probe dataset.
2. Run [model-difference analysis](Model-difference-analysis/README.md) for welfare and general prompts; summarize layer/module sensitivity.
3. Run [linear probes](linear-probing/README.md) for linguistic identification, welfare-concept detection and welfare-behavior prediction using a fixed estimator protocol.
4. Run [LoRA analysis](LoRA-analysis/README.md) over all adapters, modules and midpoint/final steps.
5. Run [activation steering](activation-steering/README.md) for **all 14 features** with separately analyzed cross-language contrasts; use validation to select layer/site and strength and hold out the test data.
6. Repeat the full training/evaluation/analysis protocol for all four families and synthesize: behavioral effect sign/magnitude and uncertainty; depth-normalized similarity profiles; probe discrimination/calibration and adapter-transfer gaps; relative LoRA geometry; steering dose-response and linguistic-feature manipulation success. Compare per-feature patterns across architectures without comparing raw coordinates across different hidden spaces.

## Inference and limitations

Report paired scenario-level effects with 95% semantic-cell-cluster bootstrap confidence intervals; use cluster-aware sign-flip/permutation tests where appropriate and Benjamini–Hochberg FDR within predeclared outcome families. Separately show feature-level and held-out-domain effects. With four distinct model families, consistent signs and patterns are strong **cross-model replication evidence within this sample**, not a universal claim; seed-level uncertainty cannot be assessed with one seed. Probes demonstrate decodability, and differences in representation or LoRA parameters are descriptive. Steering supports limited intervention-based evidence but can move activations off-distribution.

## Reproducibility contract

Record resolved Hugging Face base-model revision, adapter hash and checkpoint step, package/CUDA/PyTorch/Transformers/PEFT versions, hardware, random seeds, full evaluation configuration, corpus and prompt hashes, split ID lists, hook definition, pooling rule, regularization choices, normalization calibration, predeclared search space, and scoring revisions. Transient activations and model weights are deliberately excluded from committed artifacts; reproducing results requires regenerating the activations from the original inputs and external base/adapter checkpoints. Documentation here specifies planned analyses; it does not claim they have already been executed.
