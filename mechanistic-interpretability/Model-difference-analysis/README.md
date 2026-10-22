# Model-Difference Analysis

## Question

Where do representations diverge after animal-welfare fine-tuning, and which additional changes distinguish a feature adapter from its appropriate linguistic control? Do differences vary between animal-welfare situations and general-domain text?

## Inputs and contrasts

For every model family, load base, canonical, Mandarin control, and all feature adapters. Evaluate the existing 800 welfare scenario prompts and a separate fixed corpus of 1,000 diverse non-welfare English prompts. Use exactly the same prompt strings within each model-family comparison. The contrast registry is: canonical−base, Mandarin control−base, 11 English features−canonical, constituent_order−canonical, and inflectional_synthesis/fusion−Mandarin control. Feature-to-feature exploratory comparisons must be labeled separately and should not replace controlled contrasts.

## Extraction

Use frozen models in evaluation mode and deterministic forward passes, with no generation required for representation similarity. For **every transformer block**, capture (i) residual stream after the block and (ii) that block's MLP output before addition to residual; include embeddings as an optional reference layer. Specify each hook precisely for Gemma, Qwen, Llama and Mistral. Do not confuse MLP internal gate/up/down projections with the entire MLP output.

At each site extract final non-padding prompt token and masked mean-pooled prompt tokens. Store only transient GPU tensors; for each model/prompt condition, compute similarity statistics in streaming batches. If SVCCA requires example-by-feature matrices, retain *only the selected pooled activations transiently on GPU* for the computation and immediately discard them. Do not export exact activations. Attention output analysis is not included in this stage; the attention projection modules are examined in LoRA analysis.

## Metrics

**Linear CKA (primary):** center activation observations, compute normalized linear CKA using a numerically stable Gram-based or streamed covariance formulation. Compare corresponding layers and sites across adapter conditions, not raw vectors from different architectures. Report `1-CKA` as representational distance and CKA as similarity. Control for sample count; use matching IDs, equal category weighting where required, and randomized minibatch order.

**SVCCA (secondary):** apply dimensionality reduction separately per condition on training/calibration observations, retain enough principal components for a specified explained-variance threshold (default 99%, capped below available sample rank), and compute mean canonical correlations on matched evaluation observations. Avoid degenerate >sample-rank comparisons and assess robustness to PCA threshold and subset size. The current SVCCA implementation uses the same observations for PCA and correlation estimation, so it is a descriptive similarity estimate rather than an independent held-out estimate.

**Contrast magnitude (descriptive):** optionally report root-mean-square activation displacement normalized to baseline activation RMS on the same prompts; this complements similarity, which can be insensitive to scale.

## Experimental procedure

1. Validate checkpoint identity, adapter/control mapping and matching prompt IDs. Tokenize separately per architecture while keeping the same underlying text.
2. Extract each site's pooled representations across the two prompt domains, computing CKA/SVCCA and displacement summaries; release GPU arrays after each calculation.
3. Create layerwise heatmaps for each adapter contrast, activation site and pooling type, separately for welfare vs general-domain prompts.
4. Calculate domain specificity as a *difference between within-domain distances*, not a raw comparison between welfare and general activation vectors. Report per-category welfare breakdowns and semantic-cell-cluster intervals where feasible.
5. Record layers and sites with the strongest differences for interpretation; do not use the held-out steering test set for layer screening.

## Outputs

Compact tables (model family, condition/control, step, domain, site, layer, pooling, n, CKA, SVCCA, displacement and uncertainty), layerwise figures and run manifests. Results from midpoint adapters may be included descriptively but the primary comparisons use final adapters; checkpoint change is evaluated in the LoRA stage.

## Interpretation

Low similarity does not identify a mechanism, and high similarity does not imply identical model behavior. MLP output similarity can differ from residual-stream similarity even at the same block. Across-family comparisons use normalized layer depth and comparable effect statistics, never hidden-vector coordinate alignment.


## Implementation

Comparison models are loaded **sequentially** on the GPU; every forward batch contributes only final-token and mean-pooled per-layer residual/MLP vectors. These pooled representations can remain temporarily on the accelerator for CKA/SVCCA computation, then are discarded. Neither full token activations nor pooled activation matrices are saved. The output includes whole-domain results and separate similarity estimates for each of the eight animal-welfare generalization categories, with layer, site, pooling and control condition attached. SVCCA is descriptive and uses the same observation sample for its PCA and canonical-correlation fit.

Run `python run.py difference --model gemma2 --condition epistemic_hedging`; all registered controlled contrasts can be scheduled with `python run_all.py --model gemma2 --stage difference --execute --resume`.
