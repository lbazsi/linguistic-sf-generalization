# Cross-Model Analysis

## Research objective

This analysis evaluates whether linguistic fine-tuning effects on animal-welfare value generalization replicate across Gemma 2, Llama 3.1, Mistral, and Qwen 3, and whether corresponding mechanistic measurements exhibit consistent patterns. The purpose is to distinguish reproducible linguistic effects from model-specific differences, assess alternative explanations, and examine associations between behavior, representations, adapter updates, and causal interventions.

## Analyses

1. **Behavioral replication:** Compare feature-minus-registered-control effects, direction, magnitude, uncertainty, FDR significance, rank agreement, pairwise Pearson/Spearman correlations, and leave-one-feature-out sensitivity. Repeat for the 11 within-English features separately from the three language-changing contrasts.
2. **Baseline and generalization:** Compare canonical-minus-base and Mandarin-control-minus-base gains; replicate effects across generalization categories, held-out dimensions, explicit tradeoffs, tradeoff interactions, and welfare-priority judgments.
3. **Robustness and confounding:** Compare judge reliability, coherence, behavioral commitment, invalid-response diagnostics, complete-case/high-coherence sensitivity, and lexical-diversity associations. Keep language identity distinct from grammatical intervention effects.
4. **Representational replication:** Compare depth-normalized CKA, SVCCA, and displacement profiles across model families, layers, residual/MLP sites, pooling methods, and welfare versus general-domain inputs.
5. **Probe replication:** Compare held-out linguistic classification, welfare-concept decoding, and welfare-behavior prediction, including layer/site patterns and available adapter contrasts.
6. **LoRA replication:** Compare normalized update geometry, subspace alignment, midpoint-to-final development, module classes, and functional output perturbations across model families.
7. **Steering replication:** Compare direction-specific validation profiles and independently judged held-out effects, token policies, dose responses, manipulation success, coherence, and random/shuffled controls when available. Validation-selected proxy results are not confirmatory evidence.
8. **Cross-stage synthesis:** Relate behavioral effect sizes to matched mechanistic summaries and identify consistently replicated, architecture-specific, null, and conflicting findings. Preserve missingness, effect directions, within-/cross-language separation, and measurement comparability.

## Inference and limitations

The input black-box tables retain the original semantic-cell-cluster bootstrap, sign-flip tests, and Benjamini–Hochberg correction. Cross-family summaries are descriptive; four families sharing benchmark inputs and data-generation procedures are not independent replications. Feature-level correlations are exploratory, especially with 11 within-language features. One fine-tuning seed per condition does not quantify seed variance. Probes establish decodability, similarity and LoRA geometry are descriptive, and steering establishes only intervention-specific evidence. Across architectures, compare metrics and normalized depths, not raw hidden-space coordinates.

## Reproduction

Run from the repository root:

```bash
python cross-model-analysis/run.py --stage behavioral
python cross-model-analysis/run.py --stage mechanistic
python cross-model-analysis/run.py --stage all
python -m unittest discover -s cross-model-analysis/tests -v
```

The behavioral analysis reads committed tables under `results/<model>/evaluation_results/analysis/output/`. Mechanistic analysis reads scalar artifacts from `mechanistic-interpretability/outputs/<model>/` (or `--mech-root`) when present. Generated CSV/JSON summaries are placed in `cross-model-analysis/output/`. Missing mechanistic outputs are reported, not substituted with estimates. No GPU, model weights, activations, or judge API calls are required.

The comparative analysis code is separate from the within-model experimental procedures in [mechanistic interpretability](../mechanistic-interpretability/README.md).
