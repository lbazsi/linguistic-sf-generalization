# Cross-Model Results

## Scope and interpretation

Comparisons cover four model families evaluated on the same animal-welfare scenario design. The primary estimand is feature-adapter minus registered language-matched control on the 0–4 target-value-support scale. The upstream analysis provides clustered confidence intervals and per-family FDR-adjusted q-values. No independent-family meta-analytic inference or seed-level replication is assumed.

## Behavioral replication

**All-family sign-consistent, BH-FDR-positive features:** constituent order, epistemic hedging, fusion, inflectional synthesis, narrative framing, nominalization.

| Linguistic feature | Gemma 2 | Llama 3.1 | Mistral | Qwen 3 | FDR-positive families |
|---|---:|---:|---:|---:|---:|
| conditionality | -0.038 | -0.047 | -0.022 | -0.016 | 0 |
| constituent order † | -0.196 | -0.254 | -0.410 | -0.180 | 4 |
| epistemic hedging | -0.139 | -0.049 | -0.078 | -0.066 | 4 |
| formality | -0.012 | -0.003 | +0.001 | -0.009 | 0 |
| fusion † | -0.116 | -0.161 | -0.104 | -0.236 | 4 |
| genericity | +0.013 | -0.021 | +0.011 | -0.031 | 0 |
| inflectional synthesis † | -0.064 | -0.267 | -0.151 | -0.154 | 4 |
| modal strength | -0.030 | -0.007 | +0.002 | +0.004 | 0 |
| narrative framing | -0.052 | -0.117 | -0.070 | -0.058 | 4 |
| nominalization | -0.054 | -0.106 | -0.071 | -0.054 | 4 |
| pronoun distribution | -0.038 | -0.014 | +0.001 | +0.011 | 0 |
| subordination | -0.022 | -0.063 | -0.001 | -0.048 | 1 |
| tense aspect | -0.010 | -0.003 | +0.048 | -0.037 | 1 |
| voice | -0.024 | -0.031 | -0.036 | -0.037 | 0 |

† Language-changing contrast; the grammatical intervention is confounded with a change in training language. Effects are not welfare opposition, but changes relative to their appropriate fine-tuning controls.

### Cross-family effect agreement

| Feature set | First family | Second family | Pearson r | Spearman ρ | Sign agreement |
|---|---|---|---:|---:|---:|
| all_14 | gemma | llama31 | +0.674 | +0.766 | 0.93 |
| all_14 | gemma | mistral | +0.861 | +0.904 | 0.71 |
| all_14 | gemma | qwen3 | +0.713 | +0.691 | 0.79 |
| all_14 | llama31 | mistral | +0.843 | +0.889 | 0.64 |
| all_14 | llama31 | qwen3 | +0.844 | +0.851 | 0.86 |
| all_14 | mistral | qwen3 | +0.714 | +0.812 | 0.79 |
| within_english_11 | gemma | llama31 | +0.388 | +0.633 | 0.91 |
| within_english_11 | gemma | mistral | +0.745 | +0.838 | 0.64 |
| within_english_11 | gemma | qwen3 | +0.459 | +0.405 | 0.73 |
| within_english_11 | llama31 | mistral | +0.769 | +0.779 | 0.55 |
| within_english_11 | llama31 | qwen3 | +0.689 | +0.727 | 0.82 |
| within_english_11 | mistral | qwen3 | +0.580 | +0.638 | 0.73 |

### Baseline gains

| Model | Contrast | Weighted mean gain |
|---|---|---:|
| gemma | canonical_minus_base | +0.906 |
| gemma | mandarin_control_minus_base | +0.773 |
| llama31 | canonical_minus_base | +1.088 |
| llama31 | mandarin_control_minus_base | +0.953 |
| mistral | canonical_minus_base | +0.786 |
| mistral | mandarin_control_minus_base | +0.558 |
| qwen3 | canonical_minus_base | +0.688 |
| qwen3 | mandarin_control_minus_base | +0.620 |

### Robustness and heterogeneity

Original category-specific, held-out-dimension, tradeoff, coherence, judge-reliability, and sensitivity tables are retained as source-data exports. Feature-level lexical correlations are exploratory and do not establish mediation.

## Mechanistic comparison

Analyzed 658 mechanistic scalar files and 562260 individual scalar records. Representational, probe, LoRA, and steering comparisons are descriptive unless the original experiment provides an independently judged held-out contrast.

## Figure index

- **fig01_feature_effects_forest** — Feature-versus-registered-control effects in all four models; whiskers are the original 95% semantic-cell bootstrap intervals. ([vector PDF](figures/behavioral/fig01_feature_effects_forest.pdf), [source data](figures/behavioral/fig01_feature_effects_forest.csv))
- **fig02_replication_heatmap** — Per-model welfare-support effects for all linguistic features. ([vector PDF](figures/behavioral/fig02_replication_heatmap.pdf), [source data](figures/behavioral/fig02_replication_heatmap.csv))
- **fig03_architecture_correlation** — Pairwise descriptive correlations of feature-effect profiles by model. ([vector PDF](figures/behavioral/fig03_architecture_correlation.pdf), [source data](figures/behavioral/fig03_architecture_correlation.csv))
- **fig04_baseline_gains** — Canonical and source-language control gains relative to each pretrained model, stratified by generalization category. ([vector PDF](figures/behavioral/fig04_baseline_gains.pdf), [source data](figures/behavioral/fig04_baseline_gains.csv))
- **fig05_generalization_categories** — Generalization-category heterogeneity. Original cluster-adjusted statistics are retained in the source-data export. ([vector PDF](figures/behavioral/fig05_generalization_categories.pdf), [source data](figures/behavioral/fig05_generalization_categories.csv))
- **fig06_heldout_dimensions** — Effects by number of held-out dimensions. Original cluster-adjusted statistics are retained in the source-data export. ([vector PDF](figures/behavioral/fig06_heldout_dimensions.pdf), [source data](figures/behavioral/fig06_heldout_dimensions.csv))
- **fig07_tradeoff_interactions** — Tradeoff versus non-tradeoff interactions. Original cluster-adjusted statistics are retained in the source-data export. ([vector PDF](figures/behavioral/fig07_tradeoff_interactions.pdf), [source data](figures/behavioral/fig07_tradeoff_interactions.csv))
- **fig08_high_coherence_sensitivity** — Comparison of primary feature estimates with estimates on high-coherence responses. ([vector PDF](figures/behavioral/fig08_high_coherence_sensitivity.pdf), [source data](figures/behavioral/fig08_high_coherence_sensitivity.csv))
- **fig09_lexical_associations** — Association between feature-level lexical-diversity change and welfare-support differences. ([vector PDF](figures/behavioral/fig09_lexical_associations.pdf), [source data](figures/behavioral/fig09_lexical_associations.csv))
- **fig10_judge_reliability** — Inter-judge agreement on welfare support, behavioral commitment, and tradeoff priority. ([vector PDF](figures/behavioral/fig10_judge_reliability.pdf), [source data](figures/behavioral/fig10_judge_reliability.csv))
- **fig11_response_coherence** — Feature-level relationship between changes in welfare-support ratings and response-coherence ratings. ([vector PDF](figures/behavioral/fig11_response_coherence.pdf), [source data](figures/behavioral/fig11_response_coherence.csv))
- **fig12_cka_depth** — Representation similarity over network depth (welfare prompts) ([vector PDF](figures/mechanistic/fig12_cka_depth.pdf), [source data](figures/mechanistic/fig12_cka_depth.csv))
- **fig13_svcca_depth** — unavailable: Legacy in-sample SVCCA invalid; requires independent fit/evaluation rerun.
- **fig14_linguistic_probes** — Held-out linguistic-feature decodability over depth ([vector PDF](figures/mechanistic/fig14_linguistic_probes.pdf), [source data](figures/mechanistic/fig14_linguistic_probes.csv))
- **fig15_behavior_probes** — Held-out welfare-behavior linear-probe performance ([vector PDF](figures/mechanistic/fig15_behavior_probes.pdf), [source data](figures/mechanistic/fig15_behavior_probes.csv))
- **fig16_lora_geometry** — LoRA update distance versus registered controls ([vector PDF](figures/mechanistic/fig16_lora_geometry.pdf), [source data](figures/mechanistic/fig16_lora_geometry.csv))
- **fig18_steering_judged** — Independently judged held-out steering effects ([vector PDF](figures/mechanistic/fig18_steering_judged.pdf), [source data](figures/mechanistic/fig18_steering_judged.csv))
- **fig17_behavior_mechanism** — Associations between welfare-support effects and comparable scalar mechanistic measurements. ([vector PDF](figures/mechanistic/fig17_behavior_mechanism.pdf), [source data](figures/mechanistic/fig17_behavior_mechanism.csv))

## Limitations

Shared benchmark and fine-tuning infrastructure limit the independence of model replications. Only the sampled model architectures and training seeds are covered. Language-changing contrasts cannot isolate syntax or morphology from language identity. Agreement among automated judges is not equivalent to human validation. Representation similarity and probe performance are descriptive; steering is an intervention-specific test, not proof of a unique causal mechanism. Cross-stage correlations are exploratory and potentially sensitive to feature choice, module aggregation, and response quality.

## Reproducibility

The per-family source files, original uncertainty estimates, figure source-data CSVs, and figure manifest are the provenance record. Plots never introduce inferential statistics beyond the source analysis.
