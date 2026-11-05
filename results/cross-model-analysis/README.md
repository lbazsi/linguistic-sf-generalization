# Cross-Model Analysis Results

## Scope

This analysis compares behavioral and mechanistic results from Gemma 2, Llama 3.1, Mistral, and Qwen 3. It distinguishes the 11 within-English feature contrasts from three language-changing contrasts and preserves each model's original uncertainty intervals and false-discovery-rate adjustments. Cross-family comparisons are descriptive; the shared benchmark and pipeline do not support an independent-family meta-analysis.

## Behavioral replication

Canonical fine-tuning increased target-value support relative to the pretrained base in every family (mean gains of +0.688 to +1.088 on the 0–4 scale). Six of 14 feature conditions showed negative, FDR-significant feature-minus-control effects in all four families: constituent order, epistemic hedging, fusion, inflectional synthesis, narrative framing, and nominalization. The first, third, and fourth of these involve language changes, so their effects cannot be attributed solely to the targeted grammatical feature.

Across the 14 feature effects, pairwise between-family Pearson correlations ranged from 0.674 to 0.861; for the 11 within-English contrasts, they ranged from 0.388 to 0.769. Category-level effects, held-out dimensions, tradeoff interactions, response quality, and judge agreement are reported separately to show heterogeneity rather than assume uniform replication.

## Mechanistic comparison

The standardized outputs cover 658 mechanistic scalar-result files, 562,260 scalar records, and 1,090 comparable profiles. Feature and control representations remained broadly similar, linguistic and concept classifiers were highly decodable, and welfare-score prediction was weaker. LoRA updates differed measurably between conditions. Independently judged activation steering produced five FDR-significant configurations across Gemma 2 and Llama 3.1, but none in Mistral or Qwen 3. Associations between behavioral effect sizes and mechanistic measurements are exploratory; no common causal mechanism is established.

## Figures

![Cross-model feature-effect replication heatmap](figures/behavioral/fig02_replication_heatmap.png)

*Overall feature-minus-control effects in each architecture.* [Vector figure](figures/behavioral/fig02_replication_heatmap.svg) · [Plotted data](figures/behavioral/fig02_replication_heatmap.csv)

![Correlations between model-family effect profiles](figures/behavioral/fig03_architecture_correlation.png)

*Pairwise effect-profile correlations, with all-feature and within-English comparisons.* [Vector figure](figures/behavioral/fig03_architecture_correlation.svg) · [Plotted data](figures/behavioral/fig03_architecture_correlation.csv)

Further visualizations and source-data exports are indexed in [the results report](research_results.md), including [behavior–mechanism associations](figures/mechanistic/fig17_behavior_mechanism.svg). Machine-readable tables are under `behavioral/`, `mechanistic/`, and `cross_stage_associations.csv`. The analysis code is in [`cross-model-analysis/`](../../cross-model-analysis/).
