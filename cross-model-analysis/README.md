# Cross-Model Analysis

## Research objective

This analysis examines the replication of animal-welfare value-generalization effects across four language-model families and compares their relationships with learned representations, probes, LoRA updates, and activation-steering interventions. It separates replicable behavioral effects from architecture-specific patterns and evaluates robustness, generalization, and plausible alternative explanations.

## Analyses

The pipeline compares all 14 linguistic transformations, the 11 within-English contrasts, and the three language-changing contrasts separately. It reports per-family effects and confidence intervals; corrected significance, sign and ranking agreement; pairwise correlations and leave-one-feature-out sensitivity; canonical and source-language baseline gains; category and held-out-dimension heterogeneity; tradeoff interactions; coherence, commitment, lexical-diversity sensitivity, and judge agreement. Available mechanistic scalar results are compared through CKA/SVCCA depth profiles, held-out probes, LoRA geometry and functional update metrics, activation-steering summaries, and exploratory behavior–mechanism associations.

## Figures

Figures are generated automatically with the tables. The publication suite includes forest plots, replication heatmaps, correlation matrices, baseline and generalization plots, tradeoff interactions, coherence/quality and lexical diagnostics, inter-judge reliability, representation-depth profiles, probe curves, LoRA comparisons, held-out steering effects, and cross-stage associations. Figures are produced only when the corresponding data exist.

Each figure is exported as **vector PDF**, **editable SVG**, **600-DPI PNG**, and **the exact plotted source data as CSV**. A machine-readable `figures/manifest.json` records figure captions, data sources, unavailable panels, and dependency versions. Consistent architecture colors, feature ordering, language-contrast markers, and explicit uncertainty/significance conventions are used throughout.

## Interpretation

The original semantic-cell-cluster bootstrap intervals and within-family Benjamini–Hochberg q-values are preserved. Four models evaluated on shared scenarios do not constitute independent model samples, so cross-family synthesis is descriptive. Cross-language interventions are confounded with language identity. Probe decodability and representational or LoRA similarity are not causal explanations; steering results require a held-out evaluation, and cross-stage correlations remain exploratory. Missing measurements are never imputed or displayed as findings.

## Reproduction

From the repository root:

```bash
python -m pip install -r cross-model-analysis/requirements.txt
python cross-model-analysis/run.py --stage all
python -m unittest discover -s cross-model-analysis/tests -v
```

Additional options: `--stage behavioral`, `--stage mechanistic`, `--mech-root PATH`, `--output PATH`, `--skip-figures`, `--figures-only`, and `--figure-dpi 600`.

Inputs are the committed `results/<model>/evaluation_results/analysis/output/` files and, when available, scalar outputs in `mechanistic-interpretability/outputs/<family>/`. The pipeline writes all generated results to `cross-model-analysis/output/`. Figure production does not load GPU model weights or make judging API requests.
