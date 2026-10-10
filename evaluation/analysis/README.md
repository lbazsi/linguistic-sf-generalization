# Black-box evaluation analysis

This directory contains the post-hoc statistical analysis and figure-generation pipeline for the completed black-box evaluation.

It consumes the evaluation outputs in `../data/` and writes a self-contained analysis bundle to `output/` containing research tables, figures, diagnostics, a Markdown report, and an input-hash manifest.

The analysis includes paired feature-minus-control effects, canonical-minus-base and control-minus-base diagnostics, semantic-cell cluster bootstrap confidence intervals, cluster sign-flip tests, Benjamini-Hochberg FDR correction, tradeoff-stratified analyses, judge agreement diagnostics, complete-case and high-coherence sensitivity analyses, and lexical-diversity association analysis.

Run from this directory:

```bash
pip install -r requirements.txt
python run_analysis.py
```

Useful options:

```bash
python run_analysis.py --bootstrap-iters 10000 --permutation-iters 20000 --seed 20261010
python run_analysis.py --evaluation-root /path/to/evaluation --output-dir /path/to/output
```

Primary effect estimates are raw differences on the 0–4 `target_value_support` scale. Uncertainty is clustered by semantic cell (`category × animal × value × context`) so repeated textual realizations of the same semantic assignment are not treated as fully independent. Multiple-testing correction is applied separately to pre-specified analysis families and the unadjusted estimates are retained alongside corrected q-values.

Generated outputs are intentionally not committed by default; `output/` is ignored.
