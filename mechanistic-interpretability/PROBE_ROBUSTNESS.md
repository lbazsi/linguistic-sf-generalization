# Probe robustness audit (revision 3)

The original revision-2 ridge metrics, coefficients and selected development layers are preserved. A supplemental forward-only audit reconstructs the selected held-out predictions from the existing weights and train-only normalization, rejecting mismatched test RMSE, MAE and R². No model fitting, development retuning or test-based layer selection occurs.

The audit reports both raw and [0,4]-clipped prediction accuracy, cluster-bootstrap intervals resampled by semantic cell (2,000 replicates), baseline-relative RMSE improvements, actual per-scenario predicted/target scores, and paired original judge disagreement. Its uncertainty is **conditional on the development-selected probe**, not on architecture or model-selection uncertainty. Both judges score the same generated response, so this can only assess rater disagreement; estimating generation variability requires new sampled completions and judging.

Run from `mechanistic-interpretability` on the Llama instance:

~~~bash
git fetch origin
git pull --ff-only origin research/mechinterp-v2-20261026
python -m pytest tests -q
python -m mechinterp.probe_audit --model llama31 --condition canonical
python -m mechinterp.probe_audit --model llama31 --all
~~~

This uses config/probe_robustness.yaml, reads existing results in outputs/revision_2/llama31/linear-probing and writes distinct reports under outputs/revision_3_probe_robustness/llama31/linear-probing. The `--all` option repeats only the held-out forward passes for each condition; the canonical audit can be skipped if already produced. The saved original regression weights remain untouched.

All future family-level probe experiments can use the robustness config directly, in which case the ridge runner generates the same supplemental artifacts during training:

~~~bash
python run_all.py --model mistral --stage behavior-probes --config config/probe_robustness.yaml --execute --resume
~~~

Report the development-selected probe once per family/condition. Avoid counting the 128 correlated layer/site/pooling combinations as independent observations, comparing R² values across differently distributed condition labels without caution, or interpreting rater agreement as a theoretical noise ceiling.
