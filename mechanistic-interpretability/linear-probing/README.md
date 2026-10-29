# Linear Probing

Three tasks test linear decodability from frozen final-token and masked-mean activations at all residual and MLP-output layers: (1) canonical-versus-feature linguistic identification on matched semantic IDs, (2) welfare-concept detection on balanced labeled prompt pairs and (3) prediction of the model-condition's original two-judge aggregated `target_value_support` from scenario prompts only.

Use semantic-ID/scenario-grouped 70/15/15 train/development/test splits. Linguistic probes are fit for base, registered control and feature adapter; cross-language accuracy cannot isolate grammar from language. Classification uses standardized linear logistic probes. Behavioral regression uses `dual_ridge_v2`, train-only normalization, a development-selected ridge penalty, held-out prediction records, 0–4 clipping diagnostics and comparisons against the train-set-mean baseline. Compare against simple lexical and length cues using `classifier_control_audit.json`. Do not reuse legacy behavioral estimators.

Development-selected layers or estimators must be reported as selected; held-out checks conditional on those choices do not remove selection uncertainty. Probe performance does not show that the model causally uses a feature. Token activations are not saved. See the [parent methodology](../README.md).
