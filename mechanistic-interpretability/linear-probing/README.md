# Linear Probing

## Questions

Which linguistic features and animal-welfare concepts are linearly decodable from representations, and can activations predict graded animal-welfare support on the established evaluation benchmark? How does decodability vary across layers, modules, adapters and model families?

## Probe families and datasets

1. **Linguistic feature identification:** for each of the 11 English interventions, use paired canonical/variant training texts from the same semantic IDs. Binary labels denote canonical=0 and variant=1. Train separate binary probes per feature; use semantic-ID-disjoint train/development/test splits, equal labels and matched semantic distributions. A pooled multiclass analysis can supplement, not replace, pairwise contrasts. The three language-changing interventions are analyzed as a separate probe family with their registered English/Mandarin control and target-language texts; resulting discrimination measures language/typology jointly and should not be called a pure grammatical probe.
2. **Animal-welfare concept detection:** use a separate balanced positive/negative text set with labels assigned from the text, not from the model's behavior. Negatives include animal-related but non-welfare statements as well as topic-matched non-animal statements; match surface properties, lexical cues, prompt length and semantic contexts where possible. Audit via adversarial simple baselines (e.g. animal-token presence) to guard against shortcut learning. Split by scenario/template or semantic cluster.
3. **Behavioral-support prediction:** use the existing 800 scenario inputs, with *the relevant model-condition's unsteered, two-judge aggregated* `target_value_support` as the supervised target, on its valid 0–4 scale. These are model-output-derived labels rather than ground truth welfare concepts. Fit on held-out grouped subsets and report whether activations forecast the model's judged continuation. Do not use the judged continuation itself as probe input or conflate this prediction with a concept representation. Evaluate within model condition; cross-condition generalization is a separate transfer experiment.

## Representations and estimators

Use the final prompt token and masked mean-pooled prompt token representations from **every residual-stream and MLP-output layer**, matching the model-difference stage. Fix the same probe model class, tuning procedure and preprocessing across features, layers and families:

- Classification: GPU-trained linear logistic classifier with weight decay, trained on balanced classes; standardize each activation dimension using streaming training-fold statistics only.
- Behavioral regression: linear squared-error predictor with weight decay on the 0–4 outcome, with the same feature standardization.
- Hyperparameters: a fixed training schedule, optimizer and regularization strength for all tasks, features, sites and model families; development scores are reported without per-task hyperparameter selection.
- Primary splits: 70% train / 15% development / 15% test, grouped by semantic ID or scenario cluster; persist IDs and guarantee source/variant examples of each semantic ID occupy the same partition. The 800 scenarios provide limited independent semantic clusters; present uncertainty accordingly.
- Primary scores: balanced accuracy and ROC-AUC (linguistic/concept classification); RMSE, MAE, held-out R² and correlation (behavioral regression). Include constant-majority/mean and simple lexical shortcut baselines.
- Keep probes deliberately linear; no nonlinear MLP probe comparisons in the primary results.

Train probes separately for each model family, adapter condition, site, layer and pooling. Intra-family cross-adapter transfer can be tested without coordinate alignment (fit on base representations, evaluate on canonical/feature adapters and vice versa) if hidden sizes and hook definitions coincide. For *different architecture families*, do not directly apply weights across incompatible latent coordinate systems; compare metrics/profiles. Report both within-adapter performance and transfer drops, with frozen training-derived normalization.

## Procedure

1. Freeze all models; verify exact semantic-ID and scenario splits and label integrity.
2. Run forward passes and collect one temporary batch of activations at a time. Fit standardized linear probes using GPU-resident or ephemeral batches/accumulated sufficient statistics as applicable, then release raw activations. Logistic regression may require repeated passes or a streaming optimizer; ensure it matches the fixed regularized logistic objective and preserves train/dev/test separation.
3. Use fixed regularization and epochs, report development performance and evaluate once on test partitions; report all layers, not just the maximum.
4. Compare linguistic probe generalization, welfare-concept detection and behavior-prediction profiles across base/control/feature adapters, separately for each model.
5. Use semantic-cell bootstrap intervals for welfare-related held-out results, with multiple-comparison correction for predeclared feature/layer summary families.

## Outputs and interpretation

Store data/split hashes, feature labels, normalization summaries, learned linear-probe coefficients (small relative to model weights), hyperparameters, test predictions/scores, aggregate layerwise reports and plotting data; do **not** save token activations. Coefficient vectors are coordinate-specific, so cross-family comparison is at the level of performance patterns, not cosine similarity of weights. Successful classification establishes that information is decodable, not that the model uses it when choosing a response.


## Implementation

The default linguistic-probe sample consists of 1,000 semantic-ID-matched canonical/variant pairs, balanced over the available animal × value × context cells; every pair remains in one split. The fixed GPU streaming procedure uses 4 optimizer epochs, AdamW (learning rate 0.01, weight decay 0.001) and identical training-fold standardization across conditions. One forward pass per batch supplies final/mean pooled representations from every layer and from residual/MLP sites to independent linear heads. Probe weights and standardization statistics are saved, but no activations are retained. Concept probes use all 1,000 matched examples; behavioral probes require each model family's own original two-judge scores. Linguistic probes run for the base model, a feature's registered control, and that feature's adapter, not unrelated feature adapters. Across-family comparisons use predictive performance, not coefficient alignment.

The implementation outputs `linear-probing/*.jsonl` and small `*_weights.safetensors` files. Model-family-specific behavioral label files must be configured before replication. Direct application of probe coefficients to different model architectures is not defined.
