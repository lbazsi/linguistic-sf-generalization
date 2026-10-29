# Model-Difference Analysis

Compare each fine-tuned feature adapter with its registered language control using identical general-domain and animal-welfare prompts. The final adapter is the primary checkpoint; the midpoint is investigated through LoRA analysis.

At every transformer layer, extract final-token and masked-mean representations at residual and MLP-output sites. Estimate centered linear CKA (primary), SVCCA (descriptive; dimensionality reduction and correlation can use the same observations), and activation displacement relative to control RMS. Report welfare categories separately, including domain specificity as the difference between *within-domain* representational distances. Keep model and adapter weights fixed, process activations transiently, and retain layerwise scalar outputs and manifests. Similarity does not establish a causally relevant mechanism. See the [parent methodology](../README.md).
