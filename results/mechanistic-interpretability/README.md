# Mechanistic Interpretability Results

## Experimental design

For Gemma 2, Llama 3.1, Mistral, and Qwen 3, we compared linguistic-feature adapters against their registered controls using four complementary analyses: representation similarity, linear probing, LoRA update analysis, and activation steering. Comparisons are matched within an architecture; we do not align raw hidden-state coordinates across architectures.

The analyses ask different questions: similarity measures *how representations change*; probes test *what can be decoded*; LoRA metrics quantify *where and how strongly updates differ*; and steering tests *whether an activation intervention changes responses*. None alone identifies a mechanism mediating the black-box effects.

## Results

| Analysis | Gemma 2 | Llama 3.1 | Mistral | Qwen 3 |
|---|---:|---:|---:|---:|
| Mean welfare-prompt linear CKA | 0.973 | 0.975 | 0.969 | 0.970 |
| Mean general-prompt linear CKA | 0.929 | 0.954 | 0.951 | 0.952 |
| Linguistic-condition probe ROC-AUC | 0.930 | 0.929 | 0.928 | 0.929 |
| Welfare-concept probe ROC-AUC | 0.983 | 0.990 | 0.983 | 0.979 |
| Welfare-score probe held-out R² | 0.062 | 0.093 | 0.082 | 0.133 |
| Final LoRA update relative distance from control | 0.884 | 0.885 | 0.926 | 0.893 |
| FDR-significant steering configurations | 3/28 | 2/28 | 0/28 | 0/28 |

*Values summarize comparable per-feature/per-layer or per-module results; they are descriptive averages, not a pooled statistical test.*

**Representations.** Linear centered kernel alignment (CKA) was high for feature versus control activations, particularly at earlier layers. Welfare-prompt CKA declined from early to late layers (approximately 0.99 to 0.91–0.97, depending on architecture). Normalized activation displacement and general-domain CKA revealed additional differences. Legacy SVCCA was fitted and evaluated in-sample, so it is not used as independent corroboration.

**Probes.** Logistic classifiers readily decoded linguistic condition and welfare-concept presence from held-out activations; predicting the model's judged welfare-support score by held-out ridge regression was much weaker, with some individual fits below zero R². Lexical TF-IDF baselines were also strong (welfare-concept AUC approximately 0.974); language-changing contrasts can be separable by length or surface vocabulary. Accordingly, high probe accuracy does not establish an abstract representation or its causal use.

**LoRA geometry.** Final feature updates differed materially from control updates; their average weighted update cosines were approximately 0.52–0.56. Midpoint-to-final update directions were comparatively stable (cosine approximately 0.97–0.98). Functional activation contributions generally showed larger RMS differences on welfare prompts than on general prompts, but magnitude and geometry are not behavioral effect sizes.

**Activation steering.** We screened 56 configurations per family, selected 28 on validation responses, and evaluated each on 100 separately held-out steering scenarios using matched steered/unsteered generations and two independent judges. The table lists **all configurations significant at q < 0.05**, correcting within each family's 28 tests:

| Model | Feature-adapter direction | Mean score change | 95% CI | q | Paired n |
|---|---|---:|---|---:|---:|
| Gemma 2 | Conditionality | −0.466 | [−0.610, −0.323] | 0.007 | 74 |
| Gemma 2 | Epistemic hedging | −0.245 | [−0.383, −0.119] | 0.007 | 100 |
| Gemma 2 | Voice | −0.150 | [−0.248, −0.065] | 0.014 | 100 |
| Llama 3.1 | Subordination | −0.200 | [−0.294, −0.115] | 0.014 | 100 |
| Llama 3.1 | Fusion | −0.215 | [−0.341, −0.087] | 0.014 | 100 |

All five effects were negative on the 0–4 target-value-support scale. None survived correction in Mistral or Qwen 3. Thus, steering demonstrates **configuration-specific intervention effects**, not a general cross-model mechanism explaining fine-tuning.

## Figures

![Layerwise representation similarity](../cross-model-analysis/figures/mechanistic/fig12_cka_depth.png)

*Welfare-prompt feature/control CKA by depth.* [SVG](../cross-model-analysis/figures/mechanistic/fig12_cka_depth.svg) · [Data](../cross-model-analysis/figures/mechanistic/fig12_cka_depth.csv)

![Held-out steering effects](../cross-model-analysis/figures/mechanistic/fig18_steering_judged.png)

*Cross-model steering summaries; significance refers to the original model-specific tests, not averages over directions.* [SVG](../cross-model-analysis/figures/mechanistic/fig18_steering_judged.svg) · [Data](../cross-model-analysis/figures/mechanistic/fig18_steering_judged.csv)

Additional figures: [linguistic probes](../cross-model-analysis/figures/mechanistic/fig14_linguistic_probes.svg), [behavior probes](../cross-model-analysis/figures/mechanistic/fig15_behavior_probes.svg), and [LoRA geometry](../cross-model-analysis/figures/mechanistic/fig16_lora_geometry.svg).

## Interpretation and files

Accepted outputs are under `final/<model>/`; `v0/` and `v1/` retain historical runs using superseded methods. The activation-steering automated quality gates passed, but manual review of generated text is still required, especially for Gemma 2, where steered generations reached the output cap more often than baselines. Probe, geometry, and cross-stage associations remain descriptive, and no common fine-tuning mediator has been established. See the [methodology](../../mechanistic-interpretability/README.md) and [cross-model report](../cross-model-analysis/research_results.md).
