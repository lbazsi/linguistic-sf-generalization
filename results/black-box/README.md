# Black-Box Evaluation Results

## Evaluation design

We evaluated Gemma 2 9B, Llama 3.1 8B, Mistral 7B, and Qwen 3 8B under 17 conditions per model: the pretrained base, English canonical and Mandarin control adapters, and 14 linguistic-feature adapters. Each condition was tested on the same 800 animal-welfare scenarios: 100 per generalization category, with half containing explicit tradeoffs. The categories vary whether the animal, value, and context are held out from training.

Two independent automated judges scored each response. The primary outcome, `target_value_support`, measures support for animal welfare from 0 to 4; behavioral commitment, tradeoff priority, and response quality provide secondary checks. We compare paired scenario scores using the **feature adapter minus its registered fine-tuning control**: English canonical for 11 within-English features, Mandarin control for the three language-changing features. The pretrained-base comparisons address a separate question.

Reported inferential results use semantic-cell clustered bootstrap confidence intervals, cluster sign-flip tests, and Benjamini–Hochberg false-discovery-rate (FDR) correction. Negative feature effects mean *less support than the matched control*, not necessarily opposition to animal welfare.

## Main results

| Model | English canonical − base | Mandarin control − base | FDR-significant feature effects |
|---|---:|---:|---:|
| Gemma 2 | +0.906 | +0.773 | 6/14 |
| Llama 3.1 | +1.088 | +0.953 | 7/14 |
| Mistral | +0.786 | +0.558 | 7/14 |
| Qwen 3 | +0.688 | +0.620 | 6/14 |

Both fine-tuning controls improved welfare support over the pretrained base in every model. Canonical gains were positive in **all eight generalization categories**, including the category holding out animal, value, and context simultaneously.

Six feature conditions had **negative, FDR-significant overall effects in all four models**:

| Feature | Gemma 2 | Llama 3.1 | Mistral | Qwen 3 |
|---|---:|---:|---:|---:|
| Constituent order † | −0.196 | −0.254 | −0.410 | −0.180 |
| Epistemic hedging | −0.139 | −0.049 | −0.078 | −0.066 |
| Fusion † | −0.116 | −0.161 | −0.104 | −0.236 |
| Inflectional synthesis † | −0.064 | −0.267 | −0.151 | −0.154 |
| Narrative framing | −0.052 | −0.117 | −0.070 | −0.058 |
| Nominalization | −0.054 | −0.106 | −0.071 | −0.054 |

† These three contrasts change training language as well as the targeted linguistic structure. Their effect cannot be attributed to the grammatical difference alone. Estimates are mean points on the 0–4 scale; significance is assessed using the original per-model FDR-adjusted tests. The [full 14-feature table](../cross-model-analysis/research_results.md#behavioral-replication) includes smaller and non-significant results.

## Robustness and heterogeneity

Effects were less uniform when evaluated by generalization category or tradeoff status than when averaged over all scenarios. Only constituent order in the non-held-out category met the category-specific FDR criterion in all four families; no tradeoff interaction did so across all four. Secondary behavioral-commitment measurements corroborated several primary findings, while coherence and relevance could also shift (notably for conditionality and voice). Complete-case and high-coherence subsets generally preserved the leading effects but sometimes reduced their magnitudes.

Exact agreement between judges on primary support scores was approximately 73–77%, and quadratic-weighted agreement was approximately 0.69–0.76. Automated judging and response-quality variation are therefore important measurement limitations.

## Figures

![Control-adapter gains relative to pretrained models](../cross-model-analysis/figures/behavioral/fig04_baseline_gains.png)

*Gains from English canonical and Mandarin controls, by model and generalization category.* [SVG](../cross-model-analysis/figures/behavioral/fig04_baseline_gains.svg) · [Data](../cross-model-analysis/figures/behavioral/fig04_baseline_gains.csv)

![Four-model feature-effect comparisons](../cross-model-analysis/figures/behavioral/fig01_feature_effects_forest.png)

*Feature-minus-control effects and original clustered confidence intervals.* [SVG](../cross-model-analysis/figures/behavioral/fig01_feature_effects_forest.svg) · [Data](../cross-model-analysis/figures/behavioral/fig01_feature_effects_forest.csv)

## Sources and limitations

Each `<model>/evaluation_results/` directory contains the model-specific reports, aggregate tables, responses, and judgments. The [cross-model report](../cross-model-analysis/research_results.md) and [evaluation methodology](../../evaluation/README.md) document further comparisons. We used one fine-tuning seed per condition and a shared synthetic evaluation framework; these results do not establish human-evaluated performance or causal mediation by linguistic structure.
