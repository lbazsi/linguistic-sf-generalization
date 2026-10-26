# Black-box evaluation analysis report

## Scope and inferential strategy

The primary behavioral outcome is the paired difference in `target_value_support` between each linguistic-feature model and its registered comparison control. Effects are reported on the original 0–4 judge scale. Confidence intervals use a semantic-cell cluster bootstrap, and two-sided p-values use a semantic-cell cluster sign-flip test. The semantic cell is `category × animal × value × context`, which prevents repeated textual realizations of the same semantic assignment from being treated as fully independent.

The analysis used 10,000 bootstrap replicates and 20,000 sign-flip/permutation replicates with alpha=0.050. Benjamini-Hochberg FDR correction is applied separately to overall feature effects, feature-by-category effects, tradeoff-stratified effects, tradeoff interactions, canonical-minus-base category effects, and control-minus-base category effects.

## Data integrity

- Scenarios: **800**
- Model conditions: **17**
- Combined condition × scenario rows: **13600 / 13600** expected
- Judge 1 pairs: **13600**
- Judge 2 pairs: **13600**
- Paired deltas: **12800**
- Feature-minus-control deltas: **11200**
- Canonical-minus-base deltas: **800**
- Control-minus-base deltas: **800**

## Overall linguistic-feature effects

| feature                |   n |   n_clusters |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|----:|-------------:|-------:|---------:|----------:|----------:|----------:|
| formality              | 800 |          432 | -0.003 |   -0.042 |     0.038 |     0.927 |     0.927 |
| tense_aspect           | 800 |          432 | -0.003 |   -0.04  |     0.031 |     0.891 |     0.927 |
| modal_strength         | 800 |          432 | -0.007 |   -0.045 |     0.031 |     0.75  |     0.875 |
| pronoun_distribution   | 800 |          432 | -0.014 |   -0.051 |     0.023 |     0.501 |     0.637 |
| genericity             | 800 |          432 | -0.021 |   -0.061 |     0.016 |     0.302 |     0.423 |
| voice                  | 800 |          432 | -0.031 |   -0.073 |     0.01  |     0.167 |     0.259 |
| conditionality         | 800 |          432 | -0.047 |   -0.091 |    -0.004 |     0.033 |     0.058 |
| epistemic_hedging      | 800 |          432 | -0.049 |   -0.088 |    -0.011 |     0.015 |     0.029 |
| subordination          | 800 |          432 | -0.063 |   -0.102 |    -0.026 |     0.001 |     0.002 |
| nominalization         | 800 |          432 | -0.106 |   -0.155 |    -0.058 |     0     |     0     |
| narrative_framing      | 800 |          432 | -0.117 |   -0.173 |    -0.062 |     0     |     0     |
| fusion                 | 800 |          432 | -0.161 |   -0.218 |    -0.105 |     0     |     0     |
| constituent_order      | 800 |          432 | -0.254 |   -0.316 |    -0.194 |     0     |     0     |
| inflectional_synthesis | 800 |          432 | -0.267 |   -0.329 |    -0.207 |     0     |     0     |

FDR-significant overall feature effects at q<0.05: **7 / 14**.

## Generalization-category heterogeneity

Largest absolute feature-by-category effects:

| feature                | category             |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-------:|---------:|----------:|----------:|----------:|
| inflectional_synthesis | animal_value         | -0.365 |   -0.544 |    -0.196 |     0     |     0.006 |
| inflectional_synthesis | animal               | -0.365 |   -0.54  |    -0.214 |     0     |     0.001 |
| constituent_order      | animal_context       | -0.365 |   -0.541 |    -0.201 |     0     |     0.001 |
| inflectional_synthesis | value_context        | -0.32  |   -0.519 |    -0.139 |     0.004 |     0.028 |
| constituent_order      | animal               | -0.31  |   -0.448 |    -0.195 |     0     |     0.001 |
| constituent_order      | value                | -0.31  |   -0.54  |    -0.1   |     0.008 |     0.056 |
| constituent_order      | context              | -0.275 |   -0.398 |    -0.158 |     0     |     0.001 |
| inflectional_synthesis | id                   | -0.255 |   -0.41  |    -0.115 |     0.001 |     0.008 |
| inflectional_synthesis | animal_context       | -0.255 |   -0.424 |    -0.094 |     0.005 |     0.041 |
| fusion                 | animal               | -0.25  |   -0.41  |    -0.11  |     0.001 |     0.008 |
| fusion                 | animal_context       | -0.25  |   -0.41  |    -0.104 |     0.003 |     0.025 |
| narrative_framing      | context              | -0.24  |   -0.404 |    -0.094 |     0.003 |     0.028 |
| constituent_order      | animal_value         | -0.24  |   -0.391 |    -0.104 |     0.001 |     0.011 |
| constituent_order      | id                   | -0.235 |   -0.34  |    -0.145 |     0     |     0.001 |
| inflectional_synthesis | value                | -0.235 |   -0.415 |    -0.06  |     0.014 |     0.085 |
| inflectional_synthesis | animal_value_context | -0.235 |   -0.418 |    -0.07  |     0.015 |     0.087 |
| fusion                 | animal_value         | -0.215 |   -0.396 |    -0.045 |     0.028 |     0.134 |
| narrative_framing      | value                | -0.195 |   -0.395 |    -0.01  |     0.059 |     0.199 |
| conditionality         | value                | -0.185 |   -0.35  |    -0.03  |     0.03  |     0.134 |
| narrative_framing      | animal_context       | -0.18  |   -0.285 |    -0.087 |     0.001 |     0.008 |

FDR-significant feature-by-category effects at q<0.05: **15 / 112**.

## Tradeoff sensitivity

Largest estimated tradeoff interactions (tradeoff minus non-tradeoff feature effect):

| feature                | category             |   tradeoff_minus_nontradeoff |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-----------------------------:|---------:|----------:|----------:|----------:|
| narrative_framing      | value_context        |                        -0.72 |   -1.113 |    -0.327 |     0     |     0.037 |
| constituent_order      | value                |                        -0.66 |   -1.05  |    -0.27  |     0.001 |     0.052 |
| narrative_framing      | value                |                        -0.59 |   -1.001 |    -0.179 |     0.005 |     0.097 |
| inflectional_synthesis | animal_value_context |                        -0.51 |   -0.907 |    -0.113 |     0.012 |     0.137 |
| fusion                 | animal               |                        -0.46 |   -0.754 |    -0.166 |     0.002 |     0.08  |
| fusion                 | animal_context       |                        -0.46 |   -0.834 |    -0.086 |     0.016 |     0.145 |
| inflectional_synthesis | animal               |                        -0.45 |   -0.751 |    -0.149 |     0.003 |     0.095 |
| constituent_order      | value_context        |                        -0.45 |   -0.983 |     0.083 |     0.098 |     0.399 |
| narrative_framing      | animal_value_context |                        -0.44 |   -0.895 |     0.015 |     0.058 |     0.308 |
| narrative_framing      | context              |                        -0.42 |   -0.733 |    -0.107 |     0.009 |     0.121 |
| constituent_order      | animal_value_context |                        -0.41 |   -0.731 |    -0.089 |     0.012 |     0.137 |
| inflectional_synthesis | animal_context       |                        -0.39 |   -0.707 |    -0.073 |     0.016 |     0.145 |
| fusion                 | animal_value         |                        -0.37 |   -0.704 |    -0.036 |     0.03  |     0.187 |
| inflectional_synthesis | context              |                        -0.37 |   -0.631 |    -0.109 |     0.005 |     0.097 |
| fusion                 | animal_value_context |                        -0.37 |   -0.829 |     0.089 |     0.114 |     0.399 |

## Canonical fine-tuning effect

| category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| animal               |  97 |  1.206 |    0.941 |     1.468 |         0 |         0 |
| animal_context       | 100 |  1.405 |    1.152 |     1.665 |         0 |         0 |
| animal_value         |  96 |  1.141 |    0.895 |     1.372 |         0 |         0 |
| animal_value_context | 100 |  0.57  |    0.354 |     0.807 |         0 |         0 |
| context              |  96 |  1.234 |    0.977 |     1.485 |         0 |         0 |
| id                   |  98 |  1.112 |    0.867 |     1.367 |         0 |         0 |
| value                |  99 |  1.187 |    0.919 |     1.46  |         0 |         0 |
| value_context        |  95 |  0.847 |    0.6   |     1.088 |         0 |         0 |

## Source-language control versus base

| condition   | category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:------------|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| control_zh  | animal               |  97 |  1.16  |    0.892 |     1.421 |         0 |         0 |
| control_zh  | animal_context       | 100 |  1.16  |    0.904 |     1.427 |         0 |         0 |
| control_zh  | animal_value         |  96 |  1.078 |    0.854 |     1.298 |         0 |         0 |
| control_zh  | animal_value_context | 100 |  0.515 |    0.296 |     0.73  |         0 |         0 |
| control_zh  | context              |  96 |  0.932 |    0.67  |     1.2   |         0 |         0 |
| control_zh  | id                   |  98 |  1.01  |    0.755 |     1.276 |         0 |         0 |
| control_zh  | value                |  99 |  0.96  |    0.704 |     1.217 |         0 |         0 |
| control_zh  | value_context        |  95 |  0.816 |    0.612 |     1.022 |         0 |         0 |

## Judge reliability

| condition   | metric                |     n |   exact_agreement | mean_absolute_difference   |   quadratic_weighted_kappa |
|:------------|:----------------------|------:|------------------:|:---------------------------|---------------------------:|
| ALL         | target_value_support  | 13524 |             0.753 | 0.277                      |                      0.758 |
| ALL         | behavioral_commitment | 13524 |             0.756 | 0.276                      |                      0.769 |
| ALL         | tradeoff_priority     |  6753 |             0.731 | 0.308                      |                      0.828 |
| ALL         | coherence_relevance   | 13600 |             0.874 | 0.127                      |                      0.449 |
| ALL         | outcome               | 13600 |             0.942 | NA                         |                      0.643 |

## Robustness checks

| feature                |   primary_mean |   primary_n |   complete_case_mean |   complete_case_n |   high_coherence_mean |   high_coherence_n |   complete_case_minus_primary |   high_coherence_minus_primary |
|:-----------------------|---------------:|------------:|---------------------:|------------------:|----------------------:|-------------------:|------------------------------:|-------------------------------:|
| conditionality         |         -0.047 |         800 |               -0.042 |               798 |                -0.047 |                717 |                         0.005 |                          0     |
| constituent_order      |         -0.254 |         800 |               -0.254 |               800 |                -0.243 |                729 |                         0     |                          0.012 |
| epistemic_hedging      |         -0.049 |         800 |               -0.049 |               800 |                -0.054 |                736 |                         0     |                         -0.005 |
| formality              |         -0.003 |         800 |               -0.003 |               800 |                -0.008 |                729 |                         0     |                         -0.006 |
| fusion                 |         -0.161 |         800 |               -0.161 |               800 |                -0.141 |                738 |                         0     |                          0.02  |
| genericity             |         -0.021 |         800 |               -0.019 |               798 |                -0.024 |                735 |                         0.002 |                         -0.003 |
| inflectional_synthesis |         -0.267 |         800 |               -0.26  |               798 |                -0.243 |                747 |                         0.007 |                          0.024 |
| modal_strength         |         -0.007 |         800 |               -0.007 |               800 |                -0.002 |                729 |                         0     |                          0.005 |
| narrative_framing      |         -0.117 |         800 |               -0.117 |               800 |                -0.119 |                720 |                         0     |                         -0.003 |
| nominalization         |         -0.106 |         800 |               -0.106 |               800 |                -0.092 |                723 |                         0     |                          0.014 |
| pronoun_distribution   |         -0.014 |         800 |               -0.013 |               799 |                -0.014 |                712 |                         0.001 |                         -0     |
| subordination          |         -0.063 |         800 |               -0.063 |               800 |                -0.072 |                725 |                         0     |                         -0.009 |
| tense_aspect           |         -0.003 |         800 |               -0.003 |               800 |                 0.01  |                724 |                         0     |                          0.013 |
| voice                  |         -0.031 |         800 |               -0.031 |               800 |                -0.033 |                722 |                         0     |                         -0.002 |

Correlation of primary and two-judge complete-case overall effects: **1.000**.

## Lexical-diversity association

Within the primary same-language feature set (n=11), Pearson r = **0.848** (permutation p=0.003) and Spearman rho = **0.591** (permutation p=0.060).

Leave-one-feature-out ranges: Pearson [0.787, 0.876], Spearman [0.479, 0.794].

## Interpretation guardrails

- The statistical unit for uncertainty is the semantic cell rather than the raw textual realization.
- Feature effects are paired against their registered controls; cross-linguistic manipulations should not be interpreted as isolating a single linguistic property when language identity changes with the intervention.
- Model-judge scores are measurement instruments rather than human ground truth. Judge agreement and complete-case sensitivity should be inspected alongside effect estimates.
- Category-level and tradeoff analyses involve many comparisons. Report q-values and confidence intervals rather than selecting results by unadjusted p-values.
- The benchmark consists of generated scenarios. Conclusions are about this benchmark distribution and should be replicated on independently generated or human-authored evaluations before strong external-validity claims.
