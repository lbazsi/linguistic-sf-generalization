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
| genericity             | 800 |          432 |  0.013 |   -0.02  |     0.047 |     0.468 |     0.546 |
| tense_aspect           | 800 |          432 | -0.01  |   -0.043 |     0.023 |     0.581 |     0.581 |
| formality              | 800 |          432 | -0.012 |   -0.05  |     0.024 |     0.561 |     0.581 |
| subordination          | 799 |          432 | -0.022 |   -0.063 |     0.019 |     0.315 |     0.401 |
| voice                  | 800 |          432 | -0.024 |   -0.063 |     0.015 |     0.261 |     0.365 |
| modal_strength         | 800 |          432 | -0.03  |   -0.072 |     0.011 |     0.166 |     0.258 |
| pronoun_distribution   | 800 |          432 | -0.038 |   -0.077 |     0.001 |     0.059 |     0.102 |
| conditionality         | 800 |          432 | -0.038 |   -0.075 |    -0.001 |     0.05  |     0.101 |
| narrative_framing      | 800 |          432 | -0.052 |   -0.095 |    -0.009 |     0.02  |     0.046 |
| nominalization         | 800 |          432 | -0.054 |   -0.097 |    -0.014 |     0.011 |     0.032 |
| inflectional_synthesis | 800 |          432 | -0.064 |   -0.108 |    -0.019 |     0.005 |     0.019 |
| fusion                 | 800 |          432 | -0.116 |   -0.165 |    -0.068 |     0     |     0     |
| epistemic_hedging      | 800 |          432 | -0.139 |   -0.179 |    -0.1   |     0     |     0     |
| constituent_order      | 798 |          432 | -0.196 |   -0.247 |    -0.147 |     0     |     0     |

FDR-significant overall feature effects at q<0.05: **6 / 14**.

## Generalization-category heterogeneity

Largest absolute feature-by-category effects:

| feature                | category             |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-------:|---------:|----------:|----------:|----------:|
| fusion                 | animal_context       | -0.255 |   -0.434 |    -0.099 |     0.004 |     0.065 |
| constituent_order      | animal_value_context | -0.245 |   -0.443 |    -0.076 |     0.014 |     0.172 |
| constituent_order      | context              | -0.24  |   -0.363 |    -0.13  |     0     |     0.003 |
| constituent_order      | value                | -0.21  |   -0.38  |    -0.055 |     0.017 |     0.175 |
| constituent_order      | id                   | -0.21  |   -0.3   |    -0.125 |     0     |     0.003 |
| constituent_order      | animal_context       | -0.2   |   -0.306 |    -0.1   |     0     |     0.011 |
| constituent_order      | animal_value         | -0.195 |   -0.369 |    -0.025 |     0.039 |     0.24  |
| epistemic_hedging      | value                | -0.185 |   -0.335 |    -0.045 |     0.018 |     0.175 |
| epistemic_hedging      | animal_context       | -0.18  |   -0.278 |    -0.091 |     0     |     0.011 |
| fusion                 | value                | -0.175 |   -0.335 |    -0.015 |     0.045 |     0.245 |
| inflectional_synthesis | animal_value         | -0.16  |   -0.296 |    -0.02  |     0.035 |     0.235 |
| fusion                 | context              | -0.155 |   -0.298 |    -0.024 |     0.041 |     0.24  |
| epistemic_hedging      | animal_value_context | -0.155 |   -0.306 |     0     |     0.079 |     0.367 |
| epistemic_hedging      | value_context        | -0.155 |   -0.263 |    -0.051 |     0.012 |     0.172 |
| subordination          | animal_value_context | -0.152 |   -0.337 |     0.01  |     0.136 |     0.407 |
| inflectional_synthesis | value                | -0.15  |   -0.32  |     0.025 |     0.115 |     0.384 |
| fusion                 | animal_value         | -0.145 |   -0.281 |    -0.026 |     0.034 |     0.235 |
| epistemic_hedging      | context              | -0.145 |   -0.23  |    -0.069 |     0.001 |     0.016 |
| constituent_order      | animal               | -0.135 |   -0.247 |    -0.029 |     0.019 |     0.175 |
| constituent_order      | value_context        | -0.135 |   -0.3   |     0.027 |     0.149 |     0.407 |

FDR-significant feature-by-category effects at q<0.05: **6 / 112**.

## Tradeoff sensitivity

Largest estimated tradeoff interactions (tradeoff minus non-tradeoff feature effect):

| feature                | category             |   tradeoff_minus_nontradeoff |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-----------------------------:|---------:|----------:|----------:|----------:|
| fusion                 | value                |                        -0.43 |   -0.771 |    -0.089 |     0.014 |     0.414 |
| constituent_order      | value                |                        -0.34 |   -0.703 |     0.023 |     0.066 |     0.743 |
| narrative_framing      | animal_context       |                        -0.31 |   -0.654 |     0.034 |     0.078 |     0.743 |
| fusion                 | animal_value_context |                        -0.29 |   -0.523 |    -0.057 |     0.015 |     0.414 |
| fusion                 | context              |                         0.27 |    0.015 |     0.525 |     0.038 |     0.705 |
| fusion                 | animal_value         |                        -0.25 |   -0.499 |    -0.001 |     0.049 |     0.743 |
| inflectional_synthesis | animal_context       |                         0.25 |   -0.07  |     0.57  |     0.126 |     0.743 |
| conditionality         | value                |                        -0.24 |   -0.523 |     0.043 |     0.097 |     0.743 |
| nominalization         | value_context        |                        -0.23 |   -0.534 |     0.074 |     0.138 |     0.772 |
| pronoun_distribution   | animal_value_context |                         0.21 |   -0.124 |     0.544 |     0.217 |     0.821 |
| inflectional_synthesis | id                   |                         0.21 |    0.043 |     0.377 |     0.014 |     0.414 |
| voice                  | animal_value_context |                         0.18 |   -0.106 |     0.466 |     0.217 |     0.821 |
| fusion                 | animal_context       |                        -0.17 |   -0.524 |     0.184 |     0.347 |     0.821 |
| modal_strength         | value_context        |                         0.17 |    0.026 |     0.314 |     0.02  |     0.454 |
| fusion                 | value_context        |                        -0.17 |   -0.462 |     0.122 |     0.253 |     0.821 |

## Canonical fine-tuning effect

| category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| animal               |  99 |  0.859 |    0.649 |     1.08  |         0 |         0 |
| animal_context       |  97 |  0.959 |    0.72  |     1.204 |         0 |         0 |
| animal_value         |  98 |  0.872 |    0.651 |     1.097 |         0 |         0 |
| animal_value_context |  98 |  0.709 |    0.464 |     0.943 |         0 |         0 |
| context              |  99 |  0.909 |    0.699 |     1.133 |         0 |         0 |
| id                   | 100 |  0.895 |    0.715 |     1.09  |         0 |         0 |
| value                | 100 |  1.15  |    0.92  |     1.385 |         0 |         0 |
| value_context        |  99 |  0.894 |    0.629 |     1.173 |         0 |         0 |

## Source-language control versus base

| condition   | category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:------------|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| control_zh  | animal               |  99 |  0.788 |    0.572 |     1.014 |     0     |     0     |
| control_zh  | animal_context       |  97 |  0.809 |    0.565 |     1.061 |     0     |     0     |
| control_zh  | animal_value         |  98 |  0.77  |    0.56  |     0.99  |     0     |     0     |
| control_zh  | animal_value_context |  98 |  0.536 |    0.253 |     0.818 |     0.001 |     0.001 |
| control_zh  | context              |  99 |  0.727 |    0.535 |     0.928 |     0     |     0     |
| control_zh  | id                   | 100 |  0.765 |    0.575 |     0.97  |     0     |     0     |
| control_zh  | value                | 100 |  1.01  |    0.78  |     1.25  |     0     |     0     |
| control_zh  | value_context        |  99 |  0.778 |    0.586 |     0.974 |     0     |     0     |

## Judge reliability

| condition   | metric                |     n |   exact_agreement | mean_absolute_difference   |   quadratic_weighted_kappa |
|:------------|:----------------------|------:|------------------:|:---------------------------|---------------------------:|
| ALL         | target_value_support  | 13558 |             0.734 | 0.288                      |                      0.693 |
| ALL         | behavioral_commitment | 13558 |             0.738 | 0.283                      |                      0.704 |
| ALL         | tradeoff_priority     |  6781 |             0.725 | 0.304                      |                      0.783 |
| ALL         | coherence_relevance   | 13600 |             0.95  | 0.050                      |                      0.498 |
| ALL         | outcome               | 13600 |             0.957 | NA                         |                      0.65  |

## Robustness checks

| feature                |   primary_mean |   primary_n |   complete_case_mean |   complete_case_n |   high_coherence_mean |   high_coherence_n |   complete_case_minus_primary |   high_coherence_minus_primary |
|:-----------------------|---------------:|------------:|---------------------:|------------------:|----------------------:|-------------------:|------------------------------:|-------------------------------:|
| conditionality         |         -0.038 |         800 |               -0.038 |               800 |                -0.031 |                777 |                         0     |                          0.007 |
| constituent_order      |         -0.196 |         798 |               -0.194 |               796 |                -0.201 |                782 |                         0.002 |                         -0.005 |
| epistemic_hedging      |         -0.139 |         800 |               -0.139 |               800 |                -0.142 |                784 |                         0     |                         -0.003 |
| formality              |         -0.012 |         800 |               -0.012 |               800 |                -0.021 |                786 |                         0     |                         -0.009 |
| fusion                 |         -0.116 |         800 |               -0.112 |               798 |                -0.112 |                787 |                         0.004 |                          0.004 |
| genericity             |          0.013 |         800 |                0.013 |               800 |                 0.004 |                791 |                         0     |                         -0.009 |
| inflectional_synthesis |         -0.064 |         800 |               -0.062 |               799 |                -0.061 |                787 |                         0.002 |                          0.003 |
| modal_strength         |         -0.03  |         800 |               -0.03  |               800 |                -0.028 |                790 |                         0     |                          0.002 |
| narrative_framing      |         -0.052 |         800 |               -0.052 |               800 |                -0.062 |                790 |                         0     |                         -0.01  |
| nominalization         |         -0.054 |         800 |               -0.051 |               798 |                -0.046 |                782 |                         0.003 |                          0.008 |
| pronoun_distribution   |         -0.038 |         800 |               -0.038 |               800 |                -0.028 |                784 |                         0     |                          0.01  |
| subordination          |         -0.022 |         799 |               -0.022 |               799 |                -0.02  |                784 |                         0     |                          0.002 |
| tense_aspect           |         -0.01  |         800 |               -0.01  |               800 |                -0.015 |                788 |                         0     |                         -0.005 |
| voice                  |         -0.024 |         800 |               -0.024 |               800 |                -0.03  |                785 |                         0     |                         -0.006 |

Correlation of primary and two-judge complete-case overall effects: **1.000**.

## Lexical-diversity association

Within the primary same-language feature set (n=11), Pearson r = **0.157** (permutation p=0.624) and Spearman rho = **0.396** (permutation p=0.228).

Leave-one-feature-out ranges: Pearson [0.031, 0.629], Spearman [0.195, 0.571].

## Interpretation guardrails

- The statistical unit for uncertainty is the semantic cell rather than the raw textual realization.
- Feature effects are paired against their registered controls; cross-linguistic manipulations should not be interpreted as isolating a single linguistic property when language identity changes with the intervention.
- Model-judge scores are measurement instruments rather than human ground truth. Judge agreement and complete-case sensitivity should be inspected alongside effect estimates.
- Category-level and tradeoff analyses involve many comparisons. Report q-values and confidence intervals rather than selecting results by unadjusted p-values.
- The benchmark consists of generated scenarios. Conclusions are about this benchmark distribution and should be replicated on independently generated or human-authored evaluations before strong external-validity claims.
