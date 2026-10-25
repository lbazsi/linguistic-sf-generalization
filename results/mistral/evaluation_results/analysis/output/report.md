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
| tense_aspect           | 800 |          432 |  0.048 |    0.01  |     0.086 |     0.015 |     0.029 |
| genericity             | 800 |          432 |  0.011 |   -0.029 |     0.051 |     0.625 |     0.875 |
| modal_strength         | 800 |          432 |  0.002 |   -0.036 |     0.04  |     0.949 |     1     |
| formality              | 800 |          432 |  0.001 |   -0.041 |     0.041 |     1     |     1     |
| pronoun_distribution   | 800 |          432 |  0.001 |   -0.042 |     0.042 |     1     |     1     |
| subordination          | 800 |          432 | -0.001 |   -0.044 |     0.043 |     1     |     1     |
| conditionality         | 800 |          432 | -0.022 |   -0.067 |     0.021 |     0.323 |     0.502 |
| voice                  | 800 |          432 | -0.036 |   -0.08  |     0.007 |     0.113 |     0.198 |
| narrative_framing      | 800 |          432 | -0.07  |   -0.122 |    -0.018 |     0.01  |     0.023 |
| nominalization         | 800 |          432 | -0.071 |   -0.12  |    -0.023 |     0.006 |     0.017 |
| epistemic_hedging      | 800 |          432 | -0.078 |   -0.117 |    -0.04  |     0     |     0     |
| fusion                 | 800 |          432 | -0.104 |   -0.16  |    -0.05  |     0     |     0.001 |
| inflectional_synthesis | 800 |          432 | -0.151 |   -0.21  |    -0.095 |     0     |     0     |
| constituent_order      | 800 |          432 | -0.41  |   -0.476 |    -0.344 |     0     |     0     |

FDR-significant overall feature effects at q<0.05: **7 / 14**.

## Generalization-category heterogeneity

Largest absolute feature-by-category effects:

| feature                | category             |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-------:|---------:|----------:|----------:|----------:|
| constituent_order      | animal_context       | -0.515 |   -0.702 |    -0.333 |     0     |     0.001 |
| constituent_order      | animal_value         | -0.49  |   -0.689 |    -0.292 |     0     |     0.001 |
| constituent_order      | value                | -0.48  |   -0.715 |    -0.26  |     0     |     0.004 |
| constituent_order      | animal               | -0.445 |   -0.609 |    -0.296 |     0     |     0.001 |
| constituent_order      | value_context        | -0.425 |   -0.668 |    -0.214 |     0     |     0.004 |
| nominalization         | value_context        | -0.33  |   -0.485 |    -0.173 |     0.001 |     0.009 |
| constituent_order      | id                   | -0.32  |   -0.465 |    -0.19  |     0     |     0.001 |
| constituent_order      | context              | -0.315 |   -0.441 |    -0.194 |     0     |     0.001 |
| constituent_order      | animal_value_context | -0.29  |   -0.53  |    -0.083 |     0.016 |     0.119 |
| inflectional_synthesis | animal_value         | -0.235 |   -0.395 |    -0.08  |     0.007 |     0.076 |
| inflectional_synthesis | animal_context       | -0.21  |   -0.358 |    -0.071 |     0.008 |     0.079 |
| narrative_framing      | value_context        | -0.2   |   -0.396 |     0     |     0.072 |     0.325 |
| epistemic_hedging      | animal_value         | -0.19  |   -0.313 |    -0.078 |     0.003 |     0.035 |
| inflectional_synthesis | animal               | -0.19  |   -0.356 |    -0.043 |     0.022 |     0.143 |
| tense_aspect           | animal_value_context |  0.19  |    0.069 |     0.321 |     0.011 |     0.092 |
| fusion                 | animal_value_context | -0.175 |   -0.345 |    -0.02  |     0.053 |     0.284 |
| inflectional_synthesis | animal_value_context | -0.17  |   -0.342 |    -0.015 |     0.06  |     0.291 |
| inflectional_synthesis | value                | -0.17  |   -0.36  |     0.02  |     0.104 |     0.418 |
| narrative_framing      | context              | -0.165 |   -0.342 |     0.01  |     0.079 |     0.34  |
| pronoun_distribution   | animal_context       | -0.145 |   -0.328 |     0.015 |     0.123 |     0.454 |

FDR-significant feature-by-category effects at q<0.05: **9 / 112**.

## Tradeoff sensitivity

Largest estimated tradeoff interactions (tradeoff minus non-tradeoff feature effect):

| feature                | category             |   tradeoff_minus_nontradeoff |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-----------------------------:|---------:|----------:|----------:|----------:|
| narrative_framing      | value_context        |                        -0.62 |   -0.951 |    -0.289 |     0     |     0.027 |
| narrative_framing      | context              |                        -0.53 |   -0.851 |    -0.209 |     0.001 |     0.069 |
| constituent_order      | value_context        |                        -0.43 |   -0.834 |    -0.026 |     0.037 |     0.406 |
| inflectional_synthesis | context              |                        -0.4  |   -0.7   |    -0.1   |     0.009 |     0.167 |
| constituent_order      | animal               |                        -0.39 |   -0.674 |    -0.106 |     0.007 |     0.158 |
| constituent_order      | animal_value_context |                        -0.38 |   -0.841 |     0.081 |     0.106 |     0.582 |
| constituent_order      | value                |                        -0.36 |   -0.814 |     0.094 |     0.12  |     0.584 |
| inflectional_synthesis | animal_value         |                        -0.33 |   -0.712 |     0.052 |     0.09  |     0.582 |
| voice                  | animal_value_context |                        -0.33 |   -0.688 |     0.028 |     0.07  |     0.582 |
| fusion                 | context              |                        -0.33 |   -0.691 |     0.031 |     0.073 |     0.582 |
| constituent_order      | animal_value         |                        -0.32 |   -0.619 |    -0.021 |     0.036 |     0.406 |
| genericity             | value_context        |                        -0.32 |   -0.553 |    -0.087 |     0.007 |     0.158 |
| constituent_order      | animal_context       |                        -0.31 |   -0.698 |     0.078 |     0.118 |     0.584 |
| constituent_order      | context              |                        -0.31 |   -0.552 |    -0.068 |     0.012 |     0.167 |
| conditionality         | context              |                        -0.3  |   -0.514 |    -0.086 |     0.006 |     0.158 |

## Canonical fine-tuning effect

| category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| animal               | 100 |  0.825 |    0.618 |     1.04  |         0 |         0 |
| animal_context       | 100 |  0.645 |    0.47  |     0.838 |         0 |         0 |
| animal_value         |  99 |  0.869 |    0.624 |     1.135 |         0 |         0 |
| animal_value_context |  99 |  0.576 |    0.429 |     0.732 |         0 |         0 |
| context              | 100 |  0.955 |    0.722 |     1.2   |         0 |         0 |
| id                   | 100 |  0.82  |    0.615 |     1.04  |         0 |         0 |
| value                | 100 |  0.865 |    0.625 |     1.115 |         0 |         0 |
| value_context        | 100 |  0.735 |    0.5   |     0.976 |         0 |         0 |

## Source-language control versus base

| condition   | category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:------------|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| control_zh  | animal               | 100 |  0.62  |    0.432 |     0.821 |     0     |     0     |
| control_zh  | animal_context       | 100 |  0.39  |    0.206 |     0.586 |     0     |     0     |
| control_zh  | animal_value         |  99 |  0.626 |    0.383 |     0.892 |     0     |     0     |
| control_zh  | animal_value_context |  99 |  0.439 |    0.216 |     0.658 |     0.001 |     0.001 |
| control_zh  | context              | 100 |  0.665 |    0.433 |     0.917 |     0     |     0     |
| control_zh  | id                   | 100 |  0.635 |    0.415 |     0.86  |     0     |     0     |
| control_zh  | value                | 100 |  0.695 |    0.45  |     0.935 |     0     |     0     |
| control_zh  | value_context        | 100 |  0.395 |    0.176 |     0.602 |     0.003 |     0.003 |

## Judge reliability

| condition   | metric                |     n |   exact_agreement | mean_absolute_difference   |   quadratic_weighted_kappa |
|:------------|:----------------------|------:|------------------:|:---------------------------|---------------------------:|
| ALL         | target_value_support  | 13584 |             0.73  | 0.299                      |                      0.739 |
| ALL         | behavioral_commitment | 13584 |             0.726 | 0.304                      |                      0.749 |
| ALL         | tradeoff_priority     |  6795 |             0.699 | 0.341                      |                      0.809 |
| ALL         | coherence_relevance   | 13600 |             0.866 | 0.134                      |                      0.518 |
| ALL         | outcome               | 13600 |             0.949 | NA                         |                      0.669 |

## Robustness checks

| feature                |   primary_mean |   primary_n |   complete_case_mean |   complete_case_n |   high_coherence_mean |   high_coherence_n |   complete_case_minus_primary |   high_coherence_minus_primary |
|:-----------------------|---------------:|------------:|---------------------:|------------------:|----------------------:|-------------------:|------------------------------:|-------------------------------:|
| conditionality         |         -0.022 |         800 |               -0.025 |               799 |                -0.015 |                624 |                        -0.003 |                          0.007 |
| constituent_order      |         -0.41  |         800 |               -0.407 |               798 |                -0.346 |                674 |                         0.003 |                          0.064 |
| epistemic_hedging      |         -0.078 |         800 |               -0.077 |               797 |                -0.073 |                714 |                         0.002 |                          0.005 |
| formality              |          0.001 |         800 |                0.001 |               799 |                 0.015 |                586 |                         0     |                          0.014 |
| fusion                 |         -0.104 |         800 |               -0.108 |               798 |                -0.099 |                710 |                        -0.003 |                          0.006 |
| genericity             |          0.011 |         800 |                0.011 |               799 |                 0.015 |                716 |                         0     |                          0.004 |
| inflectional_synthesis |         -0.151 |         800 |               -0.155 |               799 |                -0.161 |                700 |                        -0.003 |                         -0.01  |
| modal_strength         |          0.002 |         800 |               -0.001 |               799 |                 0.023 |                659 |                        -0.003 |                          0.021 |
| narrative_framing      |         -0.07  |         800 |               -0.07  |               798 |                -0.046 |                707 |                        -0     |                          0.024 |
| nominalization         |         -0.071 |         800 |               -0.073 |               799 |                -0.048 |                696 |                        -0.002 |                          0.023 |
| pronoun_distribution   |          0.001 |         800 |                0.001 |               798 |                 0.013 |                656 |                         0.001 |                          0.012 |
| subordination          |         -0.001 |         800 |               -0.003 |               799 |                 0.01  |                698 |                        -0.003 |                          0.011 |
| tense_aspect           |          0.048 |         800 |                0.046 |               799 |                 0.043 |                702 |                        -0.002 |                         -0.005 |
| voice                  |         -0.036 |         800 |               -0.039 |               799 |                -0.03  |                643 |                        -0.003 |                          0.006 |

Correlation of primary and two-judge complete-case overall effects: **1.000**.

## Lexical-diversity association

Within the primary same-language feature set (n=11), Pearson r = **0.542** (permutation p=0.086) and Spearman rho = **0.624** (permutation p=0.044).

Leave-one-feature-out ranges: Pearson [0.376, 0.726], Spearman [0.498, 0.839].

## Interpretation guardrails

- The statistical unit for uncertainty is the semantic cell rather than the raw textual realization.
- Feature effects are paired against their registered controls; cross-linguistic manipulations should not be interpreted as isolating a single linguistic property when language identity changes with the intervention.
- Model-judge scores are measurement instruments rather than human ground truth. Judge agreement and complete-case sensitivity should be inspected alongside effect estimates.
- Category-level and tradeoff analyses involve many comparisons. Report q-values and confidence intervals rather than selecting results by unadjusted p-values.
- The benchmark consists of generated scenarios. Conclusions are about this benchmark distribution and should be replicated on independently generated or human-authored evaluations before strong external-validity claims.
