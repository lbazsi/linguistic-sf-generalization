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
| pronoun_distribution   | 800 |          432 |  0.011 |   -0.019 |     0.041 |     0.513 |     0.599 |
| modal_strength         | 800 |          432 |  0.004 |   -0.029 |     0.038 |     0.827 |     0.827 |
| formality              | 800 |          432 | -0.009 |   -0.051 |     0.034 |     0.707 |     0.761 |
| conditionality         | 800 |          432 | -0.016 |   -0.054 |     0.022 |     0.45  |     0.572 |
| genericity             | 800 |          432 | -0.031 |   -0.071 |     0.008 |     0.139 |     0.195 |
| voice                  | 800 |          432 | -0.037 |   -0.081 |     0.007 |     0.107 |     0.167 |
| tense_aspect           | 800 |          432 | -0.037 |   -0.078 |    -0.001 |     0.064 |     0.113 |
| subordination          | 800 |          432 | -0.048 |   -0.091 |    -0.008 |     0.032 |     0.064 |
| nominalization         | 800 |          432 | -0.054 |   -0.098 |    -0.01  |     0.02  |     0.048 |
| narrative_framing      | 799 |          432 | -0.058 |   -0.103 |    -0.015 |     0.01  |     0.028 |
| epistemic_hedging      | 800 |          432 | -0.066 |   -0.103 |    -0.028 |     0.001 |     0.003 |
| inflectional_synthesis | 800 |          432 | -0.154 |   -0.208 |    -0.101 |     0     |     0     |
| constituent_order      | 800 |          432 | -0.18  |   -0.235 |    -0.126 |     0     |     0     |
| fusion                 | 800 |          432 | -0.236 |   -0.289 |    -0.184 |     0     |     0     |

FDR-significant overall feature effects at q<0.05: **6 / 14**.

## Generalization-category heterogeneity

Largest absolute feature-by-category effects:

| feature                | category             |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-------:|---------:|----------:|----------:|----------:|
| fusion                 | animal_context       | -0.36  |   -0.526 |    -0.204 |     0     |     0.003 |
| constituent_order      | context              | -0.335 |   -0.571 |    -0.092 |     0.012 |     0.079 |
| fusion                 | context              | -0.315 |   -0.485 |    -0.157 |     0     |     0.006 |
| inflectional_synthesis | value                | -0.305 |   -0.49  |    -0.13  |     0.001 |     0.007 |
| fusion                 | value                | -0.29  |   -0.46  |    -0.135 |     0.001 |     0.007 |
| fusion                 | animal_value         | -0.275 |   -0.4   |    -0.16  |     0     |     0.002 |
| inflectional_synthesis | animal               | -0.24  |   -0.415 |    -0.083 |     0.005 |     0.045 |
| subordination          | value_context        | -0.23  |   -0.342 |    -0.126 |     0     |     0.004 |
| constituent_order      | animal_value_context | -0.215 |   -0.395 |    -0.056 |     0.025 |     0.128 |
| inflectional_synthesis | animal_value         | -0.21  |   -0.365 |    -0.071 |     0.006 |     0.051 |
| fusion                 | animal               | -0.195 |   -0.33  |    -0.086 |     0     |     0.003 |
| inflectional_synthesis | animal_context       | -0.19  |   -0.33  |    -0.065 |     0.008 |     0.063 |
| inflectional_synthesis | context              | -0.185 |   -0.379 |     0.016 |     0.085 |     0.225 |
| constituent_order      | animal               | -0.185 |   -0.273 |    -0.108 |     0     |     0.002 |
| fusion                 | animal_value_context | -0.18  |   -0.381 |    -0.01  |     0.083 |     0.225 |
| modal_strength         | context              |  0.16  |    0.045 |     0.298 |     0.016 |     0.099 |
| constituent_order      | id                   | -0.16  |   -0.235 |    -0.09  |     0     |     0.002 |
| constituent_order      | animal_context       | -0.16  |   -0.291 |    -0.036 |     0.022 |     0.117 |
| subordination          | animal_value         | -0.15  |   -0.289 |    -0.029 |     0.032 |     0.149 |
| nominalization         | animal_value_context | -0.145 |   -0.282 |    -0.01  |     0.062 |     0.197 |

FDR-significant feature-by-category effects at q<0.05: **12 / 112**.

## Tradeoff sensitivity

Largest estimated tradeoff interactions (tradeoff minus non-tradeoff feature effect):

| feature                | category             |   tradeoff_minus_nontradeoff |   ci_low |   ci_high |   p_value |   q_value |
|:-----------------------|:---------------------|-----------------------------:|---------:|----------:|----------:|----------:|
| constituent_order      | animal_value_context |                       -0.71  |   -1.011 |    -0.409 |     0     |     0     |
| narrative_framing      | animal_value_context |                       -0.543 |   -0.905 |    -0.18  |     0.003 |     0.062 |
| fusion                 | animal_value_context |                       -0.52  |   -1.033 |    -0.007 |     0.047 |     0.219 |
| fusion                 | context              |                       -0.51  |   -0.883 |    -0.137 |     0.007 |     0.083 |
| constituent_order      | value_context        |                       -0.45  |   -0.706 |    -0.194 |     0.001 |     0.031 |
| constituent_order      | context              |                       -0.41  |   -0.904 |     0.084 |     0.104 |     0.297 |
| fusion                 | value                |                       -0.4   |   -0.773 |    -0.027 |     0.036 |     0.191 |
| fusion                 | animal_context       |                       -0.38  |   -0.744 |    -0.016 |     0.041 |     0.208 |
| narrative_framing      | value_context        |                       -0.38  |   -0.626 |    -0.134 |     0.002 |     0.062 |
| inflectional_synthesis | value                |                       -0.37  |   -0.732 |    -0.008 |     0.045 |     0.219 |
| fusion                 | animal               |                       -0.35  |   -0.582 |    -0.118 |     0.003 |     0.062 |
| fusion                 | value_context        |                       -0.35  |   -0.636 |    -0.064 |     0.017 |     0.143 |
| fusion                 | animal_value         |                       -0.33  |   -0.57  |    -0.09  |     0.007 |     0.083 |
| genericity             | animal_value_context |                       -0.33  |   -0.525 |    -0.135 |     0.001 |     0.033 |
| nominalization         | animal_value_context |                       -0.33  |   -0.568 |    -0.092 |     0.007 |     0.083 |

## Canonical fine-tuning effect

| category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| animal               | 100 |  0.9   |    0.678 |     1.138 |         0 |         0 |
| animal_context       | 100 |  0.54  |    0.348 |     0.747 |         0 |         0 |
| animal_value         | 100 |  0.825 |    0.602 |     1.053 |         0 |         0 |
| animal_value_context | 100 |  0.59  |    0.392 |     0.791 |         0 |         0 |
| context              | 100 |  0.9   |    0.633 |     1.176 |         0 |         0 |
| id                   | 100 |  0.49  |    0.32  |     0.685 |         0 |         0 |
| value                | 100 |  0.62  |    0.41  |     0.845 |         0 |         0 |
| value_context        | 100 |  0.64  |    0.467 |     0.822 |         0 |         0 |

## Source-language control versus base

| condition   | category             |   n |   mean |   ci_low |   ci_high |   p_value |   q_value |
|:------------|:---------------------|----:|-------:|---------:|----------:|----------:|----------:|
| control_zh  | animal               | 100 |  0.84  |    0.618 |     1.076 |         0 |         0 |
| control_zh  | animal_context       | 100 |  0.6   |    0.392 |     0.814 |         0 |         0 |
| control_zh  | animal_value         | 100 |  0.82  |    0.601 |     1.04  |         0 |         0 |
| control_zh  | animal_value_context | 100 |  0.35  |    0.21  |     0.49  |         0 |         0 |
| control_zh  | context              | 100 |  0.835 |    0.61  |     1.081 |         0 |         0 |
| control_zh  | id                   | 100 |  0.39  |    0.215 |     0.585 |         0 |         0 |
| control_zh  | value                | 100 |  0.605 |    0.405 |     0.815 |         0 |         0 |
| control_zh  | value_context        | 100 |  0.52  |    0.339 |     0.703 |         0 |         0 |

## Judge reliability

| condition   | metric                |     n |   exact_agreement | mean_absolute_difference   |   quadratic_weighted_kappa |
|:------------|:----------------------|------:|------------------:|:---------------------------|---------------------------:|
| ALL         | target_value_support  | 13588 |             0.769 | 0.255                      |                      0.741 |
| ALL         | behavioral_commitment | 13588 |             0.754 | 0.271                      |                      0.746 |
| ALL         | tradeoff_priority     |  6794 |             0.755 | 0.274                      |                      0.839 |
| ALL         | coherence_relevance   | 13600 |             0.892 | 0.108                      |                      0.482 |
| ALL         | outcome               | 13600 |             0.951 | NA                         |                      0.629 |

## Robustness checks

| feature                |   primary_mean |   primary_n |   complete_case_mean |   complete_case_n |   high_coherence_mean |   high_coherence_n |   complete_case_minus_primary |   high_coherence_minus_primary |
|:-----------------------|---------------:|------------:|---------------------:|------------------:|----------------------:|-------------------:|------------------------------:|-------------------------------:|
| conditionality         |         -0.016 |         800 |               -0.014 |               798 |                -0.006 |                743 |                         0.001 |                          0.01  |
| constituent_order      |         -0.18  |         800 |               -0.181 |               799 |                -0.184 |                741 |                        -0.001 |                         -0.004 |
| epistemic_hedging      |         -0.066 |         800 |               -0.067 |               799 |                -0.055 |                758 |                        -0.001 |                          0.01  |
| formality              |         -0.009 |         800 |               -0.009 |               798 |                -0.002 |                607 |                        -0.001 |                          0.006 |
| fusion                 |         -0.236 |         800 |               -0.236 |               800 |                -0.239 |                688 |                         0     |                         -0.003 |
| genericity             |         -0.031 |         800 |               -0.032 |               799 |                -0.012 |                763 |                        -0.001 |                          0.019 |
| inflectional_synthesis |         -0.154 |         800 |               -0.154 |               800 |                -0.141 |                669 |                         0     |                          0.013 |
| modal_strength         |          0.004 |         800 |                0.003 |               799 |                 0.015 |                744 |                        -0.001 |                          0.01  |
| narrative_framing      |         -0.058 |         799 |               -0.054 |               796 |                -0.043 |                719 |                         0.004 |                          0.015 |
| nominalization         |         -0.054 |         800 |               -0.048 |               798 |                -0.041 |                764 |                         0.006 |                          0.014 |
| pronoun_distribution   |          0.011 |         800 |                0.009 |               799 |                 0.018 |                758 |                        -0.001 |                          0.007 |
| subordination          |         -0.048 |         800 |               -0.046 |               799 |                -0.026 |                722 |                         0.002 |                          0.022 |
| tense_aspect           |         -0.037 |         800 |               -0.038 |               799 |                -0.023 |                758 |                        -0.001 |                          0.014 |
| voice                  |         -0.037 |         800 |               -0.038 |               799 |                -0.027 |                733 |                        -0.001 |                          0.01  |

Correlation of primary and two-judge complete-case overall effects: **1.000**.

## Lexical-diversity association

Within the primary same-language feature set (n=11), Pearson r = **0.518** (permutation p=0.100) and Spearman rho = **0.336** (permutation p=0.314).

Leave-one-feature-out ranges: Pearson [0.449, 0.677], Spearman [0.224, 0.552].

## Interpretation guardrails

- The statistical unit for uncertainty is the semantic cell rather than the raw textual realization.
- Feature effects are paired against their registered controls; cross-linguistic manipulations should not be interpreted as isolating a single linguistic property when language identity changes with the intervention.
- Model-judge scores are measurement instruments rather than human ground truth. Judge agreement and complete-case sensitivity should be inspected alongside effect estimates.
- Category-level and tradeoff analyses involve many comparisons. Report q-values and confidence intervals rather than selecting results by unadjusted p-values.
- The benchmark consists of generated scenarios. Conclusions are about this benchmark distribution and should be replicated on independently generated or human-authored evaluations before strong external-validity claims.
