# Interactions Between Linguistic Features and Value Generalization

## Research question

When multiple linguistic features are manipulated simultaneously in otherwise semantically matched fine-tuning data, do their effects on value generalization combine additively, amplify one another, or cancel out? Can selectively strengthening or reducing individual features produce larger and more reliable behavioral effects than applying each feature in isolation?

## Hypothesis

Linguistic interventions may interact nonlinearly. Some combinations could reinforce the same learned behavioral tendencies, while others could suppress them. Consequently, the strongest generalization effect may arise from a particular combination and intensity of features rather than from the strongest individual intervention.

## Proposed methodology

Starting from a common corpus with fixed semantic content, construct controlled training conditions for selected pairs of linguistic features. Include a canonical control, each feature individually, and both features together. Where feasible, add graded manipulation levels (including reduced or absent realization) to test whether weakening one feature strengthens the effect of another.

Fine-tune models from the same base checkpoint using matched training budgets, data IDs, and evaluation settings. Prioritize a small number of theoretically motivated combinations before expanding to broader factorial experiments.

Evaluate all conditions on the same held-out value-generalization scenarios. Compare the combined intervention against the sum of the individual effects relative to the canonical control. This interaction contrast distinguishes approximate additivity from **synergy** (a larger-than-additive effect) and **antagonism** (a smaller-than-additive or opposing effect). Examine whether interactions vary across generalization categories and competing-value tradeoffs.

## Expected contribution

The study would move beyond estimating the effects of isolated linguistic features to identifying how they interact, and whether coordinated increases or reductions can systematically alter value generalization. It could also inform more efficient linguistic intervention design by identifying feature combinations that maximize, reverse, or stabilize downstream effects.

Interpretation would require care: combined rewrites may introduce semantic drift or change text length and lexical diversity. Matched reviews, manipulation-strength checks, and repeated training seeds would be important for separating feature interactions from these confounds.
