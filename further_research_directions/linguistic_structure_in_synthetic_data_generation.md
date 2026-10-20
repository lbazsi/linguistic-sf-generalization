# Linguistic Structure and Value Propagation in Synthetic Data Generation

## Research question

Can low-level linguistic structures influence the value-related content of model-generated synthetic data, even when the semantic content of the models' fine-tuning corpora is controlled? Could such differences propagate or accumulate when generated data is used to train subsequent models?

## Hypothesis

If linguistic structure changes how models generalize values, these differences may also appear in the examples that models generate. Synthetic-data pipelines could therefore transmit or amplify subtle value-related biases through data selection and repeated training, even without an explicit change to the intended values of the source corpus.

## Proposed methodology

Build on the canonical and linguistic-feature fine-tunes from the main project. Ask each model to generate synthetic examples from identical, held-out prompts, using matched decoding settings and generation budgets. Compare the resulting corpora for animal-welfare-related value framing, decisions, and tradeoffs, while separately measuring surface linguistic properties, semantic diversity, and prompt adherence.

A primarily theoretical analysis would characterize the pathway from linguistic intervention to model behavior, generated-data distribution, and possible downstream learning effects. A small illustrative extension could train new models on equal-sized synthetic corpora from different generators and test them on the same held-out value-generalization evaluation. If warranted, repeat generation and training for a limited number of iterations to examine whether differences persist, diminish, or grow.

Use multiple prompts and generation seeds, independent value assessments, and controls for explicit value content, filtering, and topic distribution. Changes in downstream models should be attributed to the *generated data distribution*, rather than assumed to come directly from linguistic features.

## Expected contribution

This project would connect mechanistic findings about linguistic structure and value generalization to the design of synthetic training pipelines. It would offer a framework for understanding when subtle linguistic choices might influence the values represented in generated corpora, and formulate testable predictions about bias propagation across training generations.

Theoretical claims should remain distinct from empirical evidence: the main project's results alone would not establish effects on synthetic-data generation, and a single generation round would not demonstrate cumulative amplification. The proposed follow-up is therefore intended as a separate, theory-led study with limited targeted validation.
