# Generalization from linguistic fine-tuning

A possible research direction is to study whether low-level linguistic fine-tuning on a semantically broad, domain-general corpus changes model behavior in a separate value domain.

## Setup

The training data would use the same controlled paired-data design as the main project:

- generate one shared canonical corpus across broad, ordinary topics;
- derive one transformed corpus per linguistic variable while preserving semantic and value content;
- fine-tune one canonical model and one model per linguistic variable from the same base checkpoint.

The training corpus would not specifically teach animal-welfare values. The linguistic intervention would therefore be learned in a domain-general setting.

## Evaluation

Evaluation would use unseen animal-welfare scenarios formatted for a base model, for example through controlled text continuations or candidate-completion likelihoods.

All models would receive the exact same evaluation items. The central comparison would be each linguistic-feature model against the canonical fine-tune, with the base model included as an additional reference.

## What this tests

This setup tests whether the effect of learning a linguistic pattern transfers across domains and changes pre-existing value-related behavior.

A positive result would suggest that linguistic fine-tuning can influence downstream judgments even when the fine-tuning corpus does not directly teach the evaluated value domain.

A null result would be less informative about value generalization itself, because the training data contains no explicit animal-welfare signal to generalize. The experiment is therefore best interpreted as a test of **cross-domain behavioral effects of linguistic fine-tuning**, rather than generalization of a newly learned animal-welfare value.

This design could be useful as a follow-up because it asks whether linguistic structure alone can systematically shift behavior outside the semantic domain in which it was learned.
