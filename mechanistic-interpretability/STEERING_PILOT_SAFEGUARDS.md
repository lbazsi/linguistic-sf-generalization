# Steering pilot split safeguard

The initial revision-2 Llama pilot used linguistic TEST pairs to measure its contrastive likelihood proxy. Those observations were exploratory and the consulted test examples cannot subsequently be described as untouched confirmation. The welfare steering validation subset was also previously involved in configuration selection.

Revision 3 moves exploratory linguistic manipulation checks to the semantic-ID-disjoint DEVELOPMENT partition. It leaves the TRAIN direction-fitting partition unchanged, does not touch welfare TEST scenarios, and writes pilot_v3.jsonl and manifest_v3.json without overwriting pilot.jsonl or manifest.json. The Llama v2 pilot remains exploratory; the split correction does not restore untouched status to previously examined examples.

A measured linguistic log-likelihood contrast is not equivalent to verifying a grammatical transformation in generated responses. Before scaling to all feature–adapter combinations, perform blinded feature-realization/coherence/language-identity annotation of matched continuations and check shuffled/random direction controls. Document selection multiplicity across layer, site, policy and dose. Do not advance to expensive judging only because a small pilot proxy has a nominally significant unadjusted confidence interval.

The full steering code's ordered JSON judging responses rely on API responses preserving item order; an ID-keyed batch response mapping and matched arm-blind judging are desirable improvements before launching the final judged test. Judge-code updates require compatible batch schema and prompt changes and must be validated on synthetic shuffled responses before deployment.
