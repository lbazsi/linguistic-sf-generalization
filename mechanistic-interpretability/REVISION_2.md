# Mechanistic Analysis — Revision 2

Revision 2 preserves legacy outputs. It repairs the behavioral regressor, stratifies LoRA functional inputs, computes global RMS correctly, adds a validation-only steering pilot, and withdraws invalid in-sample SVCCA from figures. No existing black-box judgments or LoRA checkpoints need rerunning.

## Llama-first sequence

Use the research/mechinterp-v2-20261026 branch and the existing attached adapter filesystem. From the repository root:

~~~bash
git fetch origin
git switch research/mechinterp-v2-20261026
cd mechanistic-interpretability
python -m pytest tests -q
python run.py validate --model llama31 --check-screening
python run.py probe --model llama31 --task behavior --condition canonical
~~~

Review outputs/revision_2/llama31/linear-probing/behavior_all_canonical.jsonl. Each layer/site record must have method dual_ridge_v2, train/development/test performance, constant-mean baseline, penalty choice and clipping diagnostics. Inspect pathological errors and non-finite values before expansion. Positive R2 is not guaranteed and is not a prerequisite for reporting a valid null.

Then run all remaining behavioral conditions and both LoRA functional checkpoints:

~~~bash
python run_all.py --model llama31 --stage behavior-probes --execute --resume
python run_all.py --model llama31 --stage functional --execute --resume
~~~

The functional analysis samples 10 examples per welfare category x tradeoff stratum (160 total) and 10 per general-domain stratum (200 total), using the same deterministic prompts across all conditions. Check both functional midpoint/final sampling manifests. Do not repeat existing LoRA weight geometry.

## Steering calibration gate

The original selected_all.json must be present in the legacy outputs directory; if absent, restore it from the committed archived results:

~~~bash
mkdir -p outputs/llama31/activation-steering
cp ../results/mechanistic-interpretability/llama31/activation-steering/selected_all.json outputs/llama31/activation-steering/selected_all.json
python -m mechinterp.steering_pilot --model llama31 --n 20
~~~

The pilot compares seeded unsteered and intervened generation, logs hook count, modified positions and actual perturbation norm, and samples 0%, 3%, 10% and 25% nominal relative norm at signed directions. It generates no remote judgments and does not access the steering test partition. A welfare-option likelihood gap is not evidence that the linguistic property was manipulated. Examine feature realization, coherence, cross-language changes, and multiple null directions separately before any full sweep.

The revision-specific direction cache can be seeded from legacy direction safetensors and matching metadata; otherwise direction extraction will be repeated under the pinned pretrained model. Never interpret the zero-strength proxy as equivalent to a separately judged matched baseline.

**Stop gate:** Expand to full steering only after the intervention passes magnitude/hook checks, demonstrates the intended linguistic manipulation, preserves response coherence, and has stable evidence beyond random/shuffled directional controls. If it fails, report the calibrated null; do not optimize on the already used 100-scenario test.

## Cross-model figures

After committing corrected scalar outputs to results/mechanistic-interpretability/llama31/revision_2, run from repository root:

~~~bash
python -m pip install -r cross-model-analysis/requirements.txt
python cross-model-analysis/run.py --stage all
python -m unittest discover -s cross-model-analysis/tests -v
~~~

The revised cross-model pipeline reads committed black-box inputs under results/black-box/<model>. Legacy in-sample SVCCA and failed behavioral probes are excluded, rather than pooled with revised metrics. The figure suite runs on CPU. Other model families should follow the same validation gate before their corrected metrics are incorporated.

## Reporting caution

Corrected probing can still yield test R2 below zero. All penalties are selected on the development split and compared with the train-mean baseline; clipped predictions and raw ranges are reported. High concept-probe ROC AUC does not establish value endorsement, and welfare steering proxies are not causal proof.
