# LoRA Analysis

## Question

How do linguistic training variants change LoRA parameter updates, which adapted modules carry these differences, and how do update magnitude, subspaces and on-input effects change from the midpoint to the final checkpoint?

## Inputs

For every model family and adapter condition, load the immutable pretrained base checkpoint and the corresponding independently trained **midpoint and final** LoRA adapter. Analyze English canonical, Mandarin control and all 14 linguistic-feature adapters. Compare each feature with its registered language control, not indiscriminately with English canonical. The existing Gemma training run uses midpoint at optimizer step 231 and final at step 462; determine half-run step counts from each new model's training summary rather than assuming those exact counts recur.

Adapters are trained independently from the same base weights; a feature adapter is not an update *on top of* canonical.

## Effective updates and module coverage

For each adapted layer and target module (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` where supported), reconstruct the effective weight update

```text
DeltaW = (lora_alpha / rank) * B @ A
```

with the implementation's applicable scaling, orientation, and any rank-stabilization settings verified against PEFT forward output. Analyze **DeltaW**, not raw A/B factors, because factor rotations/rescalings do not uniquely represent the same effective matrix. Record module shape and target-module coverage for every architecture; exclude architecture-mismatched modules from direct module-class comparison.

## Metrics

- **Magnitude:** Frobenius and spectral norms of DeltaW; normalize Frobenius norm by base weight Frobenius norm for within-model contrasts. Report feature/control norm ratios and normalized DeltaW_feature−DeltaW_control norm.
- **Rank/spectrum:** singular-value spectrum, effective rank (entropy-based), and the fraction of singular energy in leading components; small-rank operations can be computed through low-rank factors without materializing large dense matrices.
- **Alignment:** Frobenius cosine similarity of effective updates, top-k left/right singular-subspace principal-angle cosines, and a scale-separated difference measure. Distinguish within-module comparisons of updates from comparisons across different module shapes.
- **Functional perturbation:** on matched unsteered prompts, compute DeltaW x on the input activation for that module, then summarize feature/control output displacement and RMS. The implementation reports absolute output RMS and feature/control RMS difference; further output normalization and directional-angle diagnostics can be added after GPU validation. Evaluate on both welfare and general-domain prompts with the same pooled-token convention. This is a local linear module response, not the full model's causal effect.

## Midpoint-to-final trajectory

Compute every metric above at midpoint and final. Additionally report norm growth, normalized direction cosine between DeltaW_mid and DeltaW_final, subspace rotation, and functional-output displacement on a fixed held-out prompt sample. Compare *control-versus-feature divergence at midpoint* with the same divergence at final; this distinguishes convergence/divergence in parameter space from simple growth in norm. Plot two-point trajectories across depth and module type. Two checkpoints cannot establish an exact temporal onset or nonlinear training dynamics.

## Procedure

1. Validate base revision, LoRA metadata, factor shapes, saved training steps and adapter integrity. Verify reconstructed DeltaW x matches the PEFT LoRA contribution on sampled inputs.
2. Enumerate all modules and compute effective-update summaries, feature-control comparisons and checkpoint-to-checkpoint summaries.
3. Run fixed welfare/general prompt batches, compute module input activations and on-input DeltaW x *on GPU*, reduce to statistics and release buffers. Do not retain raw token activations or full dense update matrices.
4. Report module/layer heatmaps, attention versus MLP aggregate summaries, and midpoint/final profiles. Comparisons across families use relative statistics and depth-normalized positions.

## Outputs

Store small metric tables, singular-value summaries, layer/module heatmaps, metadata, checksums and plots. The original midpoint and final adapter tensors are required externally for reexecution but are not duplicated in the repository. No extra checkpoints or per-token activation dumps are created.

## Interpretation

A large parameter difference need not produce a large activation or behavioral difference. Low-rank subspaces can differ while producing similar functional outputs on the tested distribution. Parameter analyses are descriptive and cannot by themselves establish that a particular module causes the observed welfare-generalization effect.
