# Stage 15 — Frozen-FC Hybrid Ablation

## Research question and boundary

Stage 15 asks whether signals from the already-trained, frozen fully connected
autoencoder add incremental validation information beyond Stage 14's 17
behavioural summaries. The FC is evaluated as a possible complementary
representation, not presumed to be the final anomaly detector. This stage uses
train and validation only. It does not retrain or tune the autoencoder, alter
the HGB configuration, introduce a neural network/GAN/SMOTE, or access test.

## Frozen FC audit

The selected FC artifact is `models/autoencoder/ae_stride30_baseline.pt`,
SHA-256 `421f1e2cdf96da1bd3231cd0baa750746d89757700d63ac428f9167f69237fa4`.
Its configuration hash is
`3408e00443df58a32f70520452266e1f3e5a2618974bf15bd913a24f595aa27d`.
The architecture is `30→16→8→4→8→16→30`, with 1,362 parameters and a
four-unit ReLU bottleneck directly available from `model.encoder`.

Stage 4 trained the model on 708,308 usable stride-30 `FLAG=0` training
windows using MSE/Adam, learning rate 0.001, batch 256, and seed 42. Epoch 30
was selected by normal-validation reconstruction MSE (`0.043589054377373454`).
Input is the existing 30-value Stage 2 scaled consumption window. Per-window
reconstruction error is `mean((reconstruction − input)^2)` over 30 positions.
The existing P95 threshold is `0.16436323150992393`; Stage 15 does not use it
as a feature or operating threshold.

The checkpoint was historically selected using normal validation windows.
Although Stage 15 performs no FC fitting or adaptation, all FC-derived features
inherit this validation reuse. Hybrid validation deltas may therefore be
optimistic and are diagnostic rather than an independent external estimate.

## Pre-registered FC-derived features

The exact list was written before Stage 15 validation scoring:

1. `reconstruction_mse`
2. `reconstruction_mae`
3. `maximum_absolute_reconstruction_error`
4. `early_half_reconstruction_mse`
5. `late_half_reconstruction_mse`
6. `late_minus_early_reconstruction_mse`
7. `latent_1`
8. `latent_2`
9. `latent_3`
10. `latent_4`

Pointwise error standard deviation was considered but excluded as redundant
with the other error magnitude features to keep the complete set at ten.
Maximum-error position was excluded because pooled window position has fragile
semantics. The pre-registered representations are `SUMMARY_ONLY`,
`FC_RECON_ONLY`, `FC_ONLY`, `SUMMARY_PLUS_RECON`, and `SUMMARY_PLUS_FC`.

## Leakage, alignment, and sanity

FC inference runs deterministically on CPU in evaluation mode with gradients
disabled and no labels supplied to extraction. The checkpoint and config hashes
were verified before inference. All 768,923 training and 165,541 validation
windows retain their customer/window mappings. Computed validation MSE aligns
one-to-one with all 165,541 saved Stage 4 validation records with maximum
absolute difference `0.0`. No FC feature is non-finite or constant.

The Stage 14 `SUMMARY_ONLY` window/customer metrics reproduce with maximum
absolute difference `5.55e-17`. Model training is the exact frozen Stage 14
unweighted HistGradientBoosting configuration: 100 iterations, learning rate
0.1, 15 leaves, minimum leaf size 100, L2=1, seed 42, and no early stopping.
For each ablation, validation-F1-maximizing thresholds are chosen separately at
window and customer-mean levels.

No test feature, label, prediction, saved test metric, or test evaluation was
loaded or used during Stage 15.

## Complete validation ablation

Validation prevalence is 8.031243% by window and 8.528718% by customer.

### Window level

| Representation | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUMMARY_ONLY | 0.676666 | 0.213635 | 0.218110 | 0.288981 | 0.248593 | 0.125655 | 138473 | 13773 | 9453 | 3842 |
| FC_RECON_ONLY | 0.602813 | 0.122877 | 0.111816 | 0.408499 | 0.175573 | 0.084625 | 109106 | 43140 | 7864 | 5431 |
| FC_ONLY | 0.660341 | 0.199862 | 0.197818 | 0.302821 | 0.239308 | 0.114170 | 135920 | 16326 | 9269 | 4026 |
| SUMMARY_PLUS_RECON | 0.676410 | 0.213784 | 0.210209 | 0.307258 | 0.249633 | 0.118116 | 136898 | 15348 | 9210 | 4085 |
| SUMMARY_PLUS_FC | 0.676596 | 0.214938 | 0.207014 | 0.311696 | 0.248792 | 0.115541 | 136372 | 15874 | 9151 | 4144 |

### Customer level: mean window probability

| Representation | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SUMMARY_ONLY** | **0.738697** | 0.261320 | 0.249110 | 0.387454 | **0.303249** | 0.108127 | 5180 | 633 | 332 | 210 |
| FC_RECON_ONLY | 0.663819 | 0.180220 | 0.177158 | 0.363469 | 0.238210 | 0.089155 | 4898 | 915 | 345 | 197 |
| FC_ONLY | 0.714750 | 0.240460 | 0.255350 | 0.330258 | 0.288013 | 0.114116 | 5291 | 522 | 363 | 179 |
| SUMMARY_PLUS_RECON | 0.738314 | 0.262006 | 0.257106 | 0.367159 | 0.302432 | 0.111438 | 5238 | 575 | 343 | 199 |
| SUMMARY_PLUS_FC | 0.738454 | **0.263043** | **0.256113** | 0.367159 | 0.301744 | 0.111762 | 5235 | 578 | 343 | 199 |

## Customer deltas and paired bootstrap

| Representation | ROC-AUC delta vs summary | AP delta vs summary | F1 delta vs summary |
|---|---:|---:|---:|
| FC_RECON_ONLY | -0.074878 | -0.081099 | -0.065039 |
| FC_ONLY | -0.023947 | -0.020860 | -0.015236 |
| SUMMARY_PLUS_RECON | -0.000383 | +0.000687 | -0.000817 |
| SUMMARY_PLUS_FC | **-0.000243** | **+0.001723** | **-0.001506** |

A deterministic paired bootstrap resampled 6,355 validation customers 2,000
times. For `SUMMARY_PLUS_FC − SUMMARY_ONLY`:

- ROC-AUC delta bootstrap mean: `-0.000256`; percentile 95% CI
  `[-0.001825, 0.001353]`; 37.8% of replicates above zero.
- Average-precision delta bootstrap mean: `+0.001668`; percentile 95% CI
  `[-0.000485, 0.003987]`; 93.5% of replicates above zero.

Both intervals include zero. The bootstrap is descriptive validation
uncertainty and does not repair inherited FC validation reuse.

## Importance and redundancy

In the full hybrid, the strongest FC-feature permutation effects on customer
ROC-AUC/AP are `reconstruction_mae` (`0.003079`/`0.003847`), early-half MSE
(`0.001843`/`0.001427`), late-minus-early MSE (`0.001192`/`0.002086`), and total
MSE (`0.001087`/`0.001210`). Each latent coordinate has ROC-AUC importance at
or below `0.000370`. FC variables are used slightly by the fitted tree, but
their conditional importance is far below the leading Stage 14 summaries.

Redundancy is substantial. Spearman correlations include total MSE with
standard deviation `0.957020`, maximum reconstruction error with maximum
consecutive change `0.961691`, reconstruction MAE with standard deviation
`0.946299`, early-half MSE with mean consecutive change `0.887796`, and
`latent_3` with mean `0.971842`. The signed late-minus-early error is the main
weakly correlated reconstruction quantity, but it does not create an aggregate
hybrid gain. Correlation is not causation, and correlated-feature permutation
importance can distribute credit among substitutes.

## Decision: C — no complementary value

Reconstruction-only features contain some supervised signal but are materially
weaker than behavioural summaries. Latent coordinates recover much of the raw
signal, raising FC-only customer ROC-AUC from `0.663819` to `0.714750`, yet the
complete FC branch does not improve customer ROC-AUC when summaries are present.
Its AP increase is microscopic and uncertain, while F1 declines. The added
neural inference dependency and inherited validation reuse are not justified by
this result.

The FC branch should therefore be removed from the current detector design.
Stage 14 `SUMMARY_ONLY` unweighted HGB remains the frozen supervised baseline.
FC/CNN/LSTM experiments are preserved as paper-relevant evidence that
reconstruction anomaly modelling was investigated through controlled ablations
and rejected for lack of incremental detection value. This negative result
supports an interpretable supervised behavioural formulation rather than a
forced hybrid claim.

## Recommended Stage 16 question

**Does a direct customer-level formulation using compact temporal aggregates of
the frozen 17 window summaries improve discrimination over mean aggregation of
window-level probabilities?** This addresses the mismatch between customer
labels and propagated window labels without requiring a GAN or another neural
architecture. Stage 16 is not executed, and test remains closed.
