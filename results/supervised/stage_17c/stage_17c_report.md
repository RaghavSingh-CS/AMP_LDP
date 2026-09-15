# Stage 17C — Confirmation Gate #1

## Frozen gate and access disclosure

This gate asks whether the exact frozen Stage 17B direct-customer HGB retains
strong discrimination on `CONFIRMATION`. Hashes, 48-feature order, HGB parameters,
four `DEV_TRAIN` persistence references, threshold, and A/B/C interpretation
boundaries were recorded in `pre_gate_audit.json` before confirmation features or
labels were loaded. The saved Stage 17B model was not retrained, and no confirmation
quantity was fitted or optimized.

**CONFIRMATION GATE #1 WAS ACCESSED ONCE FOR THE FROZEN STAGE 17B PIPELINE.**
**FINAL_TEST WAS NOT ACCESSED.**

## Confirmation representation and primary result

All 6,355 governed confirmation customers are representation-available and form
one finite 48-column row each. There are 5,813 normal and 542 theft customers;
prevalence is `0.085287175452`. IDs are exactly the original
validation/`CONFIRMATION` role, with no development or final-test contamination.
The frozen Stage 17B threshold is `0.20070617709942135`.

| Model | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Stage 17B direct HGB | 0.789084 | 0.379058 | 0.364668 | 0.415129 | 0.388266 | 0.200706 | 5421 | 392 | 317 | 225 |
| Frozen Stage 14 window HGB → customer mean | 0.738697 | 0.261320 | 0.249110 | 0.387454 | 0.303249 | 0.108127 | 5180 | 633 | 332 | 210 |

Threshold metrics for the direct pipeline use the frozen pre-confirmation Stage
17B threshold. The Stage 14 comparator uses its pre-existing frozen threshold,
which was historically selected on this same original-validation population;
its threshold metrics are therefore optimistic and only ranking comparisons are
scientifically emphasized.

## Development-to-confirmation change

Confirmation minus Stage 17B `DEV_VAL` is `+0.000253226265` ROC-AUC,
`+0.019263265545` AP,
`+0.002050924734` precision, `+0.030970735450` recall,
and `+0.015188823256` F1.

The independent two-sample bootstrap gives ROC-AUC delta mean
`+0.000369`, 95% CI
`[-0.029446, +0.030762]`,
and AP delta mean
`+0.019691`, CI
`[-0.039080, +0.079767]`.
The populations were resampled independently and never pooled.

## Confirmation uncertainty

Across 2,000 confirmation-customer resamples with seed 42, ROC-AUC is
`0.789084` with 95% percentile CI
`[0.768115, 0.810559]`.
AP is `0.379058` with CI
`[0.337648, 0.422308]`.

## Frozen Stage 14 formulation comparator

The optional comparator was validly evaluated without retraining or tuning.
Direct Stage 17B HGB minus Stage 14 window-to-customer HGB is
`+0.050387126957` ROC-AUC
with paired 95% CI
`[+0.034383, +0.066312]`
and `100.000%` above zero. AP delta is
`+0.117738006510`
with CI
`[+0.085089, +0.151752]`
and `100.000%` above zero.

## Gate decision and interpretation

Decision **A — STRONG CONFIRMATION** follows the pre-gate numerical
policy. AP is interpreted against the confirmation prevalence baseline, and the
development-to-confirmation intervals determine whether modest changes are
compatible with ordinary sampling variation. This gate supports generalization
to a held-out SGCC subset, not another dataset.

The direct HGB retains strong discrimination: AP is about `4.44×` the `8.53%`
prevalence baseline. Both independent development-to-confirmation intervals
include zero, so the small ROC-AUC change and modest AP increase are compatible
with ordinary sampling variation. The frozen direct formulation also retains a
clear paired ranking advantage over Stage 14. Gate #1 therefore strengthens the
customer-formulation hypothesis without proving that propagated-window label
noise is the causal mechanism.

If confirmation is strong, the Stage 17B direct-customer HGB remains frozen and
confirmation returns to closed status. The next recommended research question,
using development data only, is: **Does the direct-customer advantage persist
across pre-specified history-length and consumption-level strata?**

For a paper, the gate completes the current evidence chain: reconstruction models
were weak, supervised signal existed, behavioural summaries improved raw windows,
FC signals added no value, direct customer modeling improved formulation, the
representation replicated under a new development split, and the frozen pipeline
was now tested once on confirmation. This can support a transparent held-out SGCC
confirmation claim, but not SOTA, deployment readiness, external generalization,
or causal proof of label noise.
