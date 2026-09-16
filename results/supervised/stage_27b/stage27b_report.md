# Stage 27B — terminal Study 2 evaluation

## Cohort governance

The final evaluation cohort had been accessed historically in Stage 13 for an obsolete supervised window-level baseline. The learned latent-distribution hypothesis, architecture, pooling rule, five-seed ensemble, controls, and confirmation decision reported in Study 2 were developed and frozen later without Study 2 row-level access to this cohort. We therefore treat it as a governed final evaluation cohort with disclosed historical baseline access, not as a pristine untouched test set.

The cohort was opened exactly once at `2026-09-16T11:14:39.840891+05:30` after preregistration `ed2457e3a6eaf4b683c3fd8aaf89cf1f1fca63500a68f0427279813533e92702` and every pre-access assertion passed. It contained 6,356 customers (5,814 normal; 542 theft-labelled), prevalence 0.085274, and 165,020 eligible windows, with zero exclusions or protected-role overlap.

## Frozen results

LATENT five-seed mean: `0.811729` ROC / `0.421863` AP. FULL_48: `0.797863/0.385326`. DIRECT: `0.794967/0.368439`. No model was fitted, refit, calibrated, tuned, or selected, and no threshold was optimized.

Individual seed results were:

| Seed | ROC-AUC | AP | Both FULL_48 parity margins |
|---:|---:|---:|:---:|
| 42 | 0.799877 | 0.399000 | yes |
| 43 | 0.794723 | 0.389762 | yes |
| 44 | 0.800114 | 0.375061 | no |
| 45 | 0.804291 | 0.394060 | yes |
| 46 | 0.794317 | 0.387527 | yes |

The individual-seed mean±population-SD was `0.798664±0.003732` ROC and
`0.389082±0.008033` AP; 4/5 seeds met both parity margins. No frozen seed-
collapse condition was present.

Observed LATENT-minus-FULL_48 deltas were `+0.013867` ROC and `+0.036538` AP.
Their paired 95% intervals were `[+0.000466,+0.027511]` and
`[+0.007993,+0.065214]`, with fractions above zero 0.977 and 0.995.
LATENT-minus-DIRECT deltas were `+0.016762/+0.053424`; intervals were
`[+0.006282,+0.027865]` and `[+0.027428,+0.077907]`, with fractions above zero
0.998 and 1.000. DIRECT-minus-FULL_48 was `-0.002895/-0.016886`; both intervals
crossed zero.

LATENT's final-minus-DEV_VAL shifts were `+0.020920/+0.054026`; its final-
minus-CONFIRMATION shifts were `-0.000062/+0.006576`. The ordering
`LATENT > FULL_48 > DIRECT` held for both metrics at DEV_VAL, CONFIRMATION, and
the final evaluation cohort with disclosed historical baseline access.

## Terminal decision

**FINAL A — STRONG FINAL REPLICATION**. **A — ANCHOR SURVIVES**.

Final anchor sentence: **Learning window representations before distribution-preserving customer aggregation adds reproducible customer-level discrimination beyond identical direct aggregation across development, confirmation, and the governed final evaluation.**

Paperworthiness: **LEVEL 1 — strong methodological result**. Study 2 model development is permanently closed; no rescue experiment is permitted.

## Scientific interpretation

Distribution-preserving aggregation solved the principal single-vector MIL
bottleneck observed in Stage 24B: the frozen ensemble recovered and exceeded
the conventional customer-profile comparator. Learned latent encoding added
information beyond identical direct distribution aggregation, with both final
paired intervals above zero. The direction persisted across DEV_VAL,
CONFIRMATION, and the terminal evaluation. Magnitudes varied—confirmation had
the largest ROC gain, while the AP gain was similar in confirmation and final
evaluation—so exact effect-size invariance is not claimed.

The result supports customer-level weak supervision and distribution-preserving
representation learning. It does not validate propagated customer labels as
window-level theft targets, and no high-scoring window is established as a
verified theft period.

## Claim and closure audit

Allowed claims are limited to the learned latent-distribution representation,
customer-level electricity-theft discrimination under weak customer labels,
distribution-preserving aggregation, reproducibility across the governed
stages, improvement over DIRECT, and improvement over FULL_48 in this protocol.
Window detection/localization, individual-window theft probabilities,
unsupported state-of-the-art status, external generalization, deployment
validity, causal claims, and pristine/untouched cohort language remain
prohibited.

Stage 27B is terminal. No refit, tuning, new seed, feature, statistic,
architecture, classifier, threshold search, or representation rescue is
permitted.
