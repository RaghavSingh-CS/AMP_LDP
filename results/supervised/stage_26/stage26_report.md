# Stage 26 — independent confirmation of learned distribution representation

## Governed access

Protocol SHA-256: `7207a75dfd31d3254aaa3d4507eb9812b38ff87ecda09abfa62178572c8be09b`. CONFIRMATION was opened once at `2026-09-16T10:41:09.316234+05:30` for frozen inference. It contained 6,355 customers (542 theft-labelled) and 165,541 eligible windows. FINAL_TEST remained closed.

## Results

Five-seed mean LATENT scored ROC-AUC `0.811791` and AP `0.415287`. Frozen FULL_48 scored `0.789084/0.379058`; frozen DIRECT scored `0.787000/0.355725`. LATENT deltas were `+0.022707/+0.036229` versus FULL_48 and `+0.024792/+0.059563` versus DIRECT.

Individual latent seed results were:

| Seed | ROC-AUC | AP | Both parity margins |
|---:|---:|---:|:---:|
| 42 | 0.796431 | 0.383587 | yes |
| 43 | 0.803876 | 0.391508 | yes |
| 44 | 0.796913 | 0.386670 | yes |
| 45 | 0.797414 | 0.386519 | yes |
| 46 | 0.793075 | 0.372681 | yes |

The individual-seed mean±population-SD was `0.797542±0.003513` ROC and
`0.384193±0.006292` AP. All 5/5 seeds satisfied both frozen parity margins.

The 2,000 paired customer bootstraps used sealed seed 2502. LATENT-minus-FULL
95% intervals were `[0.009839,0.035194]` ROC and `[0.009530,0.062749]` AP,
with fractions above zero `1.000/0.996`. LATENT-minus-DIRECT intervals were
`[0.013757,0.035633]` ROC and `[0.034935,0.083791]` AP, both with fraction
above zero 1.000. DIRECT-minus-FULL intervals crossed zero:
`[-0.014867,0.010374]` ROC and `[-0.049094,0.003775]` AP.

From DEV_VAL to CONFIRMATION, LATENT shifted `+0.020982` ROC and `+0.047450`
AP, FULL_48 shifted `+0.000253/+0.019263`, and DIRECT shifted
`-0.001234/+0.006481`. The ordering `LATENT > FULL_48 > DIRECT` replicated for
both metrics. The stronger confirmation effect does not imply the exact effect
size is stable; it establishes directionally consistent independent support.

## Decision

**CONFIRMATION A — POSITIVE REPLICATION**. **ANCHOR_EARNED = YES**. The independent result supports a non-tiny, uncertainty-supported learned-encoder AP contribution across cohorts.

Anchor sentence: **Distribution-preserving learned window representations
reproducibly add customer-level precision-recall signal beyond direct
distribution aggregation across development and independent confirmation.**

The main limitation is that CONFIRMATION is one governed cohort from the same
historical data source, not external validation. The primary ensemble also
outperformed every individual seed, so the claim concerns the frozen five-seed
pipeline rather than a single encoder.

No model was fitted, selected, tuned, or altered using confirmation. No instance-level FLAG supervision was used. FINAL_TEST was not accessed.
