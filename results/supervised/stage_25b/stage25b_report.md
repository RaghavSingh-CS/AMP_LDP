# Stage 25B — learned customer-distribution experiment

## Result

Outcome **B — REPRESENTATION PARITY**. `ANCHOR_EARNED = NO`. Material inferiority: **False**.

The sealed specification SHA-256 is `006c781899f74f0f62d01282fbf4164ac3ee27d8a61e3f2ea5fcb09b9e445470`. Exactly 11 models were fitted: five end-to-end distribution models, five latent-distribution HGB controls, and one direct SUMMARY_17 distribution HGB control.

## Representation and data

The model uses the frozen 17→32→16 encoder (1,104 parameters), dimension-major mean/population-SD/q10/median/q90/maximum pooling (96 values), and a linear 96→1 head (97 parameters), for exactly 1,201 trainable parameters. The direct control applies the same six statistics to SUMMARY_17, producing 102 values. DEV_TRAIN contains 23,724 customers and 614,976 windows; DEV_VAL contains 5,931 customers and 153,947 windows.

## Primary findings

Mean end-to-end: ROC-AUC 0.777354, AP 0.350802. Mean latent-HGB: ROC-AUC 0.790809, AP 0.367837. Direct-distribution HGB: ROC-AUC 0.788233, AP 0.349244. Frozen FULL_48 HGB: ROC-AUC 0.788831, AP 0.359795.

Latent-HGB minus FULL_48: ROC +0.001978, AP +0.008043. Latent-HGB minus direct control: ROC +0.002576, AP +0.018593.

Seed-level ROC-AUC/AP results were:

| Seed | End-to-end | Latent-HGB |
|---:|---:|---:|
| 42 | 0.787797 / 0.376655 | 0.794037 / 0.378558 |
| 43 | 0.779190 / 0.351480 | 0.787755 / 0.365537 |
| 44 | 0.762317 / 0.331907 | 0.788919 / 0.354154 |
| 45 | 0.778715 / 0.349588 | 0.786804 / 0.367526 |
| 46 | 0.778750 / 0.344381 | 0.796531 / 0.373413 |

End-to-end population SD was 0.008275 ROC and 0.014619 AP. Latent-HGB
population SD was 0.003797 ROC and 0.008230 AP. All five latent-HGB seeds met
both representation-parity margins, while two improved both metrics over
FULL_48.

The preregistered customer bootstrap first averaged predictions across the five
latent representations. Against FULL_48, the resulting paired 95% intervals
were `[0.000181, 0.029310]` for ROC delta and `[0.007365, 0.064951]` for AP
delta. Against direct distribution aggregation, they were
`[0.003552, 0.026893]` and `[0.017151, 0.072318]`, respectively. These are
ensemble-prediction uncertainty estimates; the seed-level metrics above remain
the primary reproducibility record.

Relative to MEAN_MIL, latent-HGB improved ROC/AP by `+0.050942/+0.100016`;
relative to MAX_MIL by `+0.034142/+0.053271`; and relative to ATTENTION_MIL by
`+0.032756/+0.070375`. Thus multi-statistic pooling recovered the entire
single-vector MIL deficit on the mean metrics. The direct control also nearly
recovered that deficit, so aggregation itself explains much of the gain.

## Interpretation

The learned representation added measurable AP value over direct aggregation
(`+0.018593`) with a paired interval above zero; its ROC gain was smaller
(`+0.002576`). Under the evaluated SUMMARY_17 plus latent-distribution
formulation, explicit FULL_48 customer profiling is not stronger on the mean
primary metrics. However, the gains over FULL_48 were below the anchor-candidate
effect thresholds and only two seeds improved both metrics. No paper anchor was
earned. This is a narrow representation result, not a general claim about
neural networks.

No instance-level FLAG target or loss was used. CONFIRMATION and FINAL_TEST remained closed.

## Stop rule

The Stage 25B weak-result stop rule does not activate.
