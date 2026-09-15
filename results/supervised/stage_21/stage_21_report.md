# Stage 21 — Independent Strong Supervised Benchmark

## Question and governance

Stage 21 tests whether the frozen 48-feature AMP customer representation remains competitive across a minimal, independently specified conventional learner panel. Every main model used identical DEV_TRAIN and DEV_VAL customers, features, and labels. No published pipeline was copied. No feature/model co-adaptation, resampling, or tuning occurred.

The machine-readable preregistration is `results/supervised/stage_21/preregistered_benchmark_specification.json` with SHA-256 `2db791b49c1d97b184dced16c719e361ecad543621d04f13d9ba6843f87473c0`. DEV_TRAIN contained 23,724 customers (2,022 theft); DEV_VAL contained 5,931 customers (505 theft); overlap was zero.

The first sealed execution passed the exact HGB reproduction gate and then
stopped because a timing-only helper required bit-for-bit equality across
parallel repeated inference. The assertion was corrected to a `1e-12` absolute
floating-point tolerance without changing the preregistration, model panel,
parameters, metrics, comparisons, or results. This implementation correction is
recorded in `implementation_correction.json`.

## Environment and panel

scikit-learn `1.9.0` and LightGBM `4.7.0` were already available. XGBoost and CatBoost were absent and were excluded without modifying the environment. The frozen panel comprised logistic regression, random forest, HGB, LightGBM, and the fixed arithmetic HGB/LightGBM probability mean.

## HGB reproduction gate

The Stage 17B HGB was reconstructed before new models were interpreted. Maximum probability difference was `9.71e-17`; ROC-AUC/AP differences were `+0`/`+0`. The `1e-12` stop gate **passed**.

## Complete benchmark

All rows contain 5,931 customers, 505 theft customers, and prevalence 0.085146. Threshold metrics use fixed probability 0.5 and are descriptive only.

| Model | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---|
| LOGISTIC_REGRESSION | 0.752240 | 0.262321 | 0.468750 | 0.059406 | 0.105448 | 5392/34/475/30 |
| RANDOM_FOREST | 0.779494 | 0.349708 | 0.744186 | 0.126733 | 0.216582 | 5404/22/441/64 |
| HGB | 0.788831 | 0.359795 | 0.650000 | 0.154455 | 0.249600 | 5384/42/427/78 |
| LIGHTGBM | 0.785410 | 0.358024 | 0.628099 | 0.150495 | 0.242812 | 5381/45/429/76 |
| BOOST_MEAN_HGB_LIGHTGBM | 0.788887 | 0.361659 | 0.635593 | 0.148515 | 0.240770 | 5383/43/430/75 |

## Complexity and runtime

Inference time is the median of five post-warmup full-DEV_VAL probability passes. Parameter counts are reported only where meaningful; tree and leaf counts describe tree ensembles.

| Model | Fit seconds | Inference seconds | Serialized bytes | Parameters | Trees | Leaves |
|---|---:|---:|---:|---:|---:|---:|
| HGB | 1.160144 | 0.017435 | 274208 | — | 100 | 1500 |
| LOGISTIC_REGRESSION | 0.295597 | 0.002237 | 2865 | 49 | — | — |
| RANDOM_FOREST | 6.912960 | 0.129388 | 39249641 | — | 300 | 244731 |
| LIGHTGBM | 1.285457 | 0.063497 | 360326 | — | 200 | 3000 |
| BOOST_MEAN_HGB_LIGHTGBM | 0.000000 | 0.110865 | 90 | — | — | — |

## Paired uncertainty

The pre-registered rule selected the strongest non-HGB individual by AP and separately by ROC-AUC if different, plus the fixed ensemble. Deltas are model minus HGB over 2,000 paired DEV_VAL customer replicates with seed 42.

| Model | Metric | Observed delta | Bootstrap mean | 95% interval | Fraction below 0 | Fraction above 0 |
|---|---|---:|---:|---|---:|---:|
| LIGHTGBM | ROC_AUC | -0.003421 | -0.003384 | [-0.009190, +0.002134] | 0.8870 | 0.1130 |
| LIGHTGBM | AP | -0.001771 | -0.001059 | [-0.013817, +0.011291] | 0.5620 | 0.4380 |
| BOOST_MEAN_HGB_LIGHTGBM | ROC_AUC | +0.000057 | +0.000081 | [-0.002757, +0.002828] | 0.4750 | 0.5250 |
| BOOST_MEAN_HGB_LIGHTGBM | AP | +0.001865 | +0.002164 | [-0.004723, +0.008461] | 0.2530 | 0.7470 |

## Decision

Decision **B — STRONG CONVENTIONAL LEARNERS ESSENTIALLY EQUIVALENT**.

No individual benchmark materially beat HGB under the frozen joint effect-size and uncertainty rule: `[]`. The fixed ensemble materially beat HGB: **False**. LightGBM fell within the frozen equivalence margins: **True**.

## Paper and next step

HGB and LightGBM cluster within the frozen practical margins, supporting the interpretation that AMP's customer representation, rather than a uniquely favorable classifier, carries the development result.

Literature is used only to motivate testing strong tabular boosting models; these results do not establish protocol-equivalent superiority over published SGCC studies and are not a state-of-the-art claim.

Next step: **RUN ONE FINAL COMPACT-REPRESENTATION EXPERIMENT**. Stage 19 found strong feature-family redundancy and Stage 21 finds classifier-level boosting equivalence, jointly motivating one pre-specified simplification test rather than more model hunting.

No adaptive tuning or post-hoc model change occurred. The frozen Stage 17B detector and 48-feature representation were not modified.

**CONFIRMATION NOT ACCESSED DURING STAGE 21.**

**FINAL_TEST NOT ACCESSED DURING STAGE 21.**
