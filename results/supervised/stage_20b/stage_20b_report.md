# Stage 20B — Controlled Label/Objective-Formulation Experiment

## Question and governance

Stage 20B tested whether closer alignment with the customer-level SGCC target provides an advantage not explained by unequal windows per customer or arithmetic-mean probability aggregation. SGCC has no temporal theft-event ground truth, so this experiment does not identify mislabeled windows.

The sealed preregistration is `results/supervised/stage_20b/preregistered_control_experiment.json` with SHA-256 `c198ebc2124402711dc0b1d41301a9590199a70e529ddf7cbfa23e0d04b50c1c`. Only original-train customers assigned DEV_TRAIN or DEV_VAL were loaded. DEV_TRAIN and DEV_VAL overlap is zero. Historical Stage 14 models were not reused.

## Population

| Role | Customers | Theft | Prevalence | Usable windows |
|---|---:|---:|---:|---:|
| DEV_TRAIN | 23724 | 2022 | 0.085230 | 614976 |
| DEV_VAL | 5931 | 505 | 0.085146 | 153947 |

## Window HGB aggregation panel

All models use the exact 17 frozen Stage 14 summaries and fixed Stage 17B HGB settings. Customer-balanced weights equalize each customer's total loss contribution and retain mean window weight 1. All four aggregators are reported without selection.

| Model | N | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| WINDOW_HGB_UNWEIGHTED_MEAN | 5931 | 505 | 0.085146 | 0.733770 | 0.250537 | 1.000000 | 0.007921 | 0.015717 | 5426/0/501/4 |
| WINDOW_HGB_UNWEIGHTED_MEDIAN | 5931 | 505 | 0.085146 | 0.699766 | 0.224961 | 0.800000 | 0.007921 | 0.015686 | 5425/1/501/4 |
| WINDOW_HGB_UNWEIGHTED_Q90 | 5931 | 505 | 0.085146 | 0.739718 | 0.274093 | 0.658537 | 0.053465 | 0.098901 | 5412/14/478/27 |
| WINDOW_HGB_UNWEIGHTED_MAXIMUM | 5931 | 505 | 0.085146 | 0.739127 | 0.254490 | 0.455285 | 0.110891 | 0.178344 | 5359/67/449/56 |
| WINDOW_HGB_CUSTOMER_BALANCED_MEAN | 5931 | 505 | 0.085146 | 0.736791 | 0.252988 | 1.000000 | 0.005941 | 0.011811 | 5426/0/502/3 |
| WINDOW_HGB_CUSTOMER_BALANCED_MEDIAN | 5931 | 505 | 0.085146 | 0.704229 | 0.226192 | 0.750000 | 0.005941 | 0.011788 | 5425/1/502/3 |
| WINDOW_HGB_CUSTOMER_BALANCED_Q90 | 5931 | 505 | 0.085146 | 0.741774 | 0.275693 | 0.705882 | 0.047525 | 0.089054 | 5416/10/481/24 |
| WINDOW_HGB_CUSTOMER_BALANCED_MAXIMUM | 5931 | 505 | 0.085146 | 0.738443 | 0.254793 | 0.439024 | 0.106931 | 0.171975 | 5357/69/451/54 |

## Matched logistic loss-placement control

Both arms use one shared customer-weighted DEV_TRAIN transformation of the same 17 local features and fixed L2 logistic settings. W_LOGIT applies loss to customer-balanced proxy-labelled windows and averages window probabilities; C_LOGIT applies loss once per customer after arithmetic feature pooling.

| Model | N | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W_LOGIT | 5931 | 505 | 0.085146 | 0.688480 | 0.212441 | 0.000000 | 0.000000 | 0.000000 | 5423/3/505/0 |
| C_LOGIT | 5931 | 505 | 0.085146 | 0.697107 | 0.214561 | 0.452381 | 0.037624 | 0.069470 | 5403/23/486/19 |

Threshold metrics use the pre-specified fixed probability threshold 0.5 and are descriptive only.

## Paired comparisons and uncertainty

Two thousand paired DEV_VAL customer bootstrap replicates used seed 42. Every comparison below was frozen before scoring.

| Comparison | Metric | Observed delta | Bootstrap mean | 95% percentile CI | Fraction below 0 | Fraction above 0 |
|---|---|---:|---:|---|---:|---:|
| BALANCED_MEAN_MINUS_UNWEIGHTED_MEAN | ROC_AUC | +0.003020 | +0.003003 | [-0.000862, +0.006691] | 0.0610 | 0.9390 |
| BALANCED_MEAN_MINUS_UNWEIGHTED_MEAN | AP | +0.002451 | +0.002470 | [-0.001055, +0.006042] | 0.0860 | 0.9140 |
| WINDOW_HGB_UNWEIGHTED_MEDIAN_MINUS_WINDOW_HGB_UNWEIGHTED_MEAN | ROC_AUC | -0.034004 | -0.033854 | [-0.045303, -0.022680] | 1.0000 | 0.0000 |
| WINDOW_HGB_UNWEIGHTED_MEDIAN_MINUS_WINDOW_HGB_UNWEIGHTED_MEAN | AP | -0.025576 | -0.025478 | [-0.034167, -0.017173] | 1.0000 | 0.0000 |
| WINDOW_HGB_UNWEIGHTED_Q90_MINUS_WINDOW_HGB_UNWEIGHTED_MEAN | ROC_AUC | +0.005948 | +0.005992 | [-0.002020, +0.014195] | 0.0725 | 0.9275 |
| WINDOW_HGB_UNWEIGHTED_Q90_MINUS_WINDOW_HGB_UNWEIGHTED_MEAN | AP | +0.023557 | +0.023583 | [+0.009649, +0.037568] | 0.0005 | 0.9995 |
| WINDOW_HGB_UNWEIGHTED_MAXIMUM_MINUS_WINDOW_HGB_UNWEIGHTED_MEAN | ROC_AUC | +0.005356 | +0.005630 | [-0.007761, +0.019594] | 0.2155 | 0.7845 |
| WINDOW_HGB_UNWEIGHTED_MAXIMUM_MINUS_WINDOW_HGB_UNWEIGHTED_MEAN | AP | +0.003954 | +0.005188 | [-0.015508, +0.027000] | 0.3205 | 0.6795 |
| WINDOW_HGB_CUSTOMER_BALANCED_MEDIAN_MINUS_WINDOW_HGB_CUSTOMER_BALANCED_MEAN | ROC_AUC | -0.032562 | -0.032467 | [-0.043936, -0.021454] | 1.0000 | 0.0000 |
| WINDOW_HGB_CUSTOMER_BALANCED_MEDIAN_MINUS_WINDOW_HGB_CUSTOMER_BALANCED_MEAN | AP | -0.026796 | -0.026565 | [-0.035300, -0.018156] | 1.0000 | 0.0000 |
| WINDOW_HGB_CUSTOMER_BALANCED_Q90_MINUS_WINDOW_HGB_CUSTOMER_BALANCED_MEAN | ROC_AUC | +0.004983 | +0.005039 | [-0.003556, +0.013983] | 0.1385 | 0.8615 |
| WINDOW_HGB_CUSTOMER_BALANCED_Q90_MINUS_WINDOW_HGB_CUSTOMER_BALANCED_MEAN | AP | +0.022705 | +0.022694 | [+0.009381, +0.036515] | 0.0005 | 0.9995 |
| WINDOW_HGB_CUSTOMER_BALANCED_MAXIMUM_MINUS_WINDOW_HGB_CUSTOMER_BALANCED_MEAN | ROC_AUC | +0.001652 | +0.002026 | [-0.013178, +0.017821] | 0.4050 | 0.5950 |
| WINDOW_HGB_CUSTOMER_BALANCED_MAXIMUM_MINUS_WINDOW_HGB_CUSTOMER_BALANCED_MEAN | AP | +0.001805 | +0.003283 | [-0.016950, +0.023329] | 0.3785 | 0.6215 |
| C_LOGIT_MINUS_W_LOGIT | ROC_AUC | +0.008627 | +0.008594 | [-0.000754, +0.018022] | 0.0370 | 0.9630 |
| C_LOGIT_MINUS_W_LOGIT | AP | +0.002121 | +0.002303 | [-0.013966, +0.017655] | 0.3955 | 0.6045 |

## Interpretation and decision

- Customer multiplicity clear material effect: **False**.
- Aggregation clear material effect in both HGB arms: **False**; qualifying rules: `[]`.
- Matched customer-level loss clear material advantage: **False**.

Decision **E — MIXED / INCONCLUSIVE**.

The frozen Stage 17B HGB DEV_VAL ROC-AUC/AP (`0.788831`/`0.359795`) is historical context only; it is not treated as a representation-matched Stage 20B comparison.

## Paper claim and next step

The pre-specified controls did not isolate a clear multiplicity, aggregation, or customer-loss mechanism; the formulation explanation remains inconclusive.

This is triangulation, not proof of temporal window-label noise. Freeze the AMP development experiment matrix.

No adaptive tuning, feature discovery, aggregation selection, threshold tuning, or post-hoc comparison invention occurred. The frozen Stage 17B detector was not modified.

**CONFIRMATION NOT ACCESSED DURING STAGE 20B.**

**FINAL_TEST NOT ACCESSED DURING STAGE 20B.**
