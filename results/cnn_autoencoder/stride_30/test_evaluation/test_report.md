# Stage 10 — Frozen CNN Test Evaluation

The frozen CNN checkpoint was evaluated once with validation P95 threshold
`0.104752443731` and rule `reconstruction_mse > threshold`. No retraining,
threshold adjustment, or test-driven selection occurred.

## Window-level results

Test windows: 165,020 (151,985 normal;
13,035 theft-labelled).

| Metric | Value |
|---|---:|
| TP / TN / FP / FN | 1,361 / 144,460 / 7,525 / 11,674 |
| Precision | 0.153162 |
| Recall | 0.104411 |
| F1 | 0.124173 |
| Specificity | 0.950489 |
| ROC-AUC | 0.555689 |
| PR-AUC | 0.104788 |
| Predicted anomaly percentage | 5.384802% |

## Validation versus test

Classification: **broadly stable**.

| Metric | Validation | Test | Absolute difference |
|---|---:|---:|---:|
| precision | 0.140163 | 0.153162 | 0.013000 |
| recall | 0.093343 | 0.104411 | 0.011068 |
| F1 | 0.112059 | 0.124173 | 0.012114 |
| specificity | 0.949995 | 0.950489 | 0.000493 |
| false_positive_rate | 0.050005 | 0.049511 | 0.000493 |
| ROC_AUC | 0.555699 | 0.555689 | 0.000009 |
| PR_AUC | 0.101596 | 0.104788 | 0.003192 |
| predicted_anomaly_percentage | 5.348524 | 5.384802 | 0.036278 |

Customer p95-MSE ROC-AUC is 0.579486.
The reconstruction-MSE/missing-fraction Spearman correlation is
-0.052406.

`FLAG` is customer-level weak supervision: positive windows are not confirmed
theft events. Customer aggregates remain diagnostics, not a tuned classifier.
