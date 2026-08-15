# Stage 12 — Frozen LSTM Test Evaluation

The fixed normal-validation P95 threshold `0.133708283305` was applied once using `reconstruction_mse > threshold`. No retraining, threshold adjustment, or test-driven selection occurred.

| Metric | Value |
|---|---:|
| TP / TN / FP / FN | 1,255 / 144,577 / 7,408 / 11,780 |
| Precision | 0.144869 |
| Recall | 0.096279 |
| F1 | 0.115679 |
| Specificity | 0.951258 |
| FPR | 0.048742 |
| ROC-AUC / PR-AUC | 0.557689 / 0.104403 |

Validation-to-test classification: **broadly stable**.

| Metric | Validation | Test | Absolute difference |
|---|---:|---:|---:|
| precision | 0.138509 | 0.144869 | 0.006360 |
| recall | 0.092065 | 0.096279 | 0.004215 |
| F1 | 0.110609 | 0.115679 | 0.005070 |
| specificity | 0.949995 | 0.951258 | 0.001263 |
| false_positive_rate | 0.050005 | 0.048742 | 0.001263 |
| ROC_AUC | 0.559887 | 0.557689 | 0.002198 |
| PR_AUC | 0.103233 | 0.104403 | 0.001169 |
| predicted_anomaly_percentage | 5.338255 | 5.249667 | 0.088588 |

Customer P95 ROC-AUC/PR-AUC: `0.580588` / `0.148500`. Overall MSE/missing-fraction Spearman: `-0.047798`. `FLAG` is customer-level weak supervision, not confirmed per-window theft.
