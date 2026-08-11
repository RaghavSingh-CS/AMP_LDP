# Stage 6 — Frozen Test Evaluation

## Frozen evaluation policy

The Stage 4 checkpoint was evaluated once using threshold `0.16436323151`
loaded from `models/autoencoder/ae_stride30_threshold.json`. The rule was
`reconstruction_mse > threshold`. No retraining, threshold adjustment,
architecture change, or test-driven selection occurred.

## Window-level test results

Test windows: 165,020 (151,985 normal;
13,035 theft-labelled).

| Metric | Value |
|---|---:|
| TP | 1,578 |
| TN | 144,478 |
| FP | 7,507 |
| FN | 11,457 |
| Precision | 0.173693 |
| Recall | 0.121059 |
| F1 | 0.142676 |
| Specificity | 0.950607 |
| False-positive rate | 0.049393 |
| Balanced accuracy | 0.535833 |
| Accuracy | 0.885081 |
| Predicted anomaly rate | 5.5054% |
| ROC-AUC | 0.561688 |
| PR-AUC | 0.114411 |

FLAG is customer-level. These are weakly supervised window-label metrics, not
confirmed theft-event detection metrics.

## Reconstruction-error distributions

| Population | Count | Mean | Median | Std | Min | P25 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAG=0 | 151,985 | 0.04277214 | 0.01824364 | 0.09207726 | 0.00004665 | 0.00620825 | 0.04179378 | 0.09718951 | 0.16287254 | 0.41991245 | 3.91847777 |
| FLAG=1 | 13,035 | 0.08020520 | 0.02127300 | 0.18537265 | 0.00004985 | 0.00939446 | 0.05705905 | 0.20541558 | 0.38818884 | 0.88286492 | 4.22015142 |

The theft/normal median ratio is
1.1661; the mean ratio is
1.8752. Theft-labelled exceedance is
12.11% at the
frozen threshold. Diagnostic exceedance of normal test p90/p95/p99 is
17.47% /
12.18% /
4.59%. Test
percentiles were not used to change the threshold.

## Validation versus test

| Metric | Validation | Test | Absolute difference |
|---|---:|---:|---:|
| precision | 0.159991 | 0.173693 | 0.013702 |
| recall | 0.109064 | 0.121059 | 0.011995 |
| F1 | 0.129707 | 0.142676 | 0.012969 |
| specificity | 0.949995 | 0.950607 | 0.000612 |
| false_positive_rate | 0.050005 | 0.049393 | 0.000612 |
| ROC_AUC | 0.564310 | 0.561688 | 0.002623 |
| PR_AUC | 0.110242 | 0.114411 | 0.004169 |
| predicted_anomaly_percentage | 5.474777 | 5.505393 | 0.030617 |

Generalization is classified as **broadly stable** under the prespecified
absolute-difference checks. The model and threshold remain unchanged.

## Customer and missingness diagnostics

Test contains 5,814 normal and
542 theft-labelled customers. Customer p95
aggregation has ROC-AUC 0.599251 and PR-AUC
0.168575. No customer classifier or customer cutoff was
created. Mean anomalous-window fractions are
0.0482
for normal customers and
0.1102
for theft-labelled customers.

Missing-fraction correlation with reconstruction MSE is weak: overall
Pearson/Spearman -0.0153/
-0.0418, normal
-0.0133/-0.0399,
and theft-labelled -0.0365/
-0.0627.

## Interpretation and boundary

This frozen fully connected stride-30 Autoencoder is classified after test
evaluation according to its ranking, F1, recall, false-positive rate, and
validation/test consistency in `baseline_final_summary.json`. It establishes a
rigorous baseline for future models. No stride comparison, GAN, ensemble,
dashboard, or deployment work was performed.
