# Stage 6 — Frozen Test Evaluation

## Frozen evaluation policy

The Stage 4 checkpoint was evaluated once using threshold `0.152458503097`
loaded from `models/autoencoder/ae_stride15_threshold.json`. The rule was
`reconstruction_mse > threshold`. No retraining, threshold adjustment,
architecture change, or test-driven selection occurred.

## Window-level test results

Test windows: 326,426 (300,631 normal;
25,795 theft-labelled).

| Metric | Value |
|---|---:|
| TP | 3,094 |
| TN | 285,739 |
| FP | 14,892 |
| FN | 22,701 |
| Precision | 0.172023 |
| Recall | 0.119946 |
| F1 | 0.141340 |
| Specificity | 0.950464 |
| False-positive rate | 0.049536 |
| Balanced accuracy | 0.535205 |
| Accuracy | 0.884835 |
| Predicted anomaly rate | 5.5100% |
| ROC-AUC | 0.561077 |
| PR-AUC | 0.113595 |

FLAG is customer-level. These are weakly supervised window-label metrics, not
confirmed theft-event detection metrics.

## Reconstruction-error distributions

| Population | Count | Mean | Median | Std | Min | P25 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAG=0 | 300,631 | 0.04035934 | 0.01758733 | 0.08758348 | 0.00004987 | 0.00602300 | 0.03967276 | 0.09032292 | 0.15158169 | 0.39827665 | 4.57640076 |
| FLAG=1 | 25,795 | 0.07538944 | 0.02038566 | 0.17798298 | 0.00005810 | 0.00919900 | 0.05338673 | 0.18769270 | 0.36041782 | 0.87120325 | 4.21635628 |

The theft/normal median ratio is
1.1591; the mean ratio is
1.8680. Theft-labelled exceedance is
11.99% at the
frozen threshold. Diagnostic exceedance of normal test p90/p95/p99 is
17.39% /
12.04% /
4.40%. Test
percentiles were not used to change the threshold.

## Validation versus test

| Metric | Validation | Test | Absolute difference |
|---|---:|---:|---:|
| precision | 0.158365 | 0.172023 | 0.013657 |
| recall | 0.107678 | 0.119946 | 0.012268 |
| F1 | 0.128193 | 0.141340 | 0.013147 |
| specificity | 0.949997 | 0.950464 | 0.000467 |
| false_positive_rate | 0.050003 | 0.049536 | 0.000467 |
| ROC_AUC | 0.563470 | 0.561077 | 0.002393 |
| PR_AUC | 0.109945 | 0.113595 | 0.003649 |
| predicted_anomaly_percentage | 5.463758 | 5.509978 | 0.046220 |

Generalization is classified as **broadly stable** under the prespecified
absolute-difference checks. The model and threshold remain unchanged.

## Customer and missingness diagnostics

Test contains 5,814 normal and
542 theft-labelled customers. Customer p95
aggregation has ROC-AUC 0.600611 and PR-AUC
0.171796. No customer classifier or customer cutoff was
created. Mean anomalous-window fractions are
0.0479
for normal customers and
0.1088
for theft-labelled customers.

Missing-fraction correlation with reconstruction MSE is weak: overall
Pearson/Spearman -0.0330/
-0.0597, normal
-0.0328/-0.0577,
and theft-labelled -0.0482/
-0.0807.

## Interpretation and boundary

This frozen fully connected stride-15 Autoencoder is classified after test
evaluation according to its ranking, F1, recall, false-positive rate, and
validation/test consistency in `baseline_final_summary.json`. It establishes a
rigorous baseline for future models. No stride comparison, GAN, ensemble,
dashboard, or deployment work was performed.
