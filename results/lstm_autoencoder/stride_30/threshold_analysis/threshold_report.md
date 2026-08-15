# Stage 12 — LSTM Validation Threshold Analysis

The manually trained LSTM uses `(B,30,1)`, latent dimension 8, and 3,177 parameters. It executed the predefined 50 epochs; epoch 50 retained the best checkpoint at normal-validation MSE `0.037662720912`. The maximum was not extended. Test data was not accessed.

| Population | Count | Mean | Median | Std | Min | P25 | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 152,246 | 0.03766272 | 0.01661586 | 0.09662068 | 0.00005016 | 0.00548498 | 0.01661586 | 0.03646705 | 0.07929339 | 0.13370828 | 0.38954343 | 14.10442162 |
| Theft-labelled | 13,295 | 0.06181643 | 0.01924833 | 0.17827807 | 0.00006678 | 0.00910366 | 0.01924833 | 0.04460334 | 0.12243102 | 0.24178034 | 0.77634470 | 6.03463745 |

Theft/normal median and mean ratios: `1.1584` / `1.6413`. Theft-labelled windows above normal P90/P95/P97.5/P99: `14.99%` / `9.21%` / `5.51%` / `3.02%`.

Validation ROC-AUC `0.559887`; PR-AUC `0.103233`. `FLAG` is customer-level weak supervision.

## Candidate thresholds

| Candidate | Threshold | Precision | Recall | F1 | Specificity | FPR | Predicted anomaly |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal_p90 | 0.0792933926 | 0.1158 | 0.1499 | 0.1306 | 0.9000 | 0.1000 | 10.40% |
| normal_p95 | 0.133708283 | 0.1385 | 0.0921 | 0.1106 | 0.9500 | 0.0500 | 5.34% |
| normal_p97.5 | 0.215146605 | 0.1613 | 0.0551 | 0.0821 | 0.9750 | 0.0250 | 2.74% |
| normal_p99 | 0.389543432 | 0.2088 | 0.0302 | 0.0528 | 0.9900 | 0.0100 | 1.16% |
| best_validation_f1 | 0.00352513185 | 0.0910 | 0.9001 | 0.1653 | 0.2146 | 0.7854 | 79.46% |
| best_validation_balanced_accuracy | 0.00285657449 | 0.0909 | 0.9105 | 0.1652 | 0.2045 | 0.7955 | 80.48% |

The frozen threshold is normal-validation P95: `0.133708283305`. Best-F1 and best-balanced-accuracy candidates are diagnostic only.
