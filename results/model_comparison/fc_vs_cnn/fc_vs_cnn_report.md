# Stage 10 — Frozen FC versus CNN Autoencoder

Both models use the same stride-30 windows, split lineage, normal-only training
policy, and normal-validation P95 threshold rule. This comparison reads saved
artifacts only.

| Metric | Fully connected | CNN | CNN − FC |
|---|---:|---:|---:|
| trainable_parameters | 1362.000000 | 2209.000000 | +847.000000 |
| epochs_executed | 35.000000 | 48.000000 | +13.000000 |
| best_epoch | 30.000000 | 43.000000 | +13.000000 |
| best_validation_mse | 0.043589 | 0.030209 | -0.013380 |
| training_runtime_seconds | 600.961045 | 1945.975401 | +1345.014356 |
| frozen_p95_threshold | 0.164363 | 0.104752 | -0.059611 |
| validation_precision | 0.159991 | 0.140163 | -0.019829 |
| validation_recall | 0.109064 | 0.093343 | -0.015720 |
| validation_F1 | 0.129707 | 0.112059 | -0.017648 |
| validation_specificity | 0.949995 | 0.949995 | +0.000000 |
| validation_false_positive_rate | 0.050005 | 0.050005 | +0.000000 |
| validation_predicted_anomaly_percentage | 5.474777 | 5.348524 | -0.126253 |
| validation_ROC_AUC | 0.564310 | 0.555699 | -0.008612 |
| validation_PR_AUC | 0.110242 | 0.101596 | -0.008646 |
| test_precision | 0.173693 | 0.153162 | -0.020531 |
| test_recall | 0.121059 | 0.104411 | -0.016647 |
| test_F1 | 0.142676 | 0.124173 | -0.018503 |
| test_specificity | 0.950607 | 0.950489 | -0.000118 |
| test_false_positive_rate | 0.049393 | 0.049511 | +0.000118 |
| test_balanced_accuracy | 0.535833 | 0.527450 | -0.008383 |
| test_ROC_AUC | 0.561688 | 0.555689 | -0.005999 |
| test_PR_AUC | 0.114411 | 0.104788 | -0.009623 |
| test_predicted_anomaly_percentage | 5.505393 | 5.384802 | -0.120591 |
| test_customer_mean_ROC_AUC | 0.591093 | 0.573850 | -0.017243 |
| test_customer_mean_PR_AUC | 0.157129 | 0.137504 | -0.019626 |
| test_customer_maximum_ROC_AUC | 0.587050 | 0.562926 | -0.024123 |
| test_customer_maximum_PR_AUC | 0.130813 | 0.116627 | -0.014186 |
| test_customer_p95_ROC_AUC | 0.599251 | 0.579486 | -0.019765 |
| test_customer_p95_PR_AUC | 0.168575 | 0.144033 | -0.024543 |

The CNN reduced best normal-validation reconstruction MSE by
**30.70%**, but required **3.24×** the
CPU training time and **1.62×**
the trainable parameters.

## Interpretation

CNN reconstructs normal validation windows better, but the frozen FC model is slightly stronger on every principal validation/test ranking and P95-threshold classification metric. The CNN does not improve theft detection enough to justify its larger parameter count and longer CPU training time as the preferred baseline.

- Normal reconstruction: **materially better** (30.70% lower validation MSE).
- Window-level anomaly discrimination: **worse**.
- Customer-level anomaly discrimination: **worse** for mean, maximum, and P95
  aggregation ROC-AUC and PR-AUC.
- Overall detection classification: **worse** than the FC baseline.

The fully connected model remains the preferred frozen baseline. This is not a
causal theft detector: `FLAG` is customer-level weak supervision, and positive
windows are not confirmed theft events.
