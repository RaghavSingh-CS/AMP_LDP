# Stage 12 — Frozen FC vs CNN vs LSTM Comparison

All models use the same fixed customer split, preprocessing/scaler, 30-day stride-30 usable windows, normal-only training, MSE/Adam protocol, validation-normal P95 cutoff method, and frozen-test methodology. The changed factor is architecture.

| Metric | FC | CNN | LSTM | CNN − FC | LSTM − FC |
|---|---:|---:|---:|---:|---:|
| trainable_parameters | 1362.000000 | 2209.000000 | 3177.000000 | +847.000000 | +1815.000000 |
| epochs_executed | 35.000000 | 48.000000 | 50.000000 | +13.000000 | +15.000000 |
| early_stopping_occurred | 1.000000 | 1.000000 | 0.000000 | +0.000000 | -1.000000 |
| best_epoch | 30.000000 | 43.000000 | 50.000000 | +13.000000 | +20.000000 |
| best_validation_mse | 0.043589 | 0.030209 | 0.037663 | -0.013380 | -0.005926 |
| training_runtime_seconds | 600.961045 | 1945.975401 | 2674.758465 | +1345.014356 | +2073.797420 |
| runtime_relative_to_fc | 1.000000 | 3.238106 | 4.450802 | +2.238106 | +3.450802 |
| frozen_p95_threshold | 0.164363 | 0.104752 | 0.133708 | -0.059611 | -0.030655 |
| validation_precision | 0.159991 | 0.140163 | 0.138509 | -0.019829 | -0.021483 |
| validation_recall | 0.109064 | 0.093343 | 0.092065 | -0.015720 | -0.016999 |
| validation_F1 | 0.129707 | 0.112059 | 0.110609 | -0.017648 | -0.019098 |
| validation_specificity | 0.949995 | 0.949995 | 0.949995 | +0.000000 | +0.000000 |
| validation_false_positive_rate | 0.050005 | 0.050005 | 0.050005 | +0.000000 | +0.000000 |
| validation_balanced_accuracy | 0.529529 | 0.521669 | 0.521030 | -0.007860 | -0.008499 |
| validation_accuracy | 0.882458 | 0.881196 | 0.881093 | -0.001263 | -0.001365 |
| validation_predicted_anomaly_percentage | 5.474777 | 5.348524 | 5.338255 | -0.126253 | -0.136522 |
| validation_ROC_AUC | 0.564310 | 0.555699 | 0.559887 | -0.008612 | -0.004423 |
| validation_PR_AUC | 0.110242 | 0.101596 | 0.103233 | -0.008646 | -0.007009 |
| test_precision | 0.173693 | 0.153162 | 0.144869 | -0.020531 | -0.028824 |
| test_recall | 0.121059 | 0.104411 | 0.096279 | -0.016647 | -0.024779 |
| test_F1 | 0.142676 | 0.124173 | 0.115679 | -0.018503 | -0.026997 |
| test_specificity | 0.950607 | 0.950489 | 0.951258 | -0.000118 | +0.000651 |
| test_false_positive_rate | 0.049393 | 0.049511 | 0.048742 | +0.000118 | -0.000651 |
| test_balanced_accuracy | 0.535833 | 0.527450 | 0.523769 | -0.008383 | -0.012064 |
| test_accuracy | 0.885081 | 0.883657 | 0.883723 | -0.001424 | -0.001357 |
| test_ROC_AUC | 0.561688 | 0.555689 | 0.557689 | -0.005999 | -0.003999 |
| test_PR_AUC | 0.114411 | 0.104788 | 0.104403 | -0.009623 | -0.010008 |
| test_predicted_anomaly_percentage | 5.505393 | 5.384802 | 5.249667 | -0.120591 | -0.255727 |
| test_customer_mean_mse_ROC_AUC | 0.591093 | 0.573850 | 0.575256 | -0.017243 | -0.015837 |
| test_customer_mean_mse_PR_AUC | 0.157129 | 0.137504 | 0.138774 | -0.019626 | -0.018355 |
| test_customer_maximum_mse_ROC_AUC | 0.587050 | 0.562926 | 0.564239 | -0.024123 | -0.022810 |
| test_customer_maximum_mse_PR_AUC | 0.130813 | 0.116627 | 0.120780 | -0.014186 | -0.010033 |
| test_customer_p95_mse_ROC_AUC | 0.599251 | 0.579486 | 0.580588 | -0.019765 | -0.018662 |
| test_customer_p95_mse_PR_AUC | 0.168575 | 0.144033 | 0.148500 | -0.024543 | -0.020076 |

## Architecture selection

FC is the selected reconstruction-model baseline. It is simplest (1,362 parameters), fastest (600.96 CPU seconds), and has the strongest frozen anomaly discrimination: test ROC-AUC/PR-AUC/F1 `0.561688` / `0.114411` / `0.142676`. It also leads every customer aggregate diagnostic.

CNN reconstructs normal windows best (validation MSE `0.030209`) but has weaker detection. LSTM reconstruction (`0.037663`) lies between FC and CNN, but LSTM does not improve frozen window-level or customer-level anomaly discrimination over FC and costs `4.45×` FC CPU runtime. All three models are weak under customer-level `FLAG` supervision.

`FLAG` is customer-level weak supervision; these results do not establish confirmed per-window theft detection.
