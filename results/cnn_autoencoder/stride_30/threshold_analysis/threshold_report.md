# Stage 10 — CNN Validation Threshold Analysis

The manually trained 2,209-parameter CNN checkpoint uses input `(B,1,30)`,
latent dimension 8, and the Stage 9 architecture. It ran on
CPU for 48 epochs;
epoch 43 is the best qualifying checkpoint with normal
validation MSE 0.030209245243. Test data was not accessed.

| Population | Count | Mean | Median | Std | Min | P25 | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 152,246 | 0.03020925 | 0.01340558 | 0.07404949 | 0.00003617 | 0.00435619 | 0.01340558 | 0.02910775 | 0.06221115 | 0.10475244 | 0.31949093 | 3.51421762 |
| Theft-labelled | 13,295 | 0.04811164 | 0.01510953 | 0.13716085 | 0.00004221 | 0.00711630 | 0.01510953 | 0.03576127 | 0.09737792 | 0.18329441 | 0.60749441 | 5.31885195 |

Theft/normal median and mean ratios are
1.1271 and
1.5926. Theft-labelled exceedance of
normal P90/P95/P97.5/P99 is
15.21% /
9.33% /
5.37% /
2.68%.

Validation ROC-AUC is 0.555699; PR-AUC is 0.101596.
FLAG is customer-level weak supervision.

## Candidate thresholds

| Candidate | Threshold | Precision | Recall | F1 | Specificity | FPR | Predicted anomaly |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal_p90 | 0.0622111522 | 0.1172 | 0.1521 | 0.1324 | 0.9000 | 0.1000 | 10.42% |
| normal_p95 | 0.104752444 | 0.1402 | 0.0933 | 0.1121 | 0.9500 | 0.0500 | 5.35% |
| normal_p97.5 | 0.172095137 | 0.1579 | 0.0537 | 0.0802 | 0.9750 | 0.0250 | 2.73% |
| normal_p99 | 0.319490932 | 0.1895 | 0.0268 | 0.0469 | 0.9900 | 0.0100 | 1.14% |
| best_validation_f1 | 0.00215843553 | 0.0906 | 0.9081 | 0.1648 | 0.2041 | 0.7959 | 80.49% |
| best_validation_balanced_accuracy | 0.0018198858 | 0.0905 | 0.9145 | 0.1648 | 0.1978 | 0.8022 | 81.12% |

The authoritative threshold is normal-validation P95:
`0.104752443731`. Precision/recall/F1 are
0.140163/0.093343/0.112059;
specificity is 0.949995, FPR is
0.050005, balanced accuracy is
0.521669, and anomaly rate is
5.3485%. Best-F1 and
best-balanced-accuracy candidates are diagnostic only.

Customer mean/max/P95 ROC-AUC values are
0.5582/
0.5464/
0.5603. Overall
missingness Pearson/Spearman values are
-0.0148/-0.0516.
No retraining, test scoring, or test-driven selection occurred.
