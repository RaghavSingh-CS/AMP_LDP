# Stage 5 — Validation-Based Autoencoder Threshold

## Scope and Stage 4 verification

The completed Stage 4 checkpoint loaded successfully with architecture
`30 → 16 → 8 → 4 → 8 → 16 → 30` and 1,362 trainable parameters. Training was
executed manually on CPU using 1,400,965
normal train windows. Early stopping occurred after 36 epochs;
the best qualifying checkpoint was epoch 31 with normal-validation
MSE 0.041186963864. Test data was not used.

## Validation reconstruction errors

| Population | Count | Mean | Median | Std | Min | P25 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAG=0 | 301,102 | 0.04118696 | 0.01768197 | 0.09573905 | 0.00005117 | 0.00578355 | 0.03952597 | 0.08993215 | 0.15245850 | 0.42514214 | 8.41493416 |
| FLAG=1 | 26,310 | 0.07360132 | 0.02058382 | 0.19733734 | 0.00004791 | 0.00952083 | 0.05227077 | 0.16470204 | 0.32115179 | 0.87486531 | 6.05808163 |

The theft-labelled/normal median ratio is
1.1641; the mean ratio is
1.7870. The score ranges overlap
substantially: 44.64%
of theft-labelled windows are at or below the normal median. FLAG is customer-level,
so these are weakly supervised window-level comparisons, not confirmed theft-event metrics.

Validation ROC-AUC is 0.563470; validation PR-AUC is
0.109945. These values describe ranking under weak customer labels only.

## Candidate thresholds

| Candidate | Threshold | Precision | Recall | F1 | Specificity | FPR | Predicted anomaly |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal_p90 | 0.0899321541 | 0.1254 | 0.1641 | 0.1422 | 0.9000 | 0.1000 | 10.52% |
| normal_p95 | 0.152458503 | 0.1584 | 0.1077 | 0.1282 | 0.9500 | 0.0500 | 5.46% |
| normal_p97.5 | 0.242344612 | 0.1902 | 0.0672 | 0.0993 | 0.9750 | 0.0250 | 2.84% |
| normal_p99 | 0.425142137 | 0.2430 | 0.0368 | 0.0639 | 0.9900 | 0.0100 | 1.22% |
| best_validation_f1 | 0.00269631739 | 0.0906 | 0.9123 | 0.1648 | 0.1996 | 0.8004 | 80.94% |
| best_validation_balanced_accuracy | 0.00269631739 | 0.0906 | 0.9123 | 0.1648 | 0.1996 | 0.8004 | 80.94% |

The best-F1 threshold (0.00269631739) attains F1
0.1648, but flags 80.94% of
all validation windows and has FPR 0.8004. The
best-balanced-accuracy candidate has balanced accuracy
0.5559, with similarly high false-positive volume.

## Selected baseline threshold

**Threshold:** `0.152458503097`  
**Method:** normal validation 95th percentile  
**Precision:** 0.158365  
**Recall:** 0.107678  
**F1:** 0.128193  
**Specificity:** 0.949997  
**False-positive rate:** 0.050003  
**Predicted anomaly rate:** 5.4638%

The normal-validation 95th percentile fixes the validation normal-window false-positive rate near 5% without fitting the cutoff to noisy customer-level theft labels. It avoids the impractical false-alarm volume of the best-F1 candidate while retaining more theft-labelled windows than normal p99. This is a validation-selected baseline, not a globally optimal or
business-cost-optimal threshold. Lower cutoffs catch more theft-labelled windows
but generate more false alarms; higher cutoffs reduce false alarms but miss more
theft-labelled windows.

## Customer-level diagnostic

Validation contains 5,813 normal and
542 theft-labelled customers. Customer
mean/max/p95 ROC-AUC values are 0.5713,
0.5611, and
0.5774; their PR-AUC values are
0.1417,
0.1242, and
0.1541. P95 aggregation improves separation
modestly relative to window-level ROC-AUC/PR-AUC, but separation remains weak and
does not justify a separate customer threshold in this stage.

## Missingness diagnostic

Missing fraction has weak negative association with reconstruction MSE:
overall Pearson -0.0284 and Spearman
-0.0585; FLAG=0 Pearson/Spearman are
-0.0285/-0.0572,
and FLAG=1 values are
-0.0413/-0.0772.
This does not indicate that the current Autoencoder score is primarily a proxy
for missing fraction, although missingness remains a potential confounder for
later sensitivity analysis.

## Boundary

No Autoencoder retraining, test-data access, test scoring, final test evaluation,
GAN work, or alternative-stride analysis was performed. The selected threshold
is now frozen in `models/autoencoder/ae_stride15_threshold.json` for later use.
