# Stage 5 — Validation-Based Autoencoder Threshold

## Scope and Stage 4 verification

The completed Stage 4 checkpoint loaded successfully with architecture
`30 → 16 → 8 → 4 → 8 → 16 → 30` and 1,362 trainable parameters. Training was
executed manually on CPU using 708,308
normal train windows. Early stopping occurred after 35 epochs;
the best qualifying checkpoint was epoch 30 with normal-validation
MSE 0.043589054377. Test data was not used.

## Validation reconstruction errors

| Population | Count | Mean | Median | Std | Min | P25 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAG=0 | 152,246 | 0.04358905 | 0.01833776 | 0.09866227 | 0.00004717 | 0.00597436 | 0.04176540 | 0.09662897 | 0.16436323 | 0.45554479 | 4.28616428 |
| FLAG=1 | 13,295 | 0.07767822 | 0.02167252 | 0.20311640 | 0.00005608 | 0.00979988 | 0.05579439 | 0.17806693 | 0.35174631 | 0.91563835 | 5.97542381 |

The theft-labelled/normal median ratio is
1.1819; the mean ratio is
1.7821. The score ranges overlap
substantially: 44.39%
of theft-labelled windows are at or below the normal median. FLAG is customer-level,
so these are weakly supervised window-level comparisons, not confirmed theft-event metrics.

Validation ROC-AUC is 0.564310; validation PR-AUC is
0.110242. These values describe ranking under weak customer labels only.

## Candidate thresholds

| Candidate | Threshold | Precision | Recall | F1 | Specificity | FPR | Predicted anomaly |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal_p90 | 0.0966289714 | 0.1263 | 0.1656 | 0.1433 | 0.9000 | 0.1000 | 10.53% |
| normal_p95 | 0.164363232 | 0.1600 | 0.1091 | 0.1297 | 0.9500 | 0.0500 | 5.47% |
| normal_p97.5 | 0.258989345 | 0.1915 | 0.0678 | 0.1002 | 0.9750 | 0.0250 | 2.84% |
| normal_p99 | 0.455544786 | 0.2423 | 0.0366 | 0.0636 | 0.9900 | 0.0100 | 1.21% |
| best_validation_f1 | 0.00324803707 | 0.0907 | 0.9064 | 0.1650 | 0.2070 | 0.7930 | 80.21% |
| best_validation_balanced_accuracy | 0.00316531118 | 0.0907 | 0.9076 | 0.1650 | 0.2058 | 0.7942 | 80.33% |

The best-F1 threshold (0.00324803707) attains F1
0.1650, but flags 80.21% of
all validation windows and has FPR 0.7930. The
best-balanced-accuracy candidate has balanced accuracy
0.5567, with similarly high false-positive volume.

## Selected baseline threshold

**Threshold:** `0.16436323151`  
**Method:** normal validation 95th percentile  
**Precision:** 0.159991  
**Recall:** 0.109064  
**F1:** 0.129707  
**Specificity:** 0.949995  
**False-positive rate:** 0.050005  
**Predicted anomaly rate:** 5.4748%

The normal-validation 95th percentile fixes the validation normal-window false-positive rate near 5% without fitting the cutoff to noisy customer-level theft labels. It avoids the impractical false-alarm volume of the best-F1 candidate while retaining more theft-labelled windows than normal p99. This is a validation-selected baseline, not a globally optimal or
business-cost-optimal threshold. Lower cutoffs catch more theft-labelled windows
but generate more false alarms; higher cutoffs reduce false alarms but miss more
theft-labelled windows.

## Customer-level diagnostic

Validation contains 5,813 normal and
542 theft-labelled customers. Customer
mean/max/p95 ROC-AUC values are 0.5751,
0.5658, and
0.5781; their PR-AUC values are
0.1428,
0.1248, and
0.1550. P95 aggregation improves separation
modestly relative to window-level ROC-AUC/PR-AUC, but separation remains weak and
does not justify a separate customer threshold in this stage.

## Missingness diagnostic

Missing fraction has weak negative association with reconstruction MSE:
overall Pearson -0.0140 and Spearman
-0.0417; FLAG=0 Pearson/Spearman are
-0.0130/-0.0405,
and FLAG=1 values are
-0.0297/-0.0594.
This does not indicate that the current Autoencoder score is primarily a proxy
for missing fraction, although missingness remains a potential confounder for
later sensitivity analysis.

## Boundary

No Autoencoder retraining, test-data access, test scoring, final test evaluation,
GAN work, or alternative-stride analysis was performed. The selected threshold
is now frozen in `models/autoencoder/ae_stride30_threshold.json` for later use.
