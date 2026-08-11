# Stage 8 — Stride-30 vs Stride-15 Autoencoder Comparison

## Controlled design

The customer split, preprocessing, train-fitted scaler, 30-day window length,
usable-window rule, architecture, loss, optimizer, learning rate, batch size,
early stopping, seed, normal-validation P95 threshold method, and frozen-test
method are fixed. Only stride changes from 30 to 15. FLAG remains weak
customer-level supervision.

Stride 15 uses 67 windows/customer versus 34 for stride 30 and increases usable
training windows from 708,308 to 1,520,861. CPU training runtime changes from
601.0s to 1251.5s.

## Reconstruction versus detection

Best normal-validation MSE decreases from 0.04358905 to
0.04118696. This lower reconstruction loss does **not** translate into better
window-level anomaly discrimination: test ROC-AUC changes by
-0.000611, PR-AUC by
-0.000816, and F1 by
-0.001336.

## Frozen test comparison

| Metric | Stride 30 | Stride 15 | Δ 15−30 |
|---|---:|---:|---:|
| precision | 0.173693 | 0.172023 | -0.001670 |
| recall | 0.121059 | 0.119946 | -0.001113 |
| F1 | 0.142676 | 0.141340 | -0.001336 |
| specificity | 0.950607 | 0.950464 | -0.000143 |
| false_positive_rate | 0.049393 | 0.049536 | +0.000143 |
| balanced_accuracy | 0.535833 | 0.535205 | -0.000628 |
| ROC_AUC | 0.561688 | 0.561077 | -0.000611 |
| PR_AUC | 0.114411 | 0.113595 | -0.000816 |
| predicted_anomaly_percentage | 5.505393 | 5.509978 | +0.004584 |

## Customer-level test comparison

| Aggregation | Metric | Stride 30 | Stride 15 | Δ 15−30 |
|---|---|---:|---:|---:|
| mean_mse | ROC_AUC | 0.591093 | 0.587743 | -0.003350 |
| mean_mse | PR_AUC | 0.157129 | 0.155437 | -0.001692 |
| maximum_mse | ROC_AUC | 0.587050 | 0.580410 | -0.006639 |
| maximum_mse | PR_AUC | 0.130813 | 0.130475 | -0.000339 |
| p95_mse | ROC_AUC | 0.599251 | 0.600611 | +0.001360 |
| p95_mse | PR_AUC | 0.168575 | 0.171796 | +0.003220 |

P95 customer PR-AUC improves by
+0.003220, but the other
customer-level changes are small and mixed. No customer threshold is selected.

## Missingness diagnostic

Overall test Pearson/Spearman correlations are
-0.0153/-0.0418
for stride 30 and
-0.0330/-0.0597
for stride 15. Both remain weak; stride 15 is modestly more negative.

## Conclusion and stride-7 recommendation

Stride 15 is **effectively equivalent and slightly worse at the window level**,
not a meaningful improvement over stride 30. The small P95 customer PR-AUC gain
does not offset lower window ROC-AUC, PR-AUC, F1, and recall.

Do **not** execute stride 7 next under the current fully connected baseline.
Stride 15 approximately doubles window volume and materially increases CPU time
without improving core detection. Stride 7 would produce 144 windows/customer,
about 2.15× the stride-15 volume, while current evidence gives no reason to
expect a worthwhile anomaly-discrimination gain. It remains unexecuted and may
be reconsidered only after a model or hypothesis that can exploit denser overlap.
