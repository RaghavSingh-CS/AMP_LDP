# Stage 18 — Customer-Stratum Robustness

## Frozen analysis

Stage 18 reuses the exact saved Stage 17B HGB `DEV_VAL` probabilities and frozen
threshold `0.20070617709942135`. No model was retrained or rescored. The 48
features, model hash, prediction hash, and governance hash were verified before
analysis. `CONFIRMATION` and `FINAL_TEST` were not accessed.

History uses a deterministic tie-aware approximation to tertiles that never splits
an observed integer window-count tie: `SHORT <= 18`, `MEDIUM 19–33`, and `LONG = 34`
usable windows. Consumption uses the existing Stage 16 `mean__customer_median`
feature: `LOW <= -0.3365242282549541`, `MEDIUM` above that through
`0.4767419298489887`, and `HIGH` above the latter cutpoint. Boundary values are
assigned to the lower group. Both definitions came only from `DEV_TRAIN`.

| Dimension | Stratum | DEV_TRAIN customers | Normal | Theft | Prevalence |
|---|---|---:|---:|---:|---:|
| History | SHORT | 6,450 | 5,752 | 698 | 0.1082 |
| History | MEDIUM | 6,503 | 5,820 | 683 | 0.1050 |
| History | LONG | 10,771 | 10,130 | 641 | 0.0595 |
| Consumption | LOW | 7,908 | 7,476 | 432 | 0.0546 |
| Consumption | MEDIUM | 7,908 | 7,261 | 647 | 0.0818 |
| Consumption | HIGH | 7,908 | 6,965 | 943 | 0.1192 |

## History-length robustness

| Stratum | Customers | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SHORT | 1,617 | 167 | 0.1033 | 0.7664 | 0.3383 | 0.3407 | 0.3713 | 0.3553 | 1330/120/105/62 |
| MEDIUM | 1,639 | 166 | 0.1013 | 0.8006 | 0.3806 | 0.3757 | 0.3916 | 0.3835 | 1365/108/101/65 |
| LONG | 2,675 | 172 | 0.0643 | 0.7851 | 0.3696 | 0.3687 | 0.3837 | 0.3761 | 2390/113/106/66 |

ROC-AUC range is `0.0341` and AP range is
`0.0423`. Worst frozen-threshold recall is
`0.3713` in `SHORT`.
All history groups retain useful ranking, their ROC-AUC intervals overlap, and
`SHORT` AP is still 3.28 times its prevalence. The aggregate result is therefore
not dependent on nearly complete histories.

## Consumption-level robustness

| Stratum | Customers | Theft | Prevalence | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LOW | 2,006 | 116 | 0.0578 | 0.7601 | 0.2126 | 0.2899 | 0.1724 | 0.2162 | 1841/49/96/20 |
| MEDIUM | 1,983 | 143 | 0.0721 | 0.7140 | 0.2493 | 0.2815 | 0.2657 | 0.2734 | 1743/97/105/38 |
| HIGH | 1,942 | 246 | 0.1267 | 0.8319 | 0.4892 | 0.4091 | 0.5488 | 0.4688 | 1501/195/111/135 |

ROC-AUC range is `0.1179` and AP range is
`0.2767`. Worst frozen-threshold recall is
`0.1724` in `LOW`.
AP lift over prevalence and bootstrap intervals are preserved in the machine-readable
tables; prevalence-normalized AP distinguishes ranking changes from base-rate shifts.
Raw AP differences are substantially influenced by prevalence: AP/prevalence is
3.68, 3.46, and 3.86 for `LOW`, `MEDIUM`, and `HIGH`, respectively. Nevertheless,
`HIGH` has materially stronger ROC-AUC than `MEDIUM`, with non-overlapping percentile
intervals. `LOW` recall is only 0.1724 at the unchanged global threshold, below the
pre-registered Decision A floor; this operating-point limitation drives Decision B.

## Bootstrap uncertainty

| Dimension/stratum | ROC-AUC 95% percentile CI | AP 95% percentile CI | Valid/attempted |
|---|---:|---:|---:|
| History SHORT | [0.7269, 0.8042] | [0.2746, 0.4140] | 2,000/2,000 |
| History MEDIUM | [0.7658, 0.8367] | [0.3088, 0.4569] | 2,000/2,000 |
| History LONG | [0.7468, 0.8232] | [0.2973, 0.4461] | 2,000/2,000 |
| Consumption LOW | [0.7138, 0.8030] | [0.1473, 0.2924] | 2,000/2,000 |
| Consumption MEDIUM | [0.6710, 0.7580] | [0.1861, 0.3245] | 2,000/2,000 |
| Consumption HIGH | [0.8040, 0.8571] | [0.4240, 0.5523] | 2,000/2,000 |

Every primary group produced all 2,000 valid customer-level replicates; no
single-class bootstrap sample was discarded.

## Intersections, failure modes, and formulation comparison

All nine pre-registered history × consumption counts are reported. Metrics are
shown only where the frozen minimum of 100 customers, 30 positives, and 30 negatives
is met. No additional subgroup search was performed. The lowest history ROC-AUC is
`0.7664` in `SHORT`; the lowest consumption
ROC-AUC is `0.7140` in `MEDIUM`.
Descriptive medians of history, typical consumption, variability, abrupt change,
and low-consumption fraction are reported without feature engineering.
The `LOW`-consumption group also has much lower median window variability (0.0775),
lower abrupt change (0.0452), and a much larger median low-consumption fraction
(0.7333) than the other consumption groups. These are descriptive correlates, not
causal explanations or prompts for model changes. All nine intersection cells met
the pre-registered count threshold, but they were not bootstrapped and are treated
as secondary descriptions rather than a subgroup search.

The Stage 14 within-stratum comparison was skipped before scoring. Its model was
trained on all original-train customers, including current `DEV_VAL`, so applying
it here would be in-sample and incompatible with Stage 17A. No leakage-safe saved
Stage 14 DEV_VAL predictions exist; compatibility was not fabricated.

## Decision and paper interpretation

Decision **B — GENERALLY ROBUST WITH A MATERIAL SUBGROUP LIMITATION** follows the pre-registered minimum
ROC-AUC, prevalence-normalized AP, recall, and between-group range rules. The frozen
detector was not changed.

Within SGCC `DEV_VAL`, the frozen direct-customer model retains useful discrimination
in every primary stratum, but consumption level matters: ranking is weaker in the
middle group and frozen-threshold recall is low in the low-consumption group. Because
the leakage-safe Stage 14 subgroup comparator does not exist, Stage 18 cannot support
the stronger comparative statement that direct modeling *improves* discrimination
within every group. It supports the narrower statement that direct-customer
discrimination is not confined to long-history or high-consumption customers.

The recommended Stage 19 question is: **Which frozen customer-feature groups and
aggregation families are necessary for the confirmed direct-customer HGB
discrimination under leakage-safe DEV_TRAIN/DEV_VAL ablation?**

**CONFIRMATION NOT ACCESSED DURING STAGE 18. FINAL_TEST NOT ACCESSED DURING STAGE 18.**
