# Stage 19 — Customer Behavioural Feature-Family Ablation

## Frozen design and reproduction

Stage 19 classifies all 48 frozen Stage 16 predictors into six semantic families
before ablation scoring: LEVEL_DISTRIBUTION (18), VARIABILITY_ABRUPT_CHANGE (12),
DIRECTIONAL_TEMPORAL_CHANGE (8), LOW_CONSUMPTION_BEHAVIOUR (2),
PERSISTENCE_EXCEEDANCE (4), and LONGITUDINAL_CONTRAST (4). Every model uses the
unchanged unweighted Stage 17B HGB configuration. The global Stage 17B threshold
`0.20070617709942135` is used for secondary threshold metrics.

The FULL_48 control exactly reproduces Stage 17B: ROC-AUC `0.788830821895`
and AP `0.359794590322`. The maximum absolute probability
difference is recorded in `full_48_reproduction.json`; the stop gate passed before
any family ablation was fitted.

## Family-only models

| Family | Features | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEVEL_DISTRIBUTION | 18 | 0.767717 | 0.309965 | 0.343693 | 0.372277 | 0.357414 | 5067/359/317/188 |
| VARIABILITY_ABRUPT_CHANGE | 12 | 0.680289 | 0.201405 | 0.287009 | 0.188119 | 0.227273 | 5190/236/410/95 |
| DIRECTIONAL_TEMPORAL_CHANGE | 8 | 0.656386 | 0.171075 | 0.247191 | 0.130693 | 0.170984 | 5225/201/439/66 |
| LOW_CONSUMPTION_BEHAVIOUR | 2 | 0.591437 | 0.108022 | 0.078947 | 0.005941 | 0.011050 | 5391/35/502/3 |
| PERSISTENCE_EXCEEDANCE | 4 | 0.718677 | 0.205721 | 0.261719 | 0.265347 | 0.263520 | 5048/378/371/134 |
| LONGITUDINAL_CONTRAST | 4 | 0.693233 | 0.218358 | 0.270784 | 0.225743 | 0.246220 | 5119/307/391/114 |

| Family | Features | ROC-AUC delta | AP delta |
|---|---:|---:|---:|
| LEVEL_DISTRIBUTION | 18 | -0.021114 | -0.049829 |
| VARIABILITY_ABRUPT_CHANGE | 12 | -0.108542 | -0.158389 |
| DIRECTIONAL_TEMPORAL_CHANGE | 8 | -0.132444 | -0.188720 |
| LOW_CONSUMPTION_BEHAVIOUR | 2 | -0.197394 | -0.251772 |
| PERSISTENCE_EXCEEDANCE | 4 | -0.070153 | -0.154074 |
| LONGITUDINAL_CONTRAST | 4 | -0.095598 | -0.141437 |

The strongest standalone family by AP-first ordering is `LEVEL_DISTRIBUTION`.
LEVEL_DISTRIBUTION alone retains ROC-AUC 0.767717 and AP 0.309965, substantially
more than any other single family, but remains below FULL_48 by 0.021114 and
0.049829. PERSISTENCE_EXCEEDANCE, LONGITUDINAL_CONTRAST, and
VARIABILITY_ABRUPT_CHANGE each carry moderate standalone signal; directional
change is weaker, and the two low-consumption features alone are weak.

## Leave-one-family-out models

| Omitted family | Features retained | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEVEL_DISTRIBUTION | 30 | 0.763537 | 0.325818 | 0.325670 | 0.336634 | 0.331061 | 5074/352/335/170 |
| VARIABILITY_ABRUPT_CHANGE | 36 | 0.788703 | 0.341956 | 0.357013 | 0.388119 | 0.371917 | 5073/353/309/196 |
| DIRECTIONAL_TEMPORAL_CHANGE | 40 | 0.785885 | 0.343353 | 0.342857 | 0.380198 | 0.360563 | 5058/368/313/192 |
| LOW_CONSUMPTION_BEHAVIOUR | 46 | 0.787614 | 0.352943 | 0.353383 | 0.372277 | 0.362584 | 5082/344/317/188 |
| PERSISTENCE_EXCEEDANCE | 44 | 0.786127 | 0.350790 | 0.361266 | 0.384158 | 0.372361 | 5083/343/311/194 |
| LONGITUDINAL_CONTRAST | 44 | 0.779620 | 0.336558 | 0.364140 | 0.390099 | 0.376673 | 5082/344/308/197 |

| Omitted family | Features retained | ROC-AUC delta | AP delta |
|---|---:|---:|---:|
| LEVEL_DISTRIBUTION | 30 | -0.025294 | -0.033977 |
| VARIABILITY_ABRUPT_CHANGE | 36 | -0.000128 | -0.017838 |
| DIRECTIONAL_TEMPORAL_CHANGE | 40 | -0.002946 | -0.016442 |
| LOW_CONSUMPTION_BEHAVIOUR | 46 | -0.001217 | -0.006852 |
| PERSISTENCE_EXCEEDANCE | 44 | -0.002704 | -0.009005 |
| LONGITUDINAL_CONTRAST | 44 | -0.009211 | -0.023237 |

The largest AP-first observed leave-out loss is `LEVEL_DISTRIBUTION`.
Omitting LEVEL_DISTRIBUTION produces the largest losses, but the remaining five
families together still achieve ROC-AUC 0.763537 and AP 0.325818. The result is
therefore level-sensitive rather than narrowly level-dominated. Omitting
LONGITUDINAL_CONTRAST gives a practical observed AP loss of 0.023237; all other
leave-out losses are below the pre-registered practical thresholds.

## Selected paired uncertainty

| Omitted family | ROC-AUC mean delta [95% CI] | AP mean delta [95% CI] | ROC below/above 0 | AP below/above 0 |
|---|---:|---:|---:|---:|
| LEVEL_DISTRIBUTION | -0.025220 [-0.034223, -0.016166] | -0.033552 [-0.049842, -0.017956] | 1.000/0.000 | 1.000/0.001 |
| LONGITUDINAL_CONTRAST | -0.009363 [-0.020744, +0.002106] | -0.021740 [-0.046490, +0.002779] | 0.945/0.056 | 0.956/0.044 |

Selection followed the frozen union rule: largest ROC-AUC drop, largest AP drop,
and any family with observed ROC-AUC loss at least 0.01 or AP loss at least 0.02.
Each selected comparison uses 2,000 paired DEV_VAL customer replicates with seed 42.
The LEVEL_DISTRIBUTION intervals exclude zero for both metrics, making it the only
conditionally necessary family. Both LONGITUDINAL_CONTRAST intervals include zero,
so its observed loss is suggestive but not conclusive under the frozen rule.

## Redundancy, Stage 18 connection, and decision

The family interpretation table combines standalone strength with conditional
leave-out evidence. Existing Stage 16 correlations and Stage 17B permutation
importance are aggregated by the frozen family map; they are supporting evidence,
not another subset search. No compact model was pre-registered or constructed.
All 35 Stage 16 correlations at absolute Spearman at least 0.95 occur within
LEVEL_DISTRIBUTION (26), VARIABILITY_ABRUPT_CHANGE (7), or
DIRECTIONAL_TEMPORAL_CHANGE (2), supporting substantial within-family redundancy.
Stage 17B permutation importance is also concentrated in level/distribution, with
meaningful secondary contributions from longitudinal, variability, and persistence
features. Their small leave-out losses indicate that much of this information is
replaceable across families.

Stage 18 found stable ranking across history length but consumption-dependent
performance and weak LOW-consumption recall at the global threshold. Stage 19
therefore distinguishes dominance by absolute level from signal distributed across
variability, persistence, directional, low-consumption, and longitudinal behaviour.
It does not redesign the low-consumption operating point.
The necessary level family makes the Stage 18 consumption-stratum variation
plausible as model dependence on consumption distribution, but the useful
without-level result shows that variability, persistence, directional, and
longitudinal information collectively retain substantial discrimination. The
LOW_CONSUMPTION_BEHAVIOUR family is neither strong alone nor conditionally
necessary, so Stage 18's low-consumption recall limitation is not explained as
dependence on those two indicators alone.

Decision **B — SIGNAL IS STRONG BUT HIGHLY REDUNDANT ACROSS FAMILIES** follows the
pre-registered family-only, leave-out, practical-loss, and paired-uncertainty rules.
Strong standalone families: `['LEVEL_DISTRIBUTION']`.
Conditionally necessary families: `['LEVEL_DISTRIBUTION']`.

For the SGCC paper, Stage 19 identifies which behavioural information families
support the confirmed direct-customer discrimination. It does not establish
causality, state of the art, or external generalization.

Recommended Stage 20 question: **Does a label-formulation control that weakens propagated window-label assumptions preserve the direct customer-level advantage under DEV_TRAIN/DEV_VAL, while keeping CONFIRMATION and FINAL_TEST closed?**

**CONFIRMATION NOT ACCESSED DURING STAGE 19. FINAL_TEST NOT ACCESSED DURING STAGE 19.**
