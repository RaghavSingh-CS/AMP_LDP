# Stage 22 — Pre-Registered Compact Representation

## Question and governance

Stage 22 tests whether deterministic, label-free within-family redundancy removal can simplify the frozen 48-feature customer representation without material DEV_VAL discrimination loss. The subset was frozen before compact scoring in `results/supervised/stage_22/preregistered_compact_representation.json`, SHA-256 `2b47224cbadfeaebd2a5f27b747e80fe512735ce1185323b92e4ce57dd376f27`.

Only DEV_TRAIN feature values determined the subset. Labels, importance, DEV_VAL performance, cross-family correlations, and protected populations played no role.

## Compaction rule

Within each frozen Stage 19 family, absolute DEV_TRAIN Spearman correlations at least `0.95` form edges in an undirected graph. Each connected component retains its first feature in frozen Stage 16 order. Singleton components remain. No pruning crosses family boundaries.

The rule retained **26 of 48 features**, a **45.83% reduction**. The pre-scoring requirement of at most 36 retained features passed.

| Family | Original | Retained | Dropped | Retained features | Dropped features |
|---|---:|---:|---:|---|---|
| LEVEL_DISTRIBUTION | 18 | 3 | 15 | `mean__customer_median`, `mean__customer_q10`, `mean__customer_q90` | `median__customer_median`, `median__customer_q10`, `median__customer_q90`, `minimum__customer_median`, `minimum__customer_q10`, `minimum__customer_q90`, `maximum__customer_median`, `maximum__customer_q10`, `maximum__customer_q90`, `q10__customer_median`, `q10__customer_q10`, `q10__customer_q90`, `q90__customer_median`, `q90__customer_q10`, `q90__customer_q90` |
| VARIABILITY_ABRUPT_CHANGE | 12 | 7 | 5 | `standard_deviation__customer_median`, `standard_deviation__customer_q90`, `interquartile_range__customer_median`, `interquartile_range__customer_q90`, `mean_absolute_consecutive_change__customer_median`, `mean_absolute_consecutive_change__customer_q90`, `maximum_absolute_consecutive_change__customer_q90` | `range__customer_median`, `range__customer_q90`, `mean_absolute_deviation__customer_median`, `mean_absolute_deviation__customer_q90`, `maximum_absolute_consecutive_change__customer_median` |
| DIRECTIONAL_TEMPORAL_CHANGE | 8 | 6 | 2 | `first_to_last_change__customer_mean`, `first_to_last_change__customer_standard_deviation`, `absolute_first_to_last_change__customer_mean`, `linear_slope__customer_mean`, `linear_slope__customer_standard_deviation`, `second_half_minus_first_half_mean__customer_mean` | `absolute_first_to_last_change__customer_standard_deviation`, `second_half_minus_first_half_mean__customer_standard_deviation` |
| LOW_CONSUMPTION_BEHAVIOUR | 2 | 2 | 0 | `fraction_below_minus_one__customer_mean`, `fraction_below_minus_one__customer_q90` | — |
| PERSISTENCE_EXCEEDANCE | 4 | 4 | 0 | `maximum__training_q90_exceedance_fraction`, `standard_deviation__training_q90_exceedance_fraction`, `mean_absolute_consecutive_change__training_q90_exceedance_fraction`, `maximum_absolute_consecutive_change__training_q90_exceedance_fraction` | — |
| LONGITUDINAL_CONTRAST | 4 | 4 | 0 | `mean__late_third_minus_early_third`, `maximum__late_third_minus_early_third`, `standard_deviation__late_third_minus_early_third`, `mean_absolute_consecutive_change__late_third_minus_early_third` | — |

Every semantic family remains represented: **True**.

## FULL_48 reproduction

The full model passed before COMPACT was interpreted. Maximum probability difference from frozen Stage 17B was `9.71e-17`; ROC-AUC/AP differences were `+0`/`+0`.

## Metrics

Both models use the exact frozen Stage 17B HGB configuration and threshold `0.20070617709942135`. Threshold metrics are descriptive.

| Model | Features | ROC-AUC | AP | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---|
| FULL_48 | 48 | 0.788831 | 0.359795 | 0.362617 | 0.384158 | 0.373077 | 5085/341/311/194 |
| COMPACT | 26 | 0.776787 | 0.337326 | 0.357009 | 0.378218 | 0.367308 | 5082/344/314/191 |

COMPACT minus FULL_48 observed ROC-AUC/AP deltas are `-0.012044`/`-0.022468`.

## Paired uncertainty

Two thousand paired DEV_VAL customer bootstrap replicates used seed 42.

| Metric | Observed delta | Mean delta | 95% interval | Fraction below 0 | Fraction above 0 |
|---|---:|---:|---|---:|---:|
| ROC-AUC | -0.012044 | -0.012108 | [-0.019878, -0.004148] | 0.9995 | 0.0005 |
| AP | -0.022468 | -0.021840 | [-0.035788, -0.007972] | 0.9990 | 0.0010 |

## Complexity

| Model | Features | Reduction | Serialized bytes | Fit seconds | Inference seconds | Trees | Leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL_48 | 48 | 0.00% | 274208 | 1.553145 | 0.016014 | 100 | 1500 |
| COMPACT | 26 | 45.83% | 230240 | 0.732888 | 0.020922 | 100 | 1500 |

## Decision and final method

Decision **C — REDUNDANCY IS NOT SAFELY REMOVABLE**.

Final-method recommendation: **FINAL_FULL_48**. FULL_48 retains the existing strong confirmation evidence, and COMPACT did not satisfy every conservative condition required to justify replacing it.

Correlation redundancy did not imply safe predictive removability: the deterministic compact representation materially degraded discrimination.

The AMP research matrix should now be frozen. No Stage 23 model development is recommended; proceed to a formal research-freeze/final-evaluation protocol.

No adaptive feature selection, label use, feature importance, DEV_VAL-guided subset change, threshold change, or hyperparameter tuning occurred. The original FULL_48 model remains the only formulation with existing confirmation evidence.

**CONFIRMATION NOT ACCESSED DURING STAGE 22.**

**FINAL_TEST NOT ACCESSED DURING STAGE 22.**
