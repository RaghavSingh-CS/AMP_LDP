# Stage 16 — Direct Customer-Level Behavioural Modeling

## Research question and boundary

Stage 16 asks whether fitting directly on compact customer aggregates of the
frozen 17 Stage 14 window summaries improves discrimination over the frozen
Stage 14 summary-only HGB fitted at window level and averaged by customer.
Only train and validation were loaded. Test features, labels, predictions, and
historical test metrics were not accessed.

## Pre-registered customer representation

The 48-feature schema was written before Stage 16 validation scoring. It contains
18 robust level features, 12 typical/upper-tail variability features, eight
direction-consistency features, two low-consumption persistence features, four
train-q90 exceedance fractions, and four chronological late-third-minus-early-third
contrasts. Exact names and definitions are in `preregistered_config.json`.
Customer/window IDs, labels, split metadata, raw dates/window position, usable
window count, missingness metadata, prior predictions, and reconstruction outputs
are excluded. The persistence reference thresholds use training windows only.

## Split, coverage, and window-count audit

The authoritative split has 29,656
train, 6,355
validation, and 6,356
test customers, with zero pairwise overlap. Stage 16 uses 768,923
train and 165,541 validation windows,
forming 29,655 train and
6,355 validation customer rows.
One training customer and no validation customers have zero usable windows and
are transparently excluded; no imputation or count feature is introduced.
The included train set has 2,527 theft
customers (prevalence `0.085213286124`);
validation has 542 (prevalence
`0.085287175452`).

Train usable-window count: min 1, median
32, mean 25.928950,
p90 34, max 34.
Validation: min 1, median
32, mean 26.048938,
p90 34, max 34.
The exact histograms are in `customer_window_count_distribution.csv`.

## Frozen baseline reproduction

The frozen Stage 14 model artifact is `models\supervised\stage_14\hist_gradient_boosting_unweighted_summary_only.joblib` with
SHA-256 `ccf6d976ad146bcf338f491a13817b785a6f4551965be687a0bafe5d50a08919`. Recomputed validation customer
probabilities match the saved Stage 14 artifact with maximum absolute difference
`9.71e-17`; all six
reported metrics reproduce within `1e-12`.

## Customer-level validation results

| Model | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage14_window_hgb_mean_customer_probability | 0.738697 | 0.261320 | 0.249110 | 0.387454 | 0.303249 | 0.108127 | 5180 | 633 | 332 | 210 |
| direct_customer_logistic | 0.744490 | 0.269984 | 0.269976 | 0.411439 | 0.326023 | 0.143462 | 5210 | 603 | 319 | 223 |
| direct_customer_hist_gradient_boosting | 0.793210 | 0.376278 | 0.410749 | 0.394834 | 0.402634 | 0.220750 | 5506 | 307 | 328 | 214 |

The selected direct model is `direct_customer_hist_gradient_boosting` by pre-registered customer
ROC-AUC then AP. Versus the frozen baseline, its exact deltas are
`+0.054513423596` ROC-AUC,
`+0.114958349481` AP, and
`+0.099384957090` F1.
Each model's operating threshold maximizes positive-class F1 on validation with
the established deterministic highest-threshold tie break. This is test-safe but
makes threshold-dependent validation metrics optimistic; ranking metrics and the
paired comparison remain the primary evidence.

## Paired customer bootstrap

Using 2,000 paired validation-customer resamples with seed 42, the selected
direct-minus-baseline ROC-AUC delta has bootstrap mean
`+0.054573` and percentile 95% CI
`[+0.037960, +0.070310]`.
The fraction of replicates above zero is
`100.000%`.
The AP delta has mean `+0.115325`
and 95% CI `[+0.082130, +0.150546]`.
Its fraction above zero is
`100.000%`.
These are descriptive validation uncertainty intervals, not an independent-test claim.

## Importance, redundancy, and stability

Selected-model validation permutation importance (five repeats, seed 42) is led by:

- `maximum__customer_q90`: ROC-AUC decrease `0.033581`, AP decrease `0.057053`.
- `mean__late_third_minus_early_third`: ROC-AUC decrease `0.020275`, AP decrease `0.049021`.
- `maximum__training_q90_exceedance_fraction`: ROC-AUC decrease `0.019730`, AP decrease `0.031658`.
- `median__customer_q10`: ROC-AUC decrease `0.015508`, AP decrease `0.020092`.
- `fraction_below_minus_one__customer_mean`: ROC-AUC decrease `0.007871`, AP decrease `0.020865`.
- `maximum_absolute_consecutive_change__customer_q90`: ROC-AUC decrease `0.007073`, AP decrease `0.016892`.
- `linear_slope__customer_standard_deviation`: ROC-AUC decrease `0.006484`, AP decrease `0.003886`.
- `maximum__late_third_minus_early_third`: ROC-AUC decrease `0.006310`, AP decrease `0.017972`.
- `interquartile_range__customer_median`: ROC-AUC decrease `0.005942`, AP decrease `0.006548`.
- `mean_absolute_consecutive_change__training_q90_exceedance_fraction`: ROC-AUC decrease `0.005628`, AP decrease `0.010393`.

Training redundancy audit (absolute Spearman >= 0.95) is led by:

- `mean__customer_median` with `median__customer_median`: Spearman `0.995992`.
- `mean__customer_q90` with `median__customer_q90`: Spearman `0.994468`.
- `mean__customer_q10` with `median__customer_q10`: Spearman `0.993709`.
- `standard_deviation__customer_median` with `mean_absolute_deviation__customer_median`: Spearman `0.989101`.
- `maximum__customer_median` with `q90__customer_median`: Spearman `0.987626`.
- `maximum__customer_q10` with `q90__customer_q10`: Spearman `0.987428`.
- `minimum__customer_q90` with `q10__customer_q90`: Spearman `0.986627`.
- `minimum__customer_median` with `q10__customer_median`: Spearman `0.986406`.
- `standard_deviation__customer_median` with `range__customer_median`: Spearman `0.985124`.
- `linear_slope__customer_standard_deviation` with `second_half_minus_first_half_mean__customer_standard_deviation`: Spearman `0.984012`.

Training effectively constant features: none.
Correlated-feature permutation importance can distribute credit among substitutes;
neither correlation nor predictive importance establishes a causal theft mechanism.

Overall level is important: upper-tail customer maximum is the largest permutation
contributor, while a lower-tail median statistic also ranks highly. Long-horizon
variability and abrupt-change persistence contribute through upper-tail consecutive
change and train-q90 exceedance features. Across-window consistency also matters:
variation in within-window slope and early-to-late level shifts carry signal. Both
lower and upper behavioural quantiles therefore matter, but the importance pattern
is distributed and partially redundant rather than a single-feature explanation.

## Decision: A — direct customer-level modeling

The pre-registered decision rule yields **A**. The result addresses the
customer-label/window-training mismatch directly: customer features are calculated
before model fitting, and validation is scored at the customer unit. It does not
show operational readiness, causal attribution, or generalization to untouched test.
The sizeable gains, with both paired intervals above zero, are consistent with the
hypothesis that propagating a customer label to every window introduces harmful
instance-level label noise. Performance alone cannot prove that mechanism; aggregation
also changes representation and model sample size. Methodological alignment and the
observed validation gain nevertheless justify making the customer the fundamental
AMP prediction unit and freezing the direct customer HGB as the next supervised baseline.

For the paper, Stage 16 is evidence about formulation rather than another broad
architecture search. It should be reported alongside the Stage 14 window baseline
and the Stage 15 negative FC hybrid result, with weak-label and repeated-validation
limitations explicit.

## Recommended Stage 17 research question

**Does the frozen direct-customer HGB retain its advantage over the frozen Stage 14
window-to-customer baseline across pre-specified customer history-length and
consumption-level strata?** Run this as a validation-only robustness audit with
fixed models and no feature changes before considering any further test use.
**TEST NOT ACCESSED DURING STAGE 16.**
