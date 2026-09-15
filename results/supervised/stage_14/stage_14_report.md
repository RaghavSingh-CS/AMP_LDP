# Stage 14 — Behavioural Representation Analysis

## Objective and boundary

Stage 14 tests whether a compact, interpretable representation of each existing
30-value consumption window improves supervised discrimination beyond the raw
values. It changes representation, not model complexity. It preserves the Stage
1–3 preprocessing, fixed customer split, usable-window rule, Stage 13 model
settings, and customer mean-probability aggregation. No neural network, GAN,
SMOTE, synthetic data, broad search, or test evaluation is used.

The Stage 13 raw-only validation metrics were reproduced within `1e-12` for all
four logistic/tree weighting variants. The authoritative split-manifest SHA-256
remains `6ad1132bbaef0c6f6ccc5bdc973844b805c4291612ac7170ec165f273835959c`.
Train/validation/test customer counts remain 29,656/6,355/6,356 with zero
pairwise overlap. Stage 14 loaded only 768,923 training and 165,541 validation
windows. The test feature matrix, labels, predictions, and historical Stage 13
test metrics were not loaded or used.

## Pre-registered behavioural features

The following 17-feature list and all definitions were written to
`preregistered_config.json` before Stage 14 validation scoring:

1. `mean`
2. `median`
3. `minimum`
4. `maximum`
5. `q10`
6. `q90`
7. `standard_deviation`
8. `range`
9. `interquartile_range`
10. `mean_absolute_deviation`
11. `first_to_last_change`
12. `absolute_first_to_last_change`
13. `mean_absolute_consecutive_change`
14. `maximum_absolute_consecutive_change`
15. `linear_slope`
16. `fraction_below_minus_one`
17. `second_half_minus_first_half_mean`

All are deterministic functions only of the 30 scaled consumption values.
`fraction_below_minus_one` means consumption at least one unit below the
established training-standardized date feature; it is not a claim of physical
zero usage. The slope uses fixed positions 0–29. Quantiles use NumPy linear
interpolation and standard deviation uses `ddof=0`.

`FLAG`, `CONS_NO`, window/customer IDs, split, dates and indices, missingness
information, prior supervised predictions, reconstruction outputs, and test
information are excluded. Labels and customer IDs remain outside the model
matrix solely for training, integrity checks, metrics, and aggregation.

## Feature sanity

All 13,071,691 training summary values and 2,814,197 validation summary values
are finite. No feature is effectively constant. Train and validation
distributions are broadly aligned. Some valid scaled-consumption extremes
remain, consistent with the established no-clipping policy; no new clipping or
transformation was introduced. Logistic pipelines fit StandardScaler on
training windows only. Histogram gradient boosting uses its fixed binning and
is not sensitive to raw magnitude in the same way.

## Fixed model matrix and threshold policy

The representations are `RAW_ONLY` (30 columns), `SUMMARY_ONLY` (17 columns),
and `RAW_PLUS_SUMMARY` (47 columns). Each is evaluated with Stage 13's logistic
regression and HistGradientBoosting settings, both unweighted and balanced.
There is no hyperparameter search.

For each fitted candidate, window and customer thresholds independently
maximize positive-class F1 on validation. Threshold-dependent validation
metrics therefore describe the selected validation operating point rather than
an independent performance estimate. Customer-level ROC-AUC and average
precision are primary. Validation prevalence is 8.031243% by window and
8.528718% by customer.

## Complete window-level validation results

| Family | Weight | Representation | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | Unweighted | RAW_ONLY | 0.619778 | 0.170985 | 0.171009 | 0.279428 | 0.212171 | 0.114403 | 134237 | 18009 | 9580 | 3715 |
| Logistic | Unweighted | SUMMARY_ONLY | 0.648541 | 0.194385 | 0.210666 | 0.269199 | 0.236362 | 0.126405 | 138836 | 13410 | 9716 | 3579 |
| Logistic | Unweighted | RAW_PLUS_SUMMARY | 0.648292 | 0.195419 | 0.209401 | 0.271756 | 0.236538 | 0.126218 | 138605 | 13641 | 9682 | 3613 |
| Logistic | Balanced | RAW_ONLY | 0.619778 | 0.170669 | 0.170693 | 0.280105 | 0.212121 | 0.589095 | 134153 | 18093 | 9571 | 3724 |
| Logistic | Balanced | SUMMARY_ONLY | 0.651069 | 0.191028 | 0.196374 | 0.294923 | 0.235765 | 0.606956 | 136200 | 16046 | 9374 | 3921 |
| Logistic | Balanced | RAW_PLUS_SUMMARY | 0.650946 | 0.192408 | 0.191765 | 0.306882 | 0.236036 | 0.602868 | 135050 | 17196 | 9215 | 4080 |
| HGB | Unweighted | RAW_ONLY | 0.661693 | 0.199797 | 0.206347 | 0.293945 | 0.242477 | 0.118816 | 137215 | 15031 | 9387 | 3908 |
| HGB | Unweighted | SUMMARY_ONLY | 0.676666 | 0.213635 | 0.218110 | 0.288981 | 0.248593 | 0.125655 | 138473 | 13773 | 9453 | 3842 |
| HGB | Unweighted | RAW_PLUS_SUMMARY | 0.675908 | 0.213380 | 0.208243 | 0.311245 | 0.249533 | 0.116292 | 136513 | 15733 | 9157 | 4138 |
| HGB | Balanced | RAW_ONLY | 0.662847 | 0.203618 | 0.206718 | 0.294848 | 0.243040 | 0.608719 | 137203 | 15043 | 9375 | 3920 |
| HGB | Balanced | SUMMARY_ONLY | 0.677033 | 0.216005 | 0.208636 | 0.309289 | 0.249182 | 0.609304 | 136649 | 15597 | 9183 | 4112 |
| HGB | Balanced | RAW_PLUS_SUMMARY | 0.676602 | 0.215352 | 0.209859 | 0.307409 | 0.249435 | 0.610624 | 136858 | 15388 | 9208 | 4087 |

## Complete customer-level validation results

Customer predictions are arithmetic means of usable-window probabilities.

| Family | Weight | Representation | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | Unweighted | RAW_ONLY | 0.642090 | 0.185769 | 0.178497 | 0.315498 | 0.228000 | 0.107934 | 5026 | 787 | 371 | 171 |
| Logistic | Unweighted | SUMMARY_ONLY | 0.682841 | 0.226928 | 0.271154 | 0.260148 | 0.265537 | 0.125488 | 5434 | 379 | 401 | 141 |
| Logistic | Unweighted | RAW_PLUS_SUMMARY | 0.683218 | 0.228988 | 0.311881 | 0.232472 | 0.266385 | 0.134346 | 5535 | 278 | 416 | 126 |
| Logistic | Balanced | RAW_ONLY | 0.633574 | 0.176297 | 0.163424 | 0.309963 | 0.214013 | 0.569862 | 4953 | 860 | 374 | 168 |
| Logistic | Balanced | SUMMARY_ONLY | 0.673819 | 0.213057 | 0.250460 | 0.250923 | 0.250691 | 0.602641 | 5406 | 407 | 406 | 136 |
| Logistic | Balanced | RAW_PLUS_SUMMARY | 0.674059 | 0.214784 | 0.247756 | 0.254613 | 0.251137 | 0.601008 | 5394 | 419 | 404 | 138 |
| HGB | Unweighted | RAW_ONLY | 0.717956 | 0.240969 | 0.203861 | 0.487085 | 0.287425 | 0.089201 | 4782 | 1031 | 278 | 264 |
| **HGB** | **Unweighted** | **SUMMARY_ONLY** | **0.738697** | **0.261320** | **0.249110** | **0.387454** | **0.303249** | **0.108127** | **5180** | **633** | **332** | **210** |
| HGB | Unweighted | RAW_PLUS_SUMMARY | 0.737707 | 0.260590 | 0.258367 | 0.356089 | 0.299457 | 0.113387 | 5259 | 554 | 349 | 193 |
| HGB | Balanced | RAW_ONLY | 0.712044 | 0.222600 | 0.202797 | 0.428044 | 0.275208 | 0.525133 | 4901 | 912 | 310 | 232 |
| HGB | Balanced | SUMMARY_ONLY | 0.733311 | 0.242302 | 0.222015 | 0.439114 | 0.294919 | 0.534001 | 4979 | 834 | 304 | 238 |
| HGB | Balanced | RAW_PLUS_SUMMARY | 0.732086 | 0.241565 | 0.214346 | 0.468635 | 0.294152 | 0.522473 | 4882 | 931 | 288 | 254 |

## Customer-level deltas versus RAW_ONLY

| Family | Weight | Representation | ROC-AUC delta | Relative ROC delta | AP delta | Relative AP delta |
|---|---|---|---:|---:|---:|---:|
| Logistic | Unweighted | SUMMARY_ONLY | +0.040752 | +6.3467% | +0.041159 | +22.1562% |
| Logistic | Unweighted | RAW_PLUS_SUMMARY | +0.041128 | +6.4053% | +0.043219 | +23.2652% |
| Logistic | Balanced | SUMMARY_ONLY | +0.040245 | +6.3520% | +0.036760 | +20.8512% |
| Logistic | Balanced | RAW_PLUS_SUMMARY | +0.040485 | +6.3899% | +0.038487 | +21.8309% |
| HGB | Unweighted | SUMMARY_ONLY | +0.020741 | +2.8889% | +0.020351 | +8.4456% |
| HGB | Unweighted | RAW_PLUS_SUMMARY | +0.019752 | +2.7511% | +0.019621 | +8.1427% |
| HGB | Balanced | SUMMARY_ONLY | +0.021267 | +2.9867% | +0.019702 | +8.8507% |
| HGB | Balanced | RAW_PLUS_SUMMARY | +0.020041 | +2.8146% | +0.018964 | +8.5195% |

The positive direction across both families and both weighting settings is more
important than the isolated magnitude of any one comparison. For the strongest
tree, adding raw positions to the summaries changes customer ROC-AUC/AP by
`-0.000989`/`-0.000730` relative to summaries alone.

## Validation permutation importance

The selected unweighted summary-only HGB was evaluated with five seeded
validation permutations per feature. Importance is the decrease in customer
metric; it is diagnostic and was not used to revise the feature set.

| Feature | Mean ROC-AUC decrease | Mean AP decrease |
|---|---:|---:|
| maximum | 0.099512 | 0.085441 |
| mean_absolute_consecutive_change | 0.027551 | 0.017857 |
| median | 0.021444 | 0.021027 |
| minimum | 0.017606 | 0.013181 |
| maximum_absolute_consecutive_change | 0.012767 | 0.008670 |
| mean_absolute_deviation | 0.007397 | 0.006495 |
| q10 | 0.006867 | 0.007488 |
| standard_deviation | 0.003259 | 0.005252 |
| range | 0.003005 | 0.002318 |
| linear_slope | 0.002147 | 0.000775 |
| absolute_first_to_last_change | 0.002006 | 0.003147 |
| interquartile_range | 0.001534 | 0.003249 |
| second_half_minus_first_half_mean | 0.000637 | 0.001935 |
| mean | 0.000607 | 0.000129 |
| fraction_below_minus_one | 0.000279 | 0.002237 |
| first_to_last_change | -0.000011 | 0.000272 |
| q90 | -0.000657 | -0.005232 |

Correlated summaries can substitute for one another, so permutation values are
not independent causal contributions. The dominant associations are maximum
level, typical/lower level, and local variability or jumps. Linear trend,
early-versus-late shift, and the relative-low-consumption fraction contribute
little conditional importance. High maxima or abrupt changes could be related
to theft-labelled consumption behaviour, but they could also reflect customer
type, load scale, meter quality, or other confounding. Association is not proof
of theft mechanism.

## Decision gate

- Behavioural summaries improve customer ROC-AUC over raw values for every
  family/weighting comparison: **yes**.
- They improve customer average precision in every comparison: **yes**.
- Summary-only retains most of the raw signal and exceeds raw-only for the
  strongest model: **yes**.
- Raw-plus-summary gives a positive delta versus raw-only but no improvement
  over summary-only for HGB: **yes versus raw, no incremental tree gain**.
- The delta is moderate but consistent enough to make summaries part of AMP:
  **yes**, as a research baseline rather than a deployment claim.
- The frozen supervised representation for the next research stage is
  **unweighted HistGradientBoosting with SUMMARY_ONLY**, selected solely by
  customer validation ROC-AUC then average precision. Its saved threshold and
  model hash are validation-only artifacts; no test evaluation is permitted.

## Recommended Stage 15 question

The justified next question is: **Do frozen FC autoencoder-derived signals add
incremental validation information beyond the Stage 14 summary-only supervised
representation?** This should be a pre-registered train/validation-only hybrid
ablation. The historical test set remains closed. Stage 15 was not executed.
