# Stage 17B — Frozen Customer-Feature Learner Benchmark

## Protocol and immutable benchmark

Stage 17B tests model family only on the exact frozen 48-feature Stage 16
direct-customer representation. The Stage 17A manifest hash was verified as
`dae7dba53260a3af6b8e2ddd904b5c7867c3321f27cdff5af5e4060b97c348c0` and the Stage 16 pre-registration as
`a722299d6207fce06180f71bf96717ce58948e127a85440e04f721c1616fcdf3` before modeling.
Train-derived behavioural references were refitted using only `DEV_TRAIN` windows.

The immutable pre-scoring benchmark includes unweighted logistic regression,
sklearn HistGradientBoosting, and LightGBM. LightGBM 4.7.0 was added from a 1.4 MB
wheel using `--no-deps`; no core package was replaced. XGBoost and CatBoost were
not installed and were skipped before scoring because adding their substantially
larger compiled runtimes was not a lightweight environment change. Exact model
parameters are in `preregistered_benchmark_config.json`. No tuning or variant
matrix was run.

## Development data audit

`DEV_TRAIN` contributes 23,724 usable customers
(2,022 theft; prevalence
`0.085230146687`) and
614,976 usable windows. The governance-only
unavailable customer remains recorded but cannot enter the 48-feature matrix.
`DEV_VAL` contributes 5,931 customers
(505 theft; prevalence
`0.085145843871`) and
153,947 windows. Customer overlap is zero and
both matrices contain only original-training IDs. `CONFIRMATION` and `FINAL_TEST`
features, labels, predictions, and performance were not loaded.

## One-pass DEV_VAL benchmark

Selection is highest average precision then ROC-AUC. Threshold-dependent metrics
maximize F1 on `DEV_VAL` using the established deterministic highest-threshold
tie break, so precision, recall, F1, and confusion counts are secondary and
optimistic.

| Model | ROC-AUC | AP | Precision | Recall | F1 | Threshold | TN/FP/FN/TP | Fit seconds | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.752240 | 0.262321 | 0.317965 | 0.396040 | 0.352734 | 0.162130 | 4997/429/305/200 | 0.339 | 2,363 |
| hist_gradient_boosting | 0.788831 | 0.359795 | 0.362617 | 0.384158 | 0.373077 | 0.200706 | 5085/341/311/194 | 1.760 | 126,780 |
| lightgbm | 0.785410 | 0.358024 | 0.319386 | 0.453465 | 0.374795 | 0.158921 | 4938/488/276/229 | 1.724 | 154,575 |

The AP-first winner is `hist_gradient_boosting`. Its artifact and configuration are frozen.
HGB exceeds LightGBM by `0.001771` AP and `0.003421` ROC-AUC. LightGBM's
slightly higher validation-optimized F1 does not override the pre-registered
ranking rule. The paired intervals below include zero, so the two tree learners
are statistically compatible; HGB is also the already established, smaller
dependency-free project baseline.

## Paired uncertainty

The pre-registered comparison is `hist_gradient_boosting − lightgbm` using 2,000
paired `DEV_VAL` customer resamples with seed 42. ROC-AUC observed delta is
`+0.003420640627`, bootstrap mean
`+0.003384`, 95% percentile CI
`[-0.002134, +0.009190]`,
and fraction above zero `88.700%`.
AP observed delta is
`+0.001770915760`, mean
`+0.001059`, CI
`[-0.011291, +0.013817]`,
and fraction above zero
`56.200%`.

## Importance stability

Five-repeat, seed-42 `DEV_VAL` permutation importance for the winning tree model:

- `maximum__customer_q90`: AP decrease `0.067916`, ROC-AUC decrease `0.045698`.
- `mean__late_third_minus_early_third`: AP decrease `0.040174`, ROC-AUC decrease `0.018872`.
- `maximum__training_q90_exceedance_fraction`: AP decrease `0.032300`, ROC-AUC decrease `0.015540`.
- `median__customer_q10`: AP decrease `0.023922`, ROC-AUC decrease `0.026977`.
- `maximum_absolute_consecutive_change__customer_q90`: AP decrease `0.021586`, ROC-AUC decrease `0.004100`.
- `maximum__late_third_minus_early_third`: AP decrease `0.017052`, ROC-AUC decrease `0.005389`.
- `minimum__customer_q90`: AP decrease `0.016729`, ROC-AUC decrease `0.003118`.
- `fraction_below_minus_one__customer_mean`: AP decrease `0.015652`, ROC-AUC decrease `0.005535`.
- `q90__customer_q10`: AP decrease `0.013351`, ROC-AUC decrease `0.006127`.
- `q90__customer_q90`: AP decrease `0.012927`, ROC-AUC decrease `0.003668`.
- `mean_absolute_consecutive_change__training_q90_exceedance_fraction`: AP decrease `0.009790`, ROC-AUC decrease `0.005003`.
- `second_half_minus_first_half_mean__customer_mean`: AP decrease `0.008370`, ROC-AUC decrease `0.007475`.

These frozen-feature results show whether Stage 16's important categories remain
visible: maximum/upper-tail level, early-to-late customer change, persistence
above train-derived references, and lower-tail/low-consumption behaviour. They
are associations with the SGCC label, not proven theft mechanisms, and no feature
was removed after inspection.

All four Stage 16 categories remain stable: upper-tail maximum is again first,
early-to-late mean change is second, maximum persistence above the DEV_TRAIN q90
is third, and lower-tail median plus mean low-consumption fraction remain in the
leading group. Upper-tail abrupt consecutive changes are also retained.

## Formulation replication and decision

The direct-customer representation remains strongly learnable if the best model
retains ROC-AUC at least 0.70 and AP at least 0.25. Historical Stage 16 values
(`~0.793` ROC-AUC, `~0.376` AP) are context only and did not influence selection.
This is an internal replication under the new development split, not external
validation. HGB's new values are lower by only about `0.0044` ROC-AUC and `0.0165`
AP, so there is no dramatic collapse suggesting that the formulation depended
entirely on the historically reused validation population.

Decision **B — models essentially equivalent or HGB remains preferred for simplicity** follows the immutable rule.

## Confirmation gate and paper relevance

Confirmation gate recommendation: **YES**.
The representation remains strong, the benchmark is frozen, and no unresolved implementation issue remains.
The proposed gate question is: **Does the frozen Stage 17B pipeline retain its
discrimination advantage on the held-out CONFIRMATION population?** The gate was
not executed.

For a paper, this benchmark adds model-family evidence to the existing negative
reconstruction, behavioural representation, and direct-customer formulation
results. It does not establish novelty, state of the art, or comparability with
literature using different protocols.

**CONFIRMATION AND FINAL_TEST WERE NOT ACCESSED OR EVALUATED DURING STAGE 17B.**
