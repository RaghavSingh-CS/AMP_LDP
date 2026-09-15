# Stage 13 — Leakage-Safe Supervised Signal Diagnostic

## Scope and question

This stage asks only whether the existing labels, preprocessing, customer split,
and consumption windows contain supervised signal that discriminates labelled
theft customers better than reconstruction error. It does not establish event
localization, causal theft indicators, deployment performance, or a production
detector. No earlier preprocessing, split, model, threshold, or output was
modified.

## Split and label audit

The authoritative `data/processed/split_manifest.parquet` assigns each customer
to exactly one stratified 70/15/15 split with seed 42. Pairwise customer overlap
is zero. The complete processed population contains 42,367 customers: 38,755
normal and 3,612 theft-labelled (8.525503%).

| Split | Customers | Normal | Theft | Customer prevalence | Usable windows | Normal windows | Theft windows | Window prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 29,656 | 27,128 | 2,528 | 8.524413% | 768,923 | 708,308 | 60,615 | 7.883104% |
| Validation | 6,355 | 5,813 | 542 | 8.528718% | 165,541 | 152,246 | 13,295 | 8.031243% |
| Test | 6,356 | 5,814 | 542 | 8.527376% | 165,020 | 151,985 | 13,035 | 7.899043% |
| Total | 42,367 | 38,755 | 3,612 | 8.525503% | 1,099,484 | 1,012,539 | 86,945 | 7.907800% |

The canonical pre-filter stride-30 manifest has 1,440,478 windows. Stage 3C's
already-adopted rule retains the 1,099,484 windows with at least one observed
source reading. One theft-labelled training customer has no usable windows;
no validation or test customer is lost.

`FLAG` is customer-level. Each window inherits its customer's label, so a
theft-labelled window is not a confirmed theft event. Aggregate test counts
were taken from the authoritative manifest and the existing Stage 3C summary
for this requested audit. Before freezing the decision, no test feature matrix
or per-window test labels were loaded and no test information affected model,
configuration, feature, or threshold selection.

## Leakage-safe input

Each model receives exactly 30 ordered float32 consumption values from an
existing usable stride-30 window. These values retain the established Stage 1
customer-wise imputation and log1p transform and the Stage 2 scaler fitted on
training customers only. Logistic regression additionally fits a StandardScaler
on training windows inside its pipeline. No large feature-engineering step was
introduced.

Excluded from model input are `FLAG`, `CONS_NO`, split indicators, window IDs,
window number, feature indices, dates, missingness mask/count/fraction, and all
reconstruction errors, anomaly flags, thresholds, and predictions from earlier
models. The identifiers and labels are retained outside the feature matrix only
for integrity checks, evaluation, and customer aggregation.

## Fixed baselines and threshold policy

All runs use seed 42. The dummy model predicts the training prior. Logistic
regression uses `C=1`, `lbfgs`, and at most 300 iterations, with unweighted and
`class_weight="balanced"` variants. HistGradientBoosting uses 100 fixed
iterations, learning rate 0.1, 15 leaf nodes, minimum leaf size 100, L2=1, no
early stopping, and unweighted/balanced variants. There is no resampling,
synthetic data, neural network, or hyperparameter search.

For each model, the window threshold maximizes positive-class F1 on validation.
Customer performance uses the arithmetic mean predicted probability across a
customer's usable windows and a separate validation-F1-maximizing threshold.
Consequently, threshold-dependent validation metrics describe the selected
validation operating point and should not be read as an independent estimate.
ROC-AUC and average precision are threshold-free. Validation window prevalence
(8.031243%) is the average-precision no-skill reference.

## Validation results

### Window level

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500000 | 0.080312 | 0.080312 | 1.000000 | 0.148684 | 0.078831 | 0 | 152,246 | 0 | 13,295 |
| Logistic, unweighted | 0.619778 | 0.170985 | 0.171009 | 0.279428 | 0.212171 | 0.114403 | 134,237 | 18,009 | 9,580 | 3,715 |
| Logistic, balanced | 0.619778 | 0.170669 | 0.170693 | 0.280105 | 0.212121 | 0.589095 | 134,153 | 18,093 | 9,571 | 3,724 |
| HistGradientBoosting, unweighted | 0.661693 | 0.199797 | 0.206347 | 0.293945 | 0.242477 | 0.118816 | 137,215 | 15,031 | 9,387 | 3,908 |
| **HistGradientBoosting, balanced** | **0.662847** | **0.203618** | **0.206718** | **0.294848** | **0.243040** | **0.608719** | **137,203** | **15,043** | **9,375** | **3,920** |

### Customer level: mean window probability

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500000 | 0.085287 | 0.085287 | 1.000000 | 0.157170 | 0.078831 | 0 | 5,813 | 0 | 542 |
| Logistic, unweighted | 0.642090 | 0.185769 | 0.178497 | 0.315498 | 0.228000 | 0.107934 | 5,026 | 787 | 371 | 171 |
| Logistic, balanced | 0.633574 | 0.176297 | 0.163424 | 0.309963 | 0.214013 | 0.569862 | 4,953 | 860 | 374 | 168 |
| HistGradientBoosting, unweighted | **0.717956** | **0.240969** | **0.203861** | **0.487085** | **0.287425** | **0.089201** | **4,782** | **1,031** | **278** | **264** |
| HistGradientBoosting, balanced | 0.712044 | 0.222600 | 0.202797 | 0.428044 | 0.275208 | 0.525133 | 4,901 | 912 | 310 | 232 |

The pre-registered selection rule is highest validation window ROC-AUC, then
window average precision, among non-dummy models. It selected balanced
HistGradientBoosting. The model SHA-256, window threshold 0.608718870015863,
and customer-mean threshold 0.5251327815829091 were written to
`frozen_decision.json` before test inference.

## Frozen test evaluation

The established Stage 6/8/10/12 methodology permits one final evaluation after
validation-only selection. Only the selected balanced HistGradientBoosting
model was evaluated. Its saved hash was verified; there was no retraining or
threshold change.

| Level | Prevalence | ROC-AUC | Average precision | Precision | Recall | F1 | Threshold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Window | 0.078990 | 0.657693 | 0.205305 | 0.218789 | 0.312313 | 0.257316 | 0.608719 | 137,449 | 14,536 | 8,964 | 4,071 |
| Customer mean | 0.085274 | 0.702314 | 0.219317 | 0.208861 | 0.426199 | 0.280340 | 0.525133 | 4,939 | 875 | 311 | 231 |

## Decision gate and interpretation

1. **Learnable signal:** yes. Logistic regression reaches validation window
   ROC-AUC 0.619778, and the small tree reaches 0.662847 with average precision
   2.54 times the 0.080312 prevalence baseline. This is not a model-complexity-only
   effect.
2. **Versus reconstruction:** yes, materially stronger in this controlled
   diagnostic. The selected validation window ROC-AUC exceeds the FC
   reconstruction validation ROC-AUC 0.564310 by 0.098536. The frozen test
   comparison is 0.657693 versus 0.561688, a difference of 0.096005.
3. **Class weighting:** no material ranking change. It changes logistic window
   ROC-AUC by less than 0.000001 and tree ROC-AUC by +0.001154. It mainly changes
   probability calibration and therefore the numerical threshold; customer
   ranking is slightly worse for both balanced variants.
4. **Architecture direction:** the result is strong enough to justify a
   controlled supervised or hybrid research track, but not a deployment claim.
   Precision, recall, and F1 remain modest, and customer-level weak labels make
   per-window interpretation especially uncertain.
5. **Recommended next stage:** investigate label/window formulation and compact,
   pre-registered consumption summaries before adding architecture complexity.
   A later hybrid comparison may test whether the frozen FC reconstruction score
   adds incremental validation signal to the supervised baseline. Test must not
   guide those choices.

This stage therefore answers its narrow question affirmatively: the existing
consumption-derived data contains supervised statistical signal beyond the
reconstruction-only objective. It does not show that labelled customers were
stealing during every window, that the learned associations are causal, or that
the model is suitable for field deployment.
