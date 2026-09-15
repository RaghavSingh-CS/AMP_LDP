# Stage 17A — Development and Confirmation Protocol

## Motivation and historical disclosure

Stages 13–16 legitimately used the original validation population for exploratory
model, representation, hybrid, and formulation decisions. That population is no
longer a statistically pristine development holdout. It remains more independent
than data used for routine future iteration, so it is reassigned to `CONFIRMATION`
and may be consulted only at explicit, recorded confirmation gates.

The historical customer split is not changed or overwritten. The hierarchy is:

`original train → DEV_TRAIN + DEV_VAL`

`original validation → CONFIRMATION`

`original test → FINAL_TEST`

## Deterministic development split

Only the 29,655 original-training customers with an established usable Stage 3C
stride-30 window were eligible for the stratified split. IDs were sorted ascending,
then split once with `sklearn.model_selection.train_test_split`, `test_size=0.20`,
`shuffle=True`, customer-`FLAG` stratification, and seed `42`. The seed follows
the repository convention and was not screened using model performance or feature
values. The one original-training customer without a usable window is retained as
`DEV_TRAIN`, marked representation-unavailable, and excluded transparently from
Stage 16 representation fitting until a separate policy exists.

## Frozen role audit

| Role | Customers | Normal | Theft | Theft prevalence | Stage 16 representation available | Unavailable |
|---|---:|---:|---:|---:|---:|---:|
| DEV_TRAIN | 23,725 | 21,702 | 2,023 | 0.085268703899 | 23,724 | 1 |
| DEV_VAL | 5,931 | 5,426 | 505 | 0.085145843871 | 5,931 | 0 |
| CONFIRMATION | 6,355 | 5,813 | 542 | 0.085287175452 | 6,355 | 0 |
| FINAL_TEST | 6,356 | 5,814 | 542 | 0.085273757080 | 6,356 | 0 |

All six pairwise role overlaps are zero. Their union is exactly all 42,367
authoritative customers. Every original-validation customer is `CONFIRMATION`,
every original-test customer is `FINAL_TEST`, and both development roles contain
only original-training customers.

The machine-readable manifest is `results/governance/stage_17a/research_protocol_manifest.csv`, SHA-256
`dae7dba53260a3af6b8e2ddd904b5c7867c3321f27cdff5af5e4060b97c348c0`. It contains customer ID, historical split,
new research role, and Stage 16 representation availability; it contains no label
or feature columns. Confirmation/final class and availability counts above come
from the already tracked aggregate Stage 13 split audit. Row-level test labels were
not reopened.

## Frozen future-development policy

Routine Stage 17+ candidate selection, feature experiments, hyperparameter choices,
imbalance experiments, and architecture comparisons must fit on `DEV_TRAIN` and
select only on `DEV_VAL`. `CONFIRMATION` and `FINAL_TEST` must not be consulted
during routine iteration. Split IDs and seed are frozen; candidate splits must not
be regenerated.

`CONFIRMATION` may be accessed only after a candidate methodology is frozen—for
example after controlled supervised benchmarking, a major representation change,
or before treating a proposed contribution as sufficiently credible to continue.
A confirmation result must be logged as a gate and must not trigger small direct
tweaks. A failed gate returns research to `DEV_TRAIN`/`DEV_VAL`; a later gate
requires a materially revised, pre-specified hypothesis.

`FINAL_TEST` remains sealed until the final AMP methodology is frozen. The one-time
historical Stage 13 test evaluation must remain disclosed, so final test is not
described as historically untouched. No Stage 14+ decision used test, and future
development must not access it before final evaluation.

## Historical results and next experiment

Stages 13–16 remain valid historical exploratory evidence under their documented
old protocol; they are not rewritten as if this hierarchy already existed. The
Stage 16 direct-customer HGB remains the current historical best, but is not
automatically confirmed. Future benchmarking must reconstruct the frozen 48-feature
representation using `DEV_TRAIN`/`DEV_VAL` only.

Stage 17B is pre-defined but not executed:

**On the frozen 48-feature direct-customer representation, which conventional
supervised learner provides the strongest discrimination under the new
DEV_TRAIN/DEV_VAL protocol?**

The expected controlled families are logistic regression, HistGradientBoosting,
and XGBoost, LightGBM, or CatBoost only if available without environment disruption.
There will be no broad AutoML, neural network, GAN, or feature redesign. Primary
metrics are customer ROC-AUC and average precision; precision, recall, and F1 are
secondary. Exact Stage 17B details will be specified separately.

**NO NEW MODEL BENCHMARKING, CONFIRMATION EVALUATION, OR FINAL-TEST EVALUATION WAS
PERFORMED DURING STAGE 17A.**
