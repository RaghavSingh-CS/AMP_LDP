# Stage 27A — final Study 2 freeze and final-evaluation protocol

## Decision

**B — READY WITH GOVERNANCE LIMITATION.** One terminal evaluation is justified
by the preregistered chain from a diagnosed single-vector MIL failure, through
development parity, to independent positive confirmation. The limitation is
that the final evaluation cohort had disclosed historical Stage 13 baseline
access and cannot be represented as pristine or untouched.

## Frozen method and refit decision

Stage 27B will use **Option A — frozen development models**. The primary method
remains frozen SUMMARY_17 windows → frozen 17→32→16 encoder → per-dimension
mean, stabilized population SD, q10, median, q90, and maximum → 96-dimensional
representation → frozen HGB. The primary customer probability is the arithmetic
mean across seeds 42–46. No seed is weighted, removed, or selected.

Refitting was rejected because it would create a new, unconfirmed pipeline and
weaken the replication claim. Frozen reuse also permits symmetric training
exposure: LATENT, DIRECT, and the selected Stage 17B FULL_48 comparator were all
fit on DEV_TRAIN only. The later Study 1 Stage 23B FULL_48 refit used
DEV_TRAIN+DEV_VAL and is historical context, not the paired Stage 27B control.

## Final comparisons

The only models are LATENT, FULL_48, and DIRECT. Customer ROC-AUC and average
precision are primary; no final-cohort threshold optimization is permitted.
The paired comparisons are LATENT−FULL_48, LATENT−DIRECT, and DIRECT−FULL_48.
Each uses 2,000 paired customer bootstraps with already-frozen Study 2 seed
2502, reporting observed and bootstrap-mean deltas, percentile intervals, and
fractions above zero.

Strong final replication requires parity to FULL_48, paired bounds compatible
with those margins, positive learned-encoder value over DIRECT with at least one
meaningful uncertainty-supported gain, at least 4/5 seeds within both parity
margins, and no concerning seed collapse. Partial/mixed replication is the
intermediate category. Material inferiority, a supported AP reversal versus
DIRECT, or severe seed collapse produces failed final replication. The exact
mechanical criteria are in `final_evaluation_criteria.json`.

## Anchor and paperworthiness

The anchor survives only under strong final replication. It survives with
qualification only under partial/mixed replication when LATENT is not
materially inferior to FULL_48, retains positive AP evidence over DIRECT, and at
least 3/5 seeds retain parity. Otherwise it does not survive.

Level 1 corresponds to strong final replication and anchor survival. Level 2
corresponds to a qualified mixed result with the failed condition stated.
Level 3 preserves development/confirmation history but makes no final-supported
anchor claim.

## Cohort disclosure

> The final evaluation cohort had been accessed historically in Stage 13 for an obsolete supervised window-level baseline. The learned latent-distribution hypothesis, architecture, pooling rule, five-seed ensemble, controls, and confirmation decision reported in Study 2 were developed and frozen later without Study 2 row-level access to this cohort. We therefore treat it as a governed final evaluation cohort with disclosed historical baseline access, not as a pristine untouched test set.

## Terminal rule

After Stage 27A, Study 2 model development is closed. Stage 27B may execute once
only after this preregistration is committed and its hash recorded. Regardless
of outcome, no refit, rescue model, additional seed, feature, statistic,
architecture, classifier, or hyperparameter search is permitted.

Stage 27A accessed no final-evaluation row-level feature or label, generated no
final prediction, and trained or refit no model.
