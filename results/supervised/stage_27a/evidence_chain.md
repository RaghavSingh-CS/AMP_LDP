# Stage 27A evidence chain

## Stage 24B — single-vector MIL failure

Mean, max, and gated-attention pooling compressed every customer to one latent
vector. All three were materially weaker than FULL_48; the strongest arm,
MAX_MIL, scored 0.756667 ROC-AUC and 0.314566 AP versus 0.788831/0.359795.
This isolated the aggregation bottleneck and ended the first MIL formulation.

## Stage 25A — distribution-information hypothesis

Committed evidence suggested that customer discrimination depends on
across-window location, dispersion, tails, and extremes. Stage 25A therefore
preregistered one bounded test of six fixed statistics per latent dimension,
plus an identically aggregated direct SUMMARY_17 control.

## Stage 25B — development parity

The classifier-matched latent-distribution HGB mean reached 0.790809 ROC-AUC
and 0.367837 AP, within the frozen FULL_48 parity margins and above direct
aggregation, particularly in AP. The outcome was representation parity rather
than an anchor because the development effect was small and only two seeds
exceeded FULL_48 on both metrics.

## Stage 26 — independent positive confirmation

Without retraining or tuning, the frozen five-seed mean prediction reached
0.811791 ROC-AUC and 0.415287 AP. Paired intervals versus FULL_48 and DIRECT
were above zero for both metrics, and all five individual seeds met both parity
margins. This independently supported learned-encoder value beyond distribution
aggregation and earned the Study 2 anchor.

## Why one final evaluation is justified

The chain progresses from a specific failure mechanism, through a preregistered
hypothesis and controlled development test, to an independently governed
confirmation. The method and claim are now fixed. One terminal evaluation can
test whether the qualitative learned-distribution advantage persists without
further model development.

## Remaining uncertainty

The final-cohort effect direction and magnitude remain unknown. The large
confirmation gain may attenuate, individual seeds may become less stable, and
the direct control may close the gap. The evidence is not external validation:
all cohorts share the historical data source. The primary ensemble also
outperformed individual seeds, so conclusions apply to the frozen five-seed
pipeline. Finally, the final evaluation cohort had historical Stage 13 access
for an obsolete window baseline and cannot be described as pristine or
untouched.
