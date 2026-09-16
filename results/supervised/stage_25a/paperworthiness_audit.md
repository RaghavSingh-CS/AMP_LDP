# Stage 25A — Paperworthiness Audit

## Standard

Beating Stage 24B is not paper-worthy. Stage 24B was materially below the
frozen conventional baseline. Any anchor must withstand the stronger question:
does learned latent-distribution representation match or materially improve
`FULL_48_HGB`, and does it add value beyond direct distribution aggregation
under the same HGB?

## Candidate A — performance improvement

> “Learned distributional pooling improved AP by X over manually engineered
> customer profiling.”

Required evidence is Outcome A: five-seed mean gains of at least `+0.010`
ROC-AUC and `+0.020` AP over `FULL_48_HGB`, at least four seeds improving both,
paired lower bounds above zero, and classifier-matched gains of at least
`+0.005/+0.010` over `DIRECT_SUMMARY_DISTRIBUTION_HGB` with both intervals above
zero. Without the direct control, more features or distribution pooling—not
learned latent behaviour—could explain the result. With all requirements met,
this would be the strongest potential anchor and reasonably difficult to
dismiss as a routine architecture tweak.

## Candidate B — representation parity

> “A learned latent-distribution representation matched explicit 48-feature
> customer profiling using only customer-level supervision.”

Parity requires mean deltas no worse than `-0.005` ROC-AUC and `-0.010` AP,
four of five seeds within both margins, and noninferiority bootstrap bounds.
It also requires a meaningful classifier-matched gain over the direct
distribution control. Even then, the claim has limited novelty: `FULL_48` also
uses customer labels, the learned representation has 96 dimensions, and it is
less interpretable. A reviewer could reasonably call parity an engineering
alternative unless encoder value, reproducibility, and the chronology/persistence
limitations are explicit.

## Candidate C — recovered discrimination

> “Distribution-preserving aggregation recovered X of the discrimination lost
> by mean/max/attention MIL.”

For metric (M), define recovery relative to Stage 24B max pooling as
`(M_candidate - M_MAX_MIL) / (M_FULL48 - M_MAX_MIL)`. Report it only when the
denominator is positive and never clip values. This is informative mechanism
evidence, but not an anchor by itself: almost any more expressive aggregation
could beat the failed MIL arms. It becomes paper-relevant only if the candidate
also reaches Outcome A or B and exceeds the direct distribution control.

## Main alternative explanations

1. The gain comes from expanding one pooled vector to 96/102 customer features,
   not from learned latent behaviours.
2. HGB, rather than the representation, recovers the performance.
3. The learned encoder approximates existing human summaries without adding new
   information.
4. Repeated `DEV_VAL` use across the broader research program makes a marginal
   advantage optimistic.
5. Any remaining deficit may be caused by chronology and exact persistence
   features that permutation-invariant marginal pooling cannot represent.

The direct-HGB control addresses explanation 1; latent-HGB versus end-to-end
addresses explanation 2; the control comparison and feature interpretation
partly address explanation 3. Explanations 4 and 5 remain limitations.

## Verdict

One final bounded experiment can be justified, but no claim is earned in
Stage 25A. A Stage 25B result that merely beats Stage 24B, only one seed, or
only the neural readout is not an anchor. Failure ends this neural
representation direction absent a qualitatively new external hypothesis.
