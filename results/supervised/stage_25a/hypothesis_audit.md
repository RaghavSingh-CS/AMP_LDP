# Stage 25A — Learned Customer-Distribution Hypothesis Audit

## Decision

**B — GO WITH LIMITATIONS.** One final bounded experiment is scientifically
defensible because latent-distribution pooling tests a different information
bottleneck from Stage 24B. It is not an open invitation to try a larger neural
network. The idea has limited novelty because `FULL_48` already encodes explicit
customer distributions, and the proposed permutation-invariant model cannot
recover chronology-dependent or exact persistence features.

Stage 25A is design only. No model was trained, no new `DEV_VAL` prediction was
created, and no row-level `CONFIRMATION` or `FINAL_TEST` data was accessed.

## Pre-stage audit

The audit began with a clean working tree on branch `study2/mil` at exact HEAD
`07b63db323870d190a8282c8c9bcac6d6c2bcc7a`. Study 1 protected paths were
unchanged. Stage 24A and 24B artifacts were intact; the Stage 24B sealed
execution specification remained at SHA-256
`15b815e32742540903977bae52d942bf7da438b11e7625f524b7a9e3cc16635f`.
Only committed reports, configurations, aggregate results, and artifact metadata
were read.

## Relevant evidence synthesis

**Stage 14** showed that named behavioural summaries captured signal absent from
the raw 30 positions. Under unweighted HGB, `SUMMARY_ONLY` improved customer
ROC-AUC/AP over `RAW_ONLY` by `+0.020741/+0.020351`, reaching
`0.738697/0.261320`. Adding raw positions to summaries did not improve the
selected result. This supports retaining `SUMMARY_17` as the fixed instance
input; it does not establish that any learned embedding will help.

**Stage 16** produced the central mechanism evidence. Direct customer HGB over
48 explicit aggregates achieved `0.793210/0.376278`, improving over the
window-HGB/mean-probability baseline by `+0.054513/+0.114958`. The representation
included customer quantiles, variability summaries, exceedance fractions, and
four early-versus-late contrasts. Upper-tail maximum, lower-tail median,
persistence, and longitudinal contrasts all contributed. This supports
preserving customer-distribution structure, but it does not separate
representation, sample-unit, and classifier effects.

**Stage 19** found `LEVEL_DISTRIBUTION` strongest and conditionally necessary:
alone it achieved `0.767717/0.309965`, and omitting it cost
`0.025294/0.033977`. The five non-level families together still achieved
`0.763537/0.325818`, and omitting `LONGITUDINAL_CONTRAST` cost an observed
`0.009211/0.023237`. Thus signal is distribution-heavy but not merely absolute
level. This supports multiple complementary summaries rather than one pooled
location vector.

**Stage 22** removed 22 highly correlated features without labels, retaining
26/48. Discrimination fell by `0.012044/0.022468`, with paired intervals below
zero. Correlation therefore did not imply safe information removal. This warns
against assuming that one latent statistic can replace a family of apparently
redundant aggregates.

**Stage 24B** held `SUMMARY_17`, the small encoder, customer loss, and development
splits fixed. Mean, max, and attention pooling reached only
`0.739867/0.267821`, `0.756667/0.314566`, and `0.758053/0.297462`. Attention
weights were non-random and perturbation-supported, yet attention did not close
the `FULL_48` gap. This supports the narrower proposition that selecting or
reweighting instances into one vector is insufficient under the evaluated
setup. It does not prove that distribution pooling will succeed, that lost
distribution information caused the entire gap, or that a larger network is
warranted.

Jointly, the evidence motivates exactly one question: can multiple fixed
statistics over learned window behaviours recover information discarded by
single-vector MIL, and does the learned encoder add anything beyond applying the
same statistics directly to `SUMMARY_17`?

## Falsifiable hypothesis

**Customer-level theft discrimination depends partly on across-window location,
dispersion, tail, and extreme structure that a single mean, maximum, or
attention-weighted latent vector cannot preserve; under the same `SUMMARY_17`
input and Stage 24B encoder, fixed multi-statistic pooling of learned latent
dimensions will improve customer discrimination, and a classifier-matched
direct-distribution control will determine whether any gain is attributable to
learned latent behaviours rather than distribution pooling alone.**

It is falsified for the evaluated setup if the classifier-matched latent
representation fails the preregistered parity criteria or fails to add
meaningful value over the direct distribution control.

## Distinction from Stage 24B

Let (h_{ij}\in\mathbb{R}^{16}) be the learned embedding for window (j) of
customer (i).

- Mean MIL returns only (E_j[h_{ij}]): one first moment per dimension.
- Max MIL returns only (\max_j h_{ij}): one upper extreme per dimension.
- Attention MIL returns (\sum_j a_{ij}h_{ij}): one learned weighted first
  moment. It can prioritize windows but does not separately expose dispersion,
  lower tails, upper tails, or extremes.
- Latent-distribution pooling returns, per dimension, mean, stabilized
  population standard deviation, q10, median, q90, and maximum. Two bags with
  the same mean/maximum/weighted mean can differ in these remaining statistics.

This is a substantive information-bottleneck comparison, not simply a larger
head. The encoder is unchanged and the proposed neural head is only linear.
However, the output grows to 96 customer features, so dimensional expansion is
an important alternative explanation and must be exposed by controls.

## Distinction from FULL_48

`FULL_48` is:

> human-specified window summaries → human-specified customer aggregates → HGB.

The proposed representation is:

> the same frozen window summaries → learned nonlinear latent behaviours →
> fixed marginal distribution aggregates → classifier.

That is genuine supervised representation learning because the encoder learns
nonlinear mixtures of the 17 window behaviours using only customer-bag loss.
It is not automatically superior or more interpretable. It has twice as many
pooled dimensions as `FULL_48`, omits explicit persistence thresholds and
chronology, and may merely reconstruct a less transparent version of the same
idea. `DIRECT_SUMMARY_DISTRIBUTION_HGB` is therefore mandatory: if it matches
the learned representation, the result belongs to distribution aggregation,
not learned latent behaviour.

## Information limits and chronology

The detailed audit is in `information_retention_matrix.csv`. The proposed
operator retains marginal latent location, dispersion, lower/central/upper
tails, and maxima. It does not retain latent covariance/copula structure, an
explicit bag-size feature, exact frequency above the frozen training-q90
thresholds, or window order.

Stage 25 remains permutation invariant. Adding sequence structure now would
confound distribution representation with temporal modelling and become
post-hoc architecture escalation. This means the model cannot reconstruct the
four `FULL_48` latest-third-minus-earliest-third features. That loss is material
to interpretation: Stage 16 ranked `mean__late_third_minus_early_third` highly,
and Stage 19 observed an AP loss of `0.023237` when the longitudinal-contrast
family was omitted. Stage 25B therefore tests whether chronology-free learned
distributions are sufficient, not whether all `FULL_48` information is learned.

## Frozen design

The input remains exact frozen `SUMMARY_17`. Reopening `RAW_30` would change
both input information and aggregation, destroying experimental isolation.

The encoder remains the exact Stage 24B `17→32→16` ReLU network with dropout
0.1 after the first layer (1,104 parameters). Pooling adds no parameters. Each
latent dimension contributes `[mean, std, q10, median, q90, max]`, producing 96
features. A single `Linear(96,1)` head adds 97 parameters; total neural count is
1,201. No hidden customer head is allowed.

Exact quantiles are computed by sorting valid embeddings and linearly
interpolating positions `q*(m-1)`. Sort/index interpolation is differentiable
almost everywhere: gradients route to selected order statistics; ordering
changes and ties are piecewise boundaries. Mean and epsilon-stabilized standard
deviation provide dense gradients to every valid window, so the encoder is not
trained only through sparse quantile/max paths. No tensor is detached and no
smooth-quantile hyperparameter is introduced.

## Crucial control and classifier fairness

`DIRECT_SUMMARY_DISTRIBUTION_HGB` applies the same six statistics directly to
the 17 frozen features, producing 102 features, then uses the exact frozen HGB.
It is the strongest single control because it asks whether learned latent
behaviours matter beyond multi-statistic aggregation.

Classifier fairness is handled with the minimum three-arm logic:

1. train five `LATENT_DISTRIBUTION_E2E` encoders with their linear heads;
2. freeze each encoder, extract its 96 features, and fit the same HGB once per
   seed (`LATENT_DISTRIBUTION_HGB`, five fits); and
3. fit one deterministic `DIRECT_SUMMARY_DISTRIBUTION_HGB` control.

The primary comparison is classifier-matched `LATENT_DISTRIBUTION_HGB` versus
committed `FULL_48_HGB`. Latent-HGB versus direct-HGB isolates encoder value.
End-to-end versus latent-HGB describes readout contribution without reopening a
classifier benchmark. Total new fits: 11. HGB/LightGBM is not re-benchmarked;
Stage 21 already found them essentially equivalent.

## Compute and governance

The padded development input is about 52.31 MiB. A maximum 64-customer input
batch is 0.141 MiB, embeddings are 0.133 MiB, and all customer features occupy
under 10 MiB per representation. The 1,201-parameter network needs far below
1 GiB total working memory. Exact per-bag sorting may make each neural fit
slower than Stage 24B; a conservative CPU estimate is 3–8 minutes per neural
fit and 15–45 minutes total for five neural plus six small HGB fits. A GPU is
optional, not required.

Only `DEV_TRAIN` may fit encoders, standardizers, and HGBs. `DEV_VAL` is scored
once after sealing. `CONFIRMATION` and `FINAL_TEST` remain closed regardless of
the result. Study 1 metrics and probabilities are reused as fixed evidence, not
adaptive targets.

## Failure interpretation

If the classifier-matched latent representation fails the strict Stage 25B
criteria, the appropriate interpretation is that explicit `FULL_48` customer
profiling remains stronger under the evaluated setup. Stage 25B then ends this
neural representation direction. Failure does not authorize more dimensions,
depth, quantiles, sequence models, attention hybrids, or hyperparameter sweeps.
