# AMP paper method freeze

## Primary research question

Given SGCC customer-level `FLAG` supervision without verified theft-event timing,
can unlabeled 30-day consumption windows be encoded and aggregated through their
latent distributions to improve customer-level electricity-theft discrimination
beyond direct distribution summaries and an engineered customer-profile benchmark?

This question is intentionally limited to customer-level discrimination on SGCC.

## Manuscript terminology

Candidate method names are:

1. **AMP–Latent Distribution Profiling (AMP-LDP)** — recommended.
2. Learned Window Distribution Profiling (LWDP).
3. Latent Distribution Customer Profiling (LDCP).
4. Window-to-Customer Latent Distribution (WCLD).
5. AMP Distribution-Preserving Encoder (AMP-DPE).

`AMP-LDP` is concise, compatible with the project identity, and describes both
the learned latent representation and customer distribution profile. It is
manuscript terminology only; historical repository identifiers are unchanged.

## Frozen proposed method

The proposed method is:

`customer daily series → non-overlapping 30-day windows → SUMMARY_17 →`
`17→32→16 encoder → six statistics per latent dimension → 96 features → HGB →`
`arithmetic mean of five seed probabilities`.

- Encoder: `Linear(17,32) → ReLU → Dropout(0.1) → Linear(32,16) → ReLU`.
- Per-latent-dimension statistics, in order: mean, stabilized population SD,
  q10, median, q90 and maximum.
- Classifier: the frozen Stage 25B latent HGB corresponding to each encoder.
- Seeds: `42, 43, 44, 45, 46`.
- Primary prediction: unweighted arithmetic mean of the five probabilities.
- Training population: representation-available DEV_TRAIN customers only.
- No chronology, attention, bag-size feature, threshold optimization, or
  instance-level theft loss is part of the proposed method.

## Required distinctions

| Role | Frozen method | Scientific purpose |
|---|---|---|
| Proposed method | AMP-LDP / `LATENT_DISTRIBUTION_HGB` | Tests learned window representation before distribution-preserving customer aggregation |
| Engineered comparator | `FULL_48_HGB` | Strong established customer-profile benchmark with engineered chronology/persistence information |
| Mechanistic control | `DIRECT_SUMMARY_DISTRIBUTION_HGB` | Applies the identical six aggregation statistics directly to SUMMARY_17, isolating learned-encoding value |
| Single-vector MIL baselines | `MEAN_MIL`, `MAX_MIL`, `GATED_ATTENTION_MIL` | Tests whether one pooled bag vector is sufficient; not part of the final pipeline |
| Reconstruction models | FC, CNN and LSTM autoencoders | Historical motivation and negative exploratory evidence; not components of AMP-LDP |

## Label-granularity statement

`FLAG` is a customer-level label. Window timestamps indicate measurement
coverage, not verified theft-event timing. Assigning `FLAG` to every temporal
window therefore creates propagated weak labels rather than verified
window-level theft labels. AMP-LDP uses customer-level supervision and does not
impose an instance-level theft loss.

## Exactly three contributions

1. **We formulate** SGCC electricity-theft discrimination at the granularity of
   its customer-level labels, treating temporal windows as unlabeled instances
   rather than verified theft periods.
2. **We investigate** a distribution-preserving learned representation that
   encodes 30-day behavioural summaries and aggregates latent location,
   dispersion, tails and extremes into a customer profile.
3. **We show empirically**, under frozen governance, that the learned
   representation improves on identical direct distribution aggregation and
   remains competitive with or better than a strong engineered customer-profile
   comparator across development, confirmation and terminal evaluation.

## Anchor freeze

The terminal evidence supports the authoritative anchor without strengthening it:

> Learning window representations before distribution-preserving customer aggregation adds reproducible customer-level discrimination beyond identical direct aggregation across development, confirmation, and the governed final evaluation.

The anchor does not imply theft localization, state-of-the-art performance,
external generalization, deployment validity, or causality.

## Dataset and split freeze

- Cleaned customers: `42,367` (`38,755` normal, `3,612` theft-labelled;
  prevalence `8.5255%`).
- Daily consumption dimensions after preprocessing: `1,034`.
- Window length: `30` days.
- DEV_TRAIN: `23,725` governed (`21,702` normal, `2,023` positive), with
  `23,724` representation-available customers (`2,022` positive).
- DEV_VAL: `5,931` (`5,426` normal, `505` positive).
- CONFIRMATION: `6,355` (`5,813` normal, `542` positive).
- Final evaluation cohort with disclosed historical baseline access: `6,356`
  (`5,814` normal, `542` positive).

The one unavailable DEV_TRAIN customer is
`B08E0670C8A6933E5EC44715F7A80E9D`. The committed governance states only that
this original-training customer had no usable Stage 3C stride-30 window. It is
retained in the role manifest, marked representation-unavailable, and excluded
from representation fitting/evaluation. No more specific cause is inferred.

## Experimental closure

Study 1 and Study 2 experimental work are closed. The paper freeze cannot be
used to reopen training, evaluation, feature design, threshold selection, model
search, or cohort access.
