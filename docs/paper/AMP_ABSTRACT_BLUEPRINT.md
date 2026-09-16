# AMP structured abstract blueprint

This is a content blueprint, not polished abstract prose.

## Problem (1–2 sentences)

- Electricity-theft screening from smart-meter histories is commonly evaluated
  with a customer label, yet the measurements form long temporal sequences.
- The evidence question is customer-level ranking under class imbalance, not
  localization of a verified theft interval.

## Label-granularity issue

- SGCC `FLAG` is customer-level and contains no authoritative theft-event timing.
- Treat 30-day windows as unlabeled instances; copying `FLAG` to every window is
  propagated weak supervision, not verified window ground truth.

## Method

- Introduce AMP–Latent Distribution Profiling (AMP-LDP).
- Convert each 30-day window to SUMMARY_17, encode `17→32→16`, aggregate mean,
  stabilized population SD, q10, median, q90, and maximum across every latent
  dimension, classify the 96-dimensional customer profile with HGB, and average
  probabilities across five frozen seeds.
- Compare with engineered FULL_48 and a DIRECT control using identical
  distribution statistics without learned encoding.

## Evaluation setup

- Fit all Study 2 models on representation-available DEV_TRAIN customers.
- Report DEV_VAL (`n=5,931`), independent CONFIRMATION (`n=6,355`), and one
  terminal final evaluation (`n=6,356`).
- Use customer ROC-AUC/AP and 2,000 paired customer bootstraps for final effects.

## Central quantitative result

- Final AMP-LDP: ROC-AUC `0.8117`, AP `0.4219`.
- Versus FULL_48: `+0.0139` ROC (95% CI `[+0.0005,+0.0275]`) and `+0.0365` AP
  (`[+0.0080,+0.0652]`).
- Versus DIRECT: `+0.0168` ROC (`[+0.0063,+0.0279]`) and `+0.0534` AP
  (`[+0.0274,+0.0779]`).
- The method ordering persisted on DEV_VAL and CONFIRMATION; four of five final
  seeds met both frozen FULL_48 parity margins, with no seed-collapse trigger.

## Contribution / implication

- Distribution-preserving aggregation resolves the principal information
  bottleneck observed with evaluated single-vector MIL.
- Learned window representations add reproducible customer-level discrimination
  beyond applying the same aggregation directly to handcrafted window summaries.

## Limitation / disclosure sentence

- Evidence is internal to SGCC and does not establish temporal theft
  localization or external utility generalization; the final evaluation cohort
  also had disclosed historical Stage 13 access for an obsolete window baseline.
