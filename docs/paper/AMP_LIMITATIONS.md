# AMP limitations and disclosures freeze

The order below reflects manuscript priority.

1. **Single source and no external validation.** All evidence comes from one
   public SGCC dataset. The studies do not establish transfer to another utility,
   geography, meter population, time period, or operating environment.
2. **Customer labels lack verified event timing.** `FLAG` is customer-level;
   no authoritative theft-period labels exist. The method supports
   customer-level discrimination only and cannot validate individual windows,
   localize theft periods, or estimate a window-level theft probability.
3. **Final-cohort historical access.** The final cohort was used in Stage 13 for
   an obsolete supervised window baseline. Study 2 was designed and frozen
   later without Study 2 row-level access, but the cohort is not a pristine
   untouched test set.
4. **Representation and ensemble scope.** AMP-LDP's primary prediction averages
   five frozen seeds and was numerically stronger than the individual seeds.
   Four of five final seeds met both parity margins. The terminal claim therefore
   applies to the preregistered ensemble, not an arbitrary single run.
5. **Published SGCC protocols are heterogeneous.** Differences in splitting,
   prevalence, preprocessing, label propagation, evaluation unit, and metric
   definitions prevent unqualified numerical ranking against the literature.
6. **No causal or deployment conclusion.** Predictive associations do not
   identify theft causes. Runtime, memory, calibration, investigation capacity,
   cost, robustness to distribution shift, and field deployment were not
   established for the final method.
7. **Comparator asymmetry in information content.** FULL_48 includes engineered
   chronology and persistence features unavailable to the permutation-invariant
   distribution methods. Training populations are symmetric in Study 2, but the
   representations answer different design questions.

## Required final-cohort disclosure

> The final evaluation cohort had been accessed historically in Stage 13 for an obsolete supervised window-level baseline. The learned latent-distribution hypothesis, architecture, pooling rule, five-seed ensemble, controls, and confirmation decision reported in Study 2 were developed and frozen later without Study 2 row-level access to this cohort. We therefore treat it as a governed final evaluation cohort with disclosed historical baseline access, not as a pristine untouched test set.

Placement is mandatory in three locations:

- **Methods:** immediately after the split/governance description, in full.
- **Results:** adjacent to the first final-evaluation table reference, at least
  by explicit cross-reference to the full Methods disclosure.
- **Limitations:** summarize the historical access and its implication for strict
  independence. It must not appear only in supplementary material.

## Literature-comparison boundary

Published SGCC metrics may contextualize the task but cannot establish
superiority unless customer populations, prevalence, preprocessing, split
membership, evaluation unit, label handling, and metric definitions are shown
to be protocol-equivalent. **No direct state-of-the-art claim is permitted.**

The Stage 28 instructions mention STL-Net natural-prevalence ROC approximately
`0.717` and PR-AUC approximately `0.274`. No supporting citation or tracked
repository source was found. Those values remain explicitly unverified and are
excluded from paper-facing quantitative comparisons until a citable source is
established. No citation is inferred or invented here.
