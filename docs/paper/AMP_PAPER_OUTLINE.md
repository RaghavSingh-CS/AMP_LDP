# AMP paper outline and title direction

## Logical manuscript story

1. **Problem.** Electricity-theft screening is imbalanced and operationally
   customer-level, while smart-meter histories are long and heterogeneous.
2. **Label granularity.** SGCC provides a customer `FLAG`, not authoritative
   theft-event timing. Temporal windows are therefore unlabeled instances, and
   copying `FLAG` to every window creates propagated weak labels.
3. **Why naïve approaches are inadequate.** Normal-only FC/CNN/LSTM
   reconstruction produced weak discrimination, and supervised Stage 13 showed
   that signal existed but remained limited under window scoring. Later
   single-vector mean, max, and attention MIL also left a substantial gap.
4. **Customer distribution hypothesis.** A customer may be characterized by
   the distribution of behaviours across windows—location, variability, tails,
   and extremes—rather than one pooled latent vector or a presumed theft window.
5. **Proposed method.** AMP-LDP encodes SUMMARY_17 windows, aggregates six
   statistics over each latent dimension, classifies the 96-dimensional customer
   profile with HGB, and averages five frozen seed probabilities.
6. **Controls.** FULL_48 tests against a strong engineered customer profile;
   DIRECT applies identical distribution statistics to SUMMARY_17 and isolates
   learned-encoding value; Stage 24B diagnoses the single-vector bottleneck.
7. **Governed evaluation.** DEV_TRAIN fitting, DEV_VAL development,
   CONFIRMATION replication, then one terminal evaluation with the historical
   Stage 13 access disclosed. No Study 2 refitting or threshold optimization.
8. **Main result.** AMP-LDP ranks first on both metrics across all three roles.
   In final evaluation it scores `0.8117` ROC / `0.4219` AP, with paired gains
   over FULL_48 of `+0.0139/+0.0365` and over DIRECT of `+0.0168/+0.0534`;
   all four corresponding lower confidence bounds exceed zero.
9. **Limits.** Evidence is one-dataset, customer-level, non-causal, and not
   external validation; the final cohort has disclosed historical baseline
   access; published SGCC protocols are not assumed comparable.

The manuscript must not be written as a sequence of repository stages. Stage
identifiers belong in provenance notes and the evidence index.

## Main-paper experiment set and necessity

| Experiment | Why it is needed |
|---|---|
| Compact FC reconstruction result, with CNN/LSTM conclusion in prose | Establishes why the project moved beyond anomaly reconstruction without turning the paper into an architecture survey |
| Stage 13 supervised customer-aggregated diagnostic | Shows usable supervised signal and transparently establishes the historical final-cohort access |
| Frozen FULL_48 benchmark across governed roles | Provides the strongest engineered customer-profile comparator and verifies that AMP-LDP is not compared only with weak baselines |
| Stage 24B single-vector MIL arms | Demonstrates the information bottleneck that motivates distribution-preserving aggregation |
| DIRECT versus AMP-LDP on DEV_VAL | Isolates distribution aggregation from learned window encoding during development |
| AMP-LDP, FULL_48 and DIRECT across DEV_VAL/CONFIRMATION/final | Establishes persistence under the complete governed sequence |
| Final paired effects and seed-collapse audit | Supports uncertainty and robustness claims rather than relying on rank alone |

## Supplementary experiment set

- Full FC/CNN/LSTM reconstruction tables and thresholds.
- Stages 14–16 behavioural-summary, hybrid, and direct-customer evolution.
- Stage 18 subgroup robustness and low-consumption limitation.
- Stage 19 family ablation, Stage 20B formulation controls, Stage 21 learner
  comparison, and Stage 22 compaction result.
- Study 1 Stage 23B terminal result, clearly labelled as a different refitted
  FULL_48 pipeline.
- Complete per-seed metrics, Brier scores, secondary operating-point metrics,
  and extended bootstrap diagnostics.

## Internal-provenance-only set

Stages 17A, 23A, 24A, 25A, and 27A are governance/design records rather than
result sections. Their split rules, freeze decisions, hashes, access history,
and no-rescue policies inform Methods and reproducibility materials but do not
become a sequential manuscript narrative.

## Working titles ranked by clarity and faithfulness

1. **Weakly Supervised Latent Distribution Profiling for Customer-Level Electricity Theft Detection** — recommended.
2. **AMP-LDP: Distribution-Preserving Smart-Meter Profiling under Customer-Level Supervision**
3. **From Unlabeled Windows to Customer Profiles: Learned Distribution Aggregation for Electricity Theft Detection**
4. **Learning Customer Distributions from Smart-Meter Windows for Weakly Supervised Electricity Theft Detection**
5. **Automated Meter Profiling with Learned Window Distributions for Electricity Theft Discrimination**

The recommended title is the most explicit about supervision, representation,
prediction unit, and task without implying temporal localization, external
generalization, or state-of-the-art status.
