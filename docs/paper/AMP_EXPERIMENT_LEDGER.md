# AMP authoritative experiment ledger

This ledger consolidates the meaningful experimental path. It is not a recommended
paper narrative: stages are provenance, while the manuscript is organized around
the scientific argument. `Main` means the result is needed to establish that
argument, `Supplement` means it supports interpretation or robustness, and
`Provenance` means it documents a decision or access boundary.

| Stage | Question | Data role | Method | Primary result | Decision | Placement |
|---|---|---|---|---|---|---|
| Reconstruction baseline / FC AE | Can normal-only reconstruction error discriminate theft-labelled consumption? | Historical train/validation/final protocol | 30-day fully connected autoencoder | Final window ROC/AP `0.5617/0.1144`; customer-p95 ROC/AP `0.5993/0.1686` | Weak discrimination; retain only as motivation | Main, one compact baseline |
| CNN/LSTM reconstruction | Does temporal model complexity rescue reconstruction scoring? | Same historical protocol | CNN and LSTM autoencoders | Final window ROC/AP: CNN `0.5557/0.1048`, LSTM `0.5577/0.1044` | No improvement over FC | Supplement |
| Stage 13 | Is there supervised signal despite coarse labels? | Historical train/validation; disclosed final-cohort access | Balanced window HGB, customer mean aggregation | Final window ROC/AP `0.6577/0.2053`; customer ROC/AP `0.7023/0.2193` | Supervised signal exists; creates mandatory historical-access disclosure | Main, compact diagnostic/disclosure |
| Stage 14 | Do behavioural summaries improve over raw windows? | Historical train/validation | SUMMARY_17 window features with HGB | Customer ROC/AP `0.7387/0.2613`, gains `+0.0207/+0.0204` over raw | Adopt behavioural summaries | Supplement |
| Stage 15 | Do frozen FC reconstruction/latent features add to summaries? | Historical train/validation | SUMMARY_17 plus FC-derived features | Best hybrid change was negligible and uncertain | Reject FC hybrid | Supplement |
| Stage 16 | Does direct customer modelling improve on window scoring? | Historical train/validation | FULL_48 customer representation with HGB | ROC/AP `0.7932/0.3763`; gains `+0.0545/+0.1150` over Stage 14 | Customer is the prediction unit | Supplement; motivates FULL_48 |
| Stage 17A | How are development, confirmation, and final roles governed? | Original train split into DEV_TRAIN/DEV_VAL; original validation/final retained | Deterministic seed-42 role manifest | All role overlaps zero; one DEV_TRAIN customer lacks a usable window | Freeze research hierarchy | Provenance |
| Stage 17B | Which conventional learner is the strongest FULL_48 benchmark? | Fit DEV_TRAIN; evaluate DEV_VAL | Logistic regression, HGB, LightGBM | HGB ROC/AP `0.7888/0.3598`; strong learners effectively equivalent | Freeze HGB comparator | Main comparator |
| Stage 17C | Does frozen FULL_48 confirm? | CONFIRMATION, one governed gate | Frozen Stage 17B HGB | ROC/AP `0.7891/0.3791` | Strong confirmation | Main comparator context |
| Stage 18 | Is FULL_48 robust across customer strata? | Saved DEV_VAL predictions | Prespecified history/consumption strata | History ROC `0.766–0.801`; consumption ROC `0.714–0.832`; low-consumption recall `0.172` | Generally robust with material subgroup limitation | Supplement; limitation in main text |
| Stage 19 | Which semantic feature families matter? | DEV_TRAIN/DEV_VAL | Family-only and leave-one-family-out ablations | Level/distribution strongest; non-level families retain `0.7635/0.3258` | Strong but redundant signal | Supplement |
| Stage 20A/B | Do multiplicity, pooling, or loss placement explain the formulation gain? | Design, then DEV_TRAIN/DEV_VAL | Matched formulation controls | No prespecified mechanism isolated; Q90 showed AP sensitivity only | Mixed/inconclusive; no causal mechanism claim | Supplement |
| Stage 21 | Is the result classifier-specific? | DEV_TRAIN/DEV_VAL | HGB, LightGBM, fixed ensemble and conventional references | HGB `0.7888/0.3598`; LightGBM `0.7854/0.3580`; fixed mean `0.7889/0.3617` | Strong boosting learners essentially equivalent | Supplement |
| Stage 22 | Can correlation redundancy be safely removed? | Selection from DEV_TRAIN; evaluate DEV_VAL | Label-free 48-to-26 feature compaction | Compact-minus-full ROC/AP `-0.0121/-0.0218` with negative intervals | Retain FULL_48 | Supplement |
| Stage 23A | What is the frozen Study 1 final protocol? | Aggregate evidence only | Governance and preregistration | FINAL_FULL_48 and final-fit policy sealed | Study 1 development closed | Provenance |
| Stage 23B | Does refitted Study 1 FULL_48 survive final evaluation? | DEV_TRAIN+DEV_VAL fit; one final evaluation | Refitted FULL_48 HGB | ROC/AP `0.7925/0.3702` | Strong stable final replication; Study 1 closed | Supplement / prior-study context |
| Stage 24A | Is customer-level weakly supervised MIL feasible and governed? | DEV_TRAIN/DEV_VAL metadata only | Bag audit and MIL preregistration | Feasible bags; protected cohorts closed | Authorize bounded MIL experiment | Provenance |
| Stage 24B | Can one pooled latent vector represent a customer's window bag? | DEV_TRAIN/DEV_VAL | Mean, max and gated-attention MIL | Best selected arm MAX_MIL `0.7567/0.3146`, below FULL_48 by `-0.0322/-0.0452` | Outcome C, weak result; single-vector bottleneck remains | Main diagnostic |
| Stage 25A | Is distribution-preserving latent aggregation a distinct hypothesis? | Committed development evidence only | Fixed hypothesis and controls | Multi-statistic pooling judged distinct but bounded | B, go with limitations | Provenance |
| Stage 25B | Does learned latent distribution pooling recover customer information? | DEV_TRAIN fit; DEV_VAL evaluation | DIRECT, latent-distribution HGB, end-to-end control | LATENT `0.7908/0.3678`; FULL_48 `0.7888/0.3598`; DIRECT `0.7882/0.3492` | Representation parity; one confirmation justified | Main |
| Stage 26 | Does the frozen learned representation independently confirm? | CONFIRMATION evaluation only | Frozen five-seed LATENT, FULL_48 and DIRECT | LATENT `0.8118/0.4153`; gains over FULL_48 `+0.0227/+0.0362`, DIRECT `+0.0248/+0.0596` | Positive replication; anchor earned | Main |
| Stage 27A | What exactly enters terminal evaluation? | Aggregate evidence only | No-refit protocol and claim criteria | Frozen DEV_TRAIN models, arithmetic five-seed ensemble, both controls | Ready with disclosed governance limitation | Provenance |
| Stage 27B | Does the learned-distribution finding survive terminal evaluation? | Final evaluation cohort with disclosed historical baseline access | Frozen five-seed LATENT, FULL_48 and DIRECT | LATENT `0.8117/0.4219`; gains over FULL_48 `+0.0139/+0.0365`, DIRECT `+0.0168/+0.0534`; all four paired lower bounds positive | FINAL A; anchor survives; Study 2 closed | Main |

## Placement summary

- **Main paper:** one compact reconstruction result; Stage 13 supervised-signal
  diagnostic and disclosure; Stage 17B/17C FULL_48 comparator; Stage 24B
  single-vector MIL limitation; and the Stage 25B/26/27B DIRECT-versus-LATENT
  sequence, final paired uncertainty, and seed robustness.
- **Supplement:** CNN/LSTM details; Stages 14–16; subgroup, feature-family,
  formulation, learner and compaction analyses; Study 1 terminal result; full
  per-seed tables and secondary metrics.
- **Internal provenance:** Stages 17A, 23A, 24A, 25A and 27A. Their material
  governance facts belong in Methods, but they are not experimental result
  sections.

## Coursework checkpoint documentation

- **Stage 37 --- October 10 methodology submission:** Documentation and
  integration only; no experiment was reopened. The independent checkpoint at
  `report/checkpoints/october_10/` preserves the submitted September problem
  statement, literature survey, research gap, objectives, citations, and author
  details, then adds a concise methodology section covering the implemented
  pipeline, dataset and preprocessing, AMP-LDP and comparison models, frozen
  training/evaluation roles, and customer-level metrics. Method facts are drawn
  from `AMP_METHOD_FREEZE.md`, `docs/preprocessing_decisions.md`, the full report
  methodology, and committed Stage 14/16/17A/17B/24A/25A/25B/27A/27B protocol
  artifacts. Validation is static when no LaTeX toolchain is available; the
  self-contained Overleaf ZIP is the compilation handoff.
