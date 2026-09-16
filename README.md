# Anomaly-Based Electricity Theft Detection Using Autoencoder and GAN Models

This project studies anomaly-based electricity theft detection from SGCC daily
smart-meter consumption histories. The planned workflow compares temporal
autoencoder and GAN-based approaches while preserving customer traceability and
preventing preprocessing leakage.

## Repository structure

    data/raw/                    Immutable source data
    data/interim/                Validated Stage 1 outputs
    data/processed/              Customer-level Stage 2 split outputs
    data/windows/                Reproducible Stage 3 window manifests
    data/windows/stride_30/usable/ Stage 3C baseline-eligible manifests
    data/windows/stride_15/usable/ Stage 7 overlap-ablation manifests
    results/missingness_analysis/ Stage 3B quality evidence and recommendation
    docs/                        Master technical decision record
    src/preprocessing/           Reproducible preprocessing stages
    src/analysis/                Reproducible descriptive analyses
    src/data/                    Lazy model-input datasets
    src/models/                  Neural-network architectures
    src/training/                Reproducible training entry points
    models/preprocessing/        Fitted train-only preprocessing artifacts

## Pipeline status

- Stage 1 — Cleaning: complete
- Stage 2 — Customer split and scaling: complete
- Stage 3 — Configurable temporal windowing: complete for stride 30
- Stage 3B — Missingness quality analysis: complete
- Stage 3C — Window quality filtering: complete for stride 30
- Stage 4 — Baseline Autoencoder: trained
- Stage 5 — Validation threshold: frozen
- Stage 6 — Frozen test evaluation: complete
- Stage 7 — Stride-15 Autoencoder ablation: preparation and manual training complete
- Stage 8 — Stride-15 threshold, frozen test, and controlled comparison: complete
- Stage 9 — 1D CNN Autoencoder: implementation and manual training complete
- Stage 10 — CNN threshold, frozen test, and FC comparison: complete
- Stage 11–12 — LSTM Autoencoder: trained, thresholded, and frozen-test evaluated
- Stage 13 — Leakage-safe supervised signal diagnostic: complete
- Stage 14 — Behavioural representation analysis: complete (validation only; test closed)
- Stage 15 — Frozen-FC hybrid ablation: complete (FC rejected; test closed)
- Stage 16 — Direct customer-level behavioural modeling: complete (direct HGB adopted; test closed)
- Stage 17A — Development/confirmation split governance: complete (protocol frozen; no modeling)
- Stage 17B — Frozen customer-feature learner benchmark: complete (HGB retained; confirmation closed)
- Stage 17C — Confirmation Gate #1: complete (strong confirmation; final test closed)
- Stage 18 — Customer-stratum robustness: complete (generally robust; low-consumption threshold limitation)
- Stage 19 — Customer feature-family ablation: complete (strong signal; high family redundancy)
- Stages 20A–20B — Label/objective formulation controls: complete (mixed/inconclusive)
- Stage 21 — Supervised model-family benchmark: complete (representation robust across strong learners)
- Stage 22 — Compact representation test: complete (FULL_48 retained; development frozen)
- Stage 23A — Research freeze and final protocol: complete
- Stage 23B — Frozen final evaluation: complete (strong stable final replication)


Stage 3B compares candidate missingness thresholds without changing the
canonical manifests. See the
[missingness quality report](results/missingness_analysis/report.md) for the
window, customer, class, temporal-position, and consecutive-run findings.

Stage 3C adopts the minimum baseline eligibility rule
`missing_fraction < 1.0` (equivalently `missing_count < 30`): a window is
retained when at least one of its 30 source readings was observed. The filtered
manifests are exact subsets under `data/windows/stride_30/usable/`; the original
Stage 3 manifests remain canonical and unchanged. The baseline model input will
be the scaled consumption sequence only. The missingness mask,
`missing_fraction`, `FLAG`, and `CONS_NO` are excluded from model inputs.

Stage 4 implements a normal-only, fully connected Autoencoder with architecture
`30 → 16 → 8 → 4 → 8 → 16 → 30`. It uses usable stride-30 `FLAG=0` training
windows, normal-only validation MSE for early stopping, Adam at `0.001`, and
MSE loss. Training completed on CPU and saved the best checkpoint at
`models/autoencoder/ae_stride30_baseline.pt` without using the test set.

Stage 5 selects the baseline threshold from validation reconstruction errors
only. The adopted normal-validation 95th-percentile threshold is stored at
`models/autoencoder/ae_stride30_threshold.json`; detailed evidence is under
`results/autoencoder/stride_30/threshold_analysis/`.

Stage 6 applies that frozen threshold once to the reserved test split without
retraining or tuning. The final stride-30 Autoencoder baseline record is under
`results/autoencoder/stride_30/test_evaluation/`.

The stride-30 fully connected Autoencoder is the complete frozen baseline.
The stride-15 Autoencoder has completed normal-only training, validation-only
P95 threshold selection, and one-time frozen test evaluation under the same
methodology. The controlled comparison is at
`results/autoencoder/stride_comparison/stride_comparison_report.md`. Stride 15
is effectively equivalent and slightly worse at the window level; lower
reconstruction loss did not improve core anomaly discrimination. Stride 7 is
not recommended next and remains unexecuted.

The compact stride-30 1D CNN Autoencoder has completed manual normal-only
training, validation-only P95 threshold selection, and frozen test evaluation.
Its best normal-validation MSE is `0.030209`, 30.70% below the FC model, but its
frozen test ROC-AUC/PR-AUC/F1 are `0.555689`/`0.104788`/`0.124173`, all below
the FC baseline's `0.561688`/`0.114411`/`0.142676`. The controlled comparison
is at `results/model_comparison/fc_vs_cnn/fc_vs_cnn_report.md`; the FC model
remains the preferred frozen baseline after the three-way comparison. The LSTM
completed normal-only training, validation P95 thresholding, and frozen test
evaluation, but did not improve anomaly discrimination: its test
ROC-AUC/PR-AUC/F1 are `0.557689`/`0.104403`/`0.115679`. The full controlled
comparison is at
`results/model_comparison/fc_vs_cnn_vs_lstm/architecture_comparison_report.md`.

Files under data/raw/ are immutable source data and must never be overwritten.

Run the implemented stages from the repository root:

    python -m src.preprocessing.clean_data
    python -m src.preprocessing.split_and_scale
    python -m src.preprocessing.generate_windows --stride 30
    python -m src.analysis.analyze_window_missingness
    python -m src.preprocessing.filter_windows --stride 30
    python -m src.training.train_autoencoder --stride 30
    python -m src.analysis.analyze_autoencoder_threshold
    python -m src.evaluation.evaluate_autoencoder_test

Stride-15 reproduction commands are:

    python -m src.training.train_autoencoder --stride 15
    python -m src.analysis.analyze_autoencoder_threshold --stride 15
    python -m src.evaluation.evaluate_autoencoder_test --stride 15
    python -m src.analysis.compare_autoencoder_strides

Stride 7 remains optional, ungenerated, and not recommended for this baseline.

CNN reproduction commands are:

    python -m src.training.train_cnn_autoencoder --stride 30
    python -m src.analysis.analyze_cnn_autoencoder_threshold
    python -m src.evaluation.evaluate_cnn_autoencoder_test
    python -m src.analysis.compare_fc_cnn

LSTM training must be run manually from the repository root:

    python -m src.training.train_lstm_autoencoder --stride 30
    python -m src.analysis.analyze_lstm_autoencoder_threshold
    python -m src.evaluation.evaluate_lstm_autoencoder_test
    python -m src.analysis.compare_fc_cnn_lstm

Stage 13 tests the existing usable stride-30 consumption windows with a dummy
classifier, unweighted/balanced logistic regression, and unweighted/balanced
HistGradientBoosting. Validation-only selection chooses balanced
HistGradientBoosting at window ROC-AUC/average precision `0.662847`/`0.203618`;
its frozen test values are `0.657693`/`0.205305`. Mean-probability customer
aggregation reaches frozen test ROC-AUC `0.702314`. This establishes moderate
supervised statistical signal beyond reconstruction error, not confirmed
per-window theft events or deployment readiness. See
`results/supervised/stage_13/stage_13_report.md`.

Stage 13 reproduction is deliberately two-phase so the validation decision is
frozen before test access:

    python -m src.training.train_supervised_baselines
    python -m src.evaluation.evaluate_supervised_test

Stage 14 pre-registers 17 interpretable window summaries and compares raw-only,
summary-only, and combined representations with the fixed Stage 13 logistic and
HistGradientBoosting settings. Unweighted summary-only HistGradientBoosting is
the validation-only winner at customer ROC-AUC/average precision
`0.738697`/`0.261320`, improving on its raw-only values by
`+0.020741`/`+0.020351`. Raw-plus-summary is slightly weaker than summary-only.
The test set is closed and was not accessed. Full evidence is at
`results/supervised/stage_14/stage_14_report.md`.

    python -m src.analysis.run_stage14_representation

Stage 15 extracts six reconstruction summaries and the four-unit latent vector
from the unchanged frozen FC autoencoder, then compares FC-only and hybrid
representations with the Stage 14 HGB configuration. `SUMMARY_PLUS_FC` changes
customer validation ROC-AUC/AP/F1 by `-0.000243`/`+0.001723`/`-0.001506` versus
`SUMMARY_ONLY`; paired customer-bootstrap intervals include zero for both
ranking deltas. FC features are largely redundant with behavioural variability
summaries, so the FC branch is rejected from the current detector. Test was not
accessed. See `results/supervised/stage_15/stage_15_report.md`.

    python -m src.analysis.run_stage15_hybrid_ablation

Stage 16 pre-registers 48 interpretable customer aggregates derived only from
the frozen 17 Stage 14 window summaries, then compares direct customer logistic
regression and HistGradientBoosting with the frozen Stage 14 window-HGB → mean
customer probability baseline. Direct customer HGB reaches validation customer
ROC-AUC/AP/F1 `0.793210`/`0.376278`/`0.402634`, with exact gains
`+0.054513`/`+0.114958`/`+0.099385`. Paired 2,000-customer bootstrap 95%
intervals are `[+0.037960, +0.070310]` for ROC-AUC and
`[+0.082130, +0.150546]` for AP. Decision A adopts and freezes direct customer
HGB as the next supervised baseline. Test was not accessed. See
`results/supervised/stage_16/stage_16_report.md`.

    python -m src.analysis.run_stage16_customer_aggregation

Stage 17A freezes the post-exploration evaluation hierarchy without changing the
historical split: original train becomes `DEV_TRAIN`/`DEV_VAL`, original
validation becomes `CONFIRMATION`, and original test becomes `FINAL_TEST`.
The deterministic customer-level 80/20 development split uses train-only `FLAG`
stratification and seed 42. Routine work must fit on `DEV_TRAIN` and select on
`DEV_VAL`; confirmation is reserved for explicit frozen-methodology gates and
final test remains sealed. No models were trained or evaluated. See
`results/governance/stage_17a/stage_17a_protocol_report.md`.

    python -m src.governance.create_research_protocol

Stage 17B reconstructs the unchanged 48-feature direct-customer representation
using only `DEV_TRAIN`/`DEV_VAL` and compares one fixed configuration each for
logistic regression, HistGradientBoosting, and LightGBM. HGB remains the AP-first
winner at DEV_VAL ROC-AUC/AP `0.788831`/`0.359795`; LightGBM is statistically
compatible at `0.785410`/`0.358024`. Decision B retains HGB for simplicity and
internally replicates the strong direct-customer formulation. A confirmation
gate is recommended but was not executed; `CONFIRMATION` and `FINAL_TEST` stayed
closed. See `results/supervised/stage_17b/stage_17b_report.md`.

    python -m src.analysis.run_stage17b_learner_benchmark

Stage 17C opens `CONFIRMATION` once for the exact saved Stage 17B HGB pipeline,
without retraining or threshold adjustment. Confirmation ROC-AUC/AP are
`0.789084`/`0.379058`, with bootstrap 95% intervals
`[0.768115, 0.810559]`/`[0.337648, 0.422308]`. Development-to-confirmation
differences are compatible with sampling variation, and the direct formulation
retains a clear paired advantage over frozen Stage 14. Decision A records strong
confirmation and closes confirmation again. `FINAL_TEST` was not accessed. See
`results/supervised/stage_17c/stage_17c_report.md`.

    python -m src.evaluation.evaluate_stage17c_confirmation

Stage 18 is an analysis-only robustness audit of the exact saved Stage 17B HGB
`DEV_VAL` probabilities. Its train-derived, tie-aware history groups are at most
18, 19–33, and exactly 34 usable windows; consumption groups use tertiles of the
existing `mean__customer_median` feature. All history groups retain ROC-AUC
`0.766–0.801`, while consumption ROC-AUC ranges from `0.714` to `0.832`.
Low-consumption recall is `0.172` at the unchanged global threshold, producing
Decision B: generally robust with a material subgroup limitation. The Stage 14
subgroup comparison is skipped because its original-train fit includes current
`DEV_VAL` customers. No model is retrained, and `CONFIRMATION` and `FINAL_TEST`
remain closed. See `results/supervised/stage_18/stage_18_report.md`.

    python -m src.analysis.run_stage18_robustness --prepare
    python -m src.analysis.run_stage18_robustness --evaluate

Stage 19 pre-registers an exhaustive semantic partition of the 48 frozen customer
features, then fits family-only and leave-one-family-out models with the unchanged
Stage 17B HGB configuration on `DEV_TRAIN` and evaluates only on `DEV_VAL`.
`FULL_48` exactly reproduces Stage 17B ROC-AUC/AP `0.788831`/`0.359795`.
Level/distribution is the strongest standalone and only conditionally necessary
family, but the five non-level families together retain `0.763537`/`0.325818`;
the signal is level-sensitive rather than narrowly level-dominated. Decision B
records strong but highly redundant family-level signal. No compact model or
tuning is performed, and `CONFIRMATION` and `FINAL_TEST` remain closed. See
`results/supervised/stage_19/stage_19_report.md`.

    python -m src.analysis.run_stage19_family_ablation --prepare
    python -m src.analysis.run_stage19_family_ablation --evaluate

Stage 20A documents a controlled label/objective-formulation design and makes
explicit that SGCC's customer-level `FLAG` cannot identify temporal window-label
noise. Stage 20B then freezes and runs three DEV-only controls using the exact 17
window summaries: equal total customer weighting, a fixed four-rule probability
aggregation panel, and matched window-versus-customer logistic loss placement.
Customer balancing produces small uncertain mean-pooled HGB changes; Q90 improves
AP but not ROC-AUC with corresponding certainty; and C_LOGIT's small advantage
over W_LOGIT is also uncertain. Decision E is mixed/inconclusive: no single
formulation mechanism is isolated, no temporal label-noise claim is made, and the
AMP development experiment matrix should now be frozen. `CONFIRMATION` and
`FINAL_TEST` were not accessed. See
`results/supervised/stage_20b/stage_20b_report.md`.

    python -m src.analysis.run_stage20b_label_formulation_controls --prepare
    python -m src.analysis.run_stage20b_label_formulation_controls --evaluate

Stage 21 freezes an independent strong supervised benchmark on the same 48
customer features for every model. Exact HGB reproduction passes before the new
panel is interpreted. HGB (`0.788831`/`0.359795` ROC-AUC/AP), LightGBM
(`0.785410`/`0.358024`), and their fixed arithmetic probability mean
(`0.788887`/`0.361659`) are statistically and practically close; neither
LightGBM nor the ensemble materially beats HGB under paired bootstrap. Random
forest is modestly lower and logistic regression clearly lower. Decision B
supports classifier robustness of the customer representation rather than a
special HGB-only result. With Stage 19 redundancy, the only recommended
remaining development work is one pre-specified compact-representation
experiment. `CONFIRMATION` and `FINAL_TEST` were not accessed. See
`results/supervised/stage_21/stage_21_report.md`.

    python -m src.analysis.run_stage21_supervised_benchmark --prepare
    python -m src.analysis.run_stage21_supervised_benchmark --evaluate

Stage 22 performs the final authorized development experiment: deterministic,
label-free, within-family correlation compaction of the frozen 48 customer
features. Using only DEV_TRAIN Spearman correlations at `|rho| >= 0.95`, the
first feature in frozen Stage 16 order is retained from each connected component.
The sealed rule keeps 26 features (45.83% reduction) and preserves all six
families, but ROC-AUC/AP fall from `0.788831`/`0.359795` to
`0.776787`/`0.337326`. Paired intervals support a real loss, producing Decision
C: correlation redundancy is not safely removable. `FINAL_FULL_48` remains the
recommended method because it performs better and already has confirmation
evidence. The AMP development research matrix is now frozen; no Stage 23 model
development is recommended. `CONFIRMATION` and `FINAL_TEST` were not accessed.
See `results/supervised/stage_22/stage_22_report.md`.

    python -m src.analysis.run_stage22_compact_representation --prepare
    python -m src.analysis.run_stage22_compact_representation --evaluate

Stage 23A freezes the research contribution and the complete Stage 23B final-
evaluation protocol without fitting or evaluating a model. The final method is
the exact Stage 16 `FINAL_FULL_48` direct-customer representation with the
unweighted Stage 17B HistGradientBoosting configuration. Its one final refit
uses the 29,655 representation-available `DEV_TRAIN + DEV_VAL` customers;
`CONFIRMATION` remains excluded to preserve Stage 17C as independent evidence.
The four label-free persistence q90 references are recomputed from final-
training windows only, and the Stage 17B threshold
`0.20070617709942135` remains fixed as a secondary operating point.

The authoritative specification is
`results/supervised/stage_23a/final_evaluation_protocol.json`, SHA-256
`dbaa2035586c3e9f8f4ed2654d5a129acb2d767d14ded5b168dfc740d8943993`.
It pre-registers primary customer ROC-AUC/AP, secondary fixed-threshold metrics,
2,000 seed-42 customer bootstrap replicates, execution order, artifact paths,
prohibited adaptations, and mandatory abort conditions. Stage 23B requires
zero methodological choices.

The cohort now called `FINAL_TEST` had disclosed historical Stage 13 access for
an obsolete window baseline; it must therefore be described as the **final
evaluation cohort with disclosed historical baseline access**, not a pristine
or never-seen test set. Stage 13 and Stage 23B results must be reported as
historical exploratory baseline evidence and frozen final-pipeline evaluation,
respectively. No row-level `CONFIRMATION` or `FINAL_TEST` data was accessed in
Stage 23A. Full freeze rationale, evidence chain, claims matrix, and final method
are under `results/supervised/stage_23a/`.

**AMP MODEL DEVELOPMENT IS CLOSED AFTER STAGE 22.** Stage 23B cannot trigger
further tuning or method revision; future modelling is a separate follow-up
study.

Stage 23B executed the frozen protocol once. All pre-access gates passed before
the final cohort was opened; the model was fit once on the 29,655 eligible
`DEV_TRAIN + DEV_VAL` customers and serialized before the access gate. On all
6,356 eligible `FINAL_TEST` customers (542 theft), final ROC-AUC was
`0.792513` (95% customer-bootstrap CI `0.771214–0.812647`) and Average Precision
was `0.370159` (CI `0.328475–0.414295`). At the unchanged secondary threshold
`0.20070617709942135`, precision/recall/F1 were
`0.387543`/`0.413284`/`0.400000`. The result is classified descriptively as
**A — STRONG STABLE FINAL REPLICATION** and was accepted without adaptation.

The cohort is reported as a **final evaluation cohort with disclosed historical
baseline access** because Stage 13 evaluated an obsolete window baseline on the
same population. Full audit, matrices, predictions, bootstrap outputs, and the
final report are under `results/supervised/stage_23b/`.

**AMP FROZEN EXPERIMENT COMPLETE.**

