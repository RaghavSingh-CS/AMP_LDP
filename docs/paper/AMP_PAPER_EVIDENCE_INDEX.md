# AMP paper evidence index

This index maps paper-facing facts to authoritative committed aggregate
artifacts. Paths are repository-relative. Row-level prediction files are never
required to support the manuscript values below.

## Core protocol and checkpoint registry

| Item | Authoritative artifact | SHA-256 / frozen value |
|---|---|---|
| Stage 25B execution specification | `results/supervised/stage_25b/stage25b_execution_spec.json` | `006c781899f74f0f62d01282fbf4164ac3ee27d8a61e3f2ea5fcb09b9e445470` |
| Stage 26 confirmation protocol | `results/supervised/stage_26/stage26_confirmation_protocol.json` | `7207a75dfd31d3254aaa3d4507eb9812b38ff87ecda09abfa62178572c8be09b` |
| Stage 27B preregistration | `results/supervised/stage_27a/stage27b_preregistration.json` | `ed2457e3a6eaf4b683c3fd8aaf89cf1f1fca63500a68f0427279813533e92702` |
| Study 2 FULL_48 comparator | `models/supervised/stage_17b/hist_gradient_boosting.joblib` | `67ed869858185d8f71d99b3dce8b358c43793178185e5aa50e6997d11b56974b` |
| DIRECT comparator | `models/supervised/stage_25b/direct_summary_distribution_hgb.joblib` | `fa11424a6eb5615bc3ce696860a59d992114d891bb9d56bb3423693c172b4a45` |
| Five encoder and latent-HGB hashes | `results/supervised/stage_27b/pre_final_access_audit.json` | Exact seed-42–46 inventory; all verified before access |
| Study 2 model/comparator policy | `results/supervised/stage_27a/final_model_freeze.json`; `final_comparator_freeze.json` | DEV_TRAIN-only, no-refit policy |

## Dataset, label and split facts

| Paper-facing fact | Value | Authoritative aggregate source |
|---|---:|---|
| Cleaned customers | 42,367 | `results/governance/stage_17a/protocol_summary.json` (`authoritative_customer_count`) and `docs/preprocessing_decisions.md` Stage 13 audit |
| Normal / theft-labelled | 38,755 / 3,612 | `results/supervised/stage_13/split_audit.json` |
| Overall prevalence | 8.5255% | `results/supervised/stage_13/split_audit.json` |
| Daily consumption dimensions | 1,034 | `docs/preprocessing_decisions.md`, preprocessing/window policy |
| Window length | 30 days | `docs/preprocessing_decisions.md`, Stage 3C / method specifications |
| DEV_TRAIN governed / available / positive governed / positive available | 23,725 / 23,724 / 2,023 / 2,022 | `results/governance/stage_17a/protocol_summary.json`; usable counts also in Stage 17B training summary |
| Unavailable customer policy and ID | No usable Stage 3C stride-30 window; retained and marked unavailable; ID recorded | `results/governance/stage_17a/protocol_summary.json` and `stage_17a_protocol_report.md` |
| DEV_VAL total / positive | 5,931 / 505 | `results/governance/stage_17a/protocol_summary.json` |
| CONFIRMATION total / positive | 6,355 / 542 | Same source; confirmed by `results/supervised/stage_26/confirmation_integrity.json` |
| Final total / positive | 6,356 / 542 | Same source; confirmed by `results/supervised/stage_27b/final_integrity.json` |
| Zero role overlap | All six pairwise overlaps zero | `results/governance/stage_17a/protocol_summary.json` |
| Customer-level label / no verified theft timing | `FLAG` is customer-level | `results/supervised/stage_13/test_metrics.json`, governance reports, and Stage 24–27 protocols |

## Historical and Study 1 results

| Claim / value | Authoritative aggregate source |
|---|---|
| FC/CNN/LSTM final window ROC/AP `0.5617/0.1144`, `0.5557/0.1048`, `0.5577/0.1044`; FC customer-p95 `0.5993/0.1686` | `results/model_comparison/fc_vs_cnn_vs_lstm/architecture_comparison.csv` |
| Stage 13 final window ROC/AP `0.6576929282/0.2053047543` | `results/supervised/stage_13/test_metrics.json` (`window_level`) |
| Stage 13 customer-mean ROC/AP `0.7023135402/0.2193165970` | Same artifact (`customer_mean_level`) |
| Stage 14 SUMMARY_17 `0.7387/0.2613`, raw gains `+0.0207/+0.0204` | `results/supervised/stage_14/validation_metrics.csv`; `customer_representation_deltas.csv` |
| Stage 15 hybrid rejection | `results/supervised/stage_15/validation_decision.json`; `validation_results_summary.json` |
| Stage 16 FULL_48 `0.7932/0.3763`, gains `+0.0545/+0.1150` | `results/supervised/stage_16/validation_metrics.csv`; `customer_deltas_vs_stage14.csv` |
| Stage 17B HGB `0.7888308219/0.3597945903` | `results/supervised/stage_17b/validation_decision.json` |
| Stage 17C HGB `0.7890840482/0.3790578559` | `results/supervised/stage_17c/evaluation_summary.json`; `confirmation_metrics.csv` |
| Stage 18 ranges and low-consumption recall | `results/supervised/stage_18/robustness_decision.json`; `primary_subgroup_metrics.csv` |
| Stage 19 non-level families `0.763537/0.325818` and redundancy conclusion | `results/supervised/stage_19/family_ablation_decision.json`; `family_only_metrics.csv` |
| Stage 20B inconclusive formulation controls | `results/supervised/stage_20b/stage_20b_decision.json`; `control_metrics.csv` |
| Stage 21 HGB/LightGBM/fixed-mean values | `results/supervised/stage_21/benchmark_metrics.csv`; `stage_21_decision.json` |
| Stage 22 compact-minus-full `−0.012108/−0.021840` | `results/supervised/stage_22/compact_minus_full_deltas.csv`; `stage_22_decision.json` |
| Study 1 Stage 23B refitted FULL_48 `0.7925125381/0.3701585998` | `results/supervised/stage_23b/final_metrics.json` |

## Study 2 development and diagnostic evidence

| Claim / value | Authoritative aggregate source |
|---|---|
| Stage 24B MEAN/MAX/ATTENTION metrics | `results/supervised/stage_24b/primary_metrics_summary.csv` |
| Stage 24B selected MAX `0.756667/0.314566`, FULL_48 deltas `−0.032164/−0.045229` | `results/supervised/stage_24b/stage24b_outcome.json` |
| Stage 25B LATENT `0.7908091587/0.3678373671` | `results/supervised/stage_25b/primary_metrics_summary.csv` |
| Stage 25B DIRECT `0.7882334050/0.3492440522` | Same artifact |
| Stage 25B FULL_48 `0.7888308219/0.3597945903` | `results/supervised/stage_17b/validation_decision.json`; referenced by `stage25b_outcome.json` |
| Stage 25B deltas and representation-parity outcome | `results/supervised/stage_25b/stage25b_outcome.json` |

## Study 2 confirmation and terminal evidence

| Claim / value | Authoritative aggregate source |
|---|---|
| CONFIRMATION LATENT/FULL_48/DIRECT metrics | `results/supervised/stage_26/confirmation_primary_metrics.csv` |
| CONFIRMATION effects and intervals | `results/supervised/stage_26/confirmation_bootstrap_comparisons.json` |
| Confirmation classification, seeds, anchor decision | `results/supervised/stage_26/stage26_outcome.json`; `confirmation_seed_metrics.csv` |
| Final LATENT/FULL_48/DIRECT exact metrics | `results/supervised/stage_27b/final_primary_metrics.csv` |
| Final paired observed effects, means, intervals, fractions | `results/supervised/stage_27b/final_bootstrap_comparisons.json` |
| Cross-role metrics, ordering, shifts and effect stability | `results/supervised/stage_27b/final_stage_stability.json` |
| Final individual seeds and summary | `results/supervised/stage_27b/final_seed_metrics.csv`; `final_outcome.json` |
| FINAL A criterion audit and no-collapse audit | `results/supervised/stage_27b/final_outcome.json` |
| Anchor survival and exact sentence | `results/supervised/stage_27b/final_anchor_audit.json` |
| Allowed/prohibited final claims and disclosure | `results/supervised/stage_27b/final_claim_audit.json` |
| Final integrity counts and hashes | `results/supervised/stage_27b/final_integrity.json` |
| Exact final access event | `results/supervised/stage_27b/final_access_event.json` |

## Literature context boundary

No tracked source was found for the Stage 28 prompt's STL-Net figures
(`≈0.717` ROC, `≈0.274` PR-AUC). They are not indexed as paper evidence and must
not enter a table, abstract, or superiority statement without a citable source
and protocol-equivalence assessment.

## Completeness rule

Every quantitative value currently proposed in the results freeze, abstract
blueprint, experiment ledger, and table/figure plan is mapped above to a
committed aggregate artifact. Future manuscript numbers must be added here
before use. Row-level prediction files are not manuscript evidence sources.
