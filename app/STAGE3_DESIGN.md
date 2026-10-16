# Interface Stage 3 — evidence audit before implementation

17 October 2026. Base `efe2ca7eb8d9ef4447d77d5f686d531a29e37e72`, branch
`study2/mil`. Only the three existing untracked ZIPs and checkpoint directory
were present. No applicable AGENTS.md. Stage 3 authorizes presentation and
read-only curve calculations, no ML experimentation or Stage 4.

## Authoritative evidence

Evidence commit `dbd978ffeebe7cee772f8ec693243895be7ff2c8`; Stage 27A freeze
`20cd26627b0601965166bab879ab3c46695ac0f0`. SHA-256 below uses CRLF-to-LF
normalization for research text; the protected snapshot uses exact file bytes.

| Source | SHA-256 |
|---|---|
| `results/supervised/stage_27a/final_model_freeze.json` | `a5ce3e6db799e3f4053a69d457a7d22a9f45db27bda6e6f08ec87c69b232c14e` |
| `results/supervised/stage_27a/final_comparator_freeze.json` | `e97724ee4e62519d70810c0054297755e5c48c873e7190eafca2e87622e34f19` |
| `results/supervised/stage_27a/stage27b_preregistration.json` | `ed2457e3a6eaf4b683c3fd8aaf89cf1f1fca63500a68f0427279813533e92702` |
| `results/supervised/stage_27b/final_primary_metrics.csv` | `2ee0b10750579c6e124b300fac21a1eed87d8629195d5efd7773b2cccd38111f` |
| `results/supervised/stage_27b/final_integrity.json` | `688f39a004341780abec390bd31b6c18b90284b8a3c104828afd299bd0af9551` |
| `results/supervised/stage_27b/final_access_event.json` | `d3715030ef9fa52e89310a4fa5daa85c62fbc2eff7e64521ae1e42f885346116` |
| `app/derived/stage1b/customer_scores.csv` | `17a7af4ac32a01981549dd6fe4e99809948aa92c82aa611b05214433b0d182ff` |
| `app/derived/stage1b/provenance.json` | `e38b28f37231acabf2b42bd6912f65ac2eb6b99d5db4aecc372ab75381b582b3` |

The original evaluator uses sklearn `roc_auc_score` and
`average_precision_score` on the same customer labels and scores for all three
methods. Stage 27A freezes symmetric DEV_TRAIN-only training; all Stage 27B rows
represent 6,356 FINAL_TEST customers, 542 positives, no exclusions, matched
ordered membership/label hashes. Display six decimals; keep full floats.

## Inclusion and exclusions

- Include LATENT (display AMP-LDP), FULL_48 and DIRECT from the one Stage 27B
  metrics file. FULL_48 is the original Stage 17B 48-feature HGB with chronology
  and persistence; DIRECT aggregates six statistics over 17 standardized window
  summaries (102 features). AMP-LDP aggregates six statistics over 16 learned
  dimensions (96 features) and averages five HGB scores, seeds 42–46.
- Exclude Stage 23B FULL_48 exports: they use a DEV_TRAIN+DEV_VAL refit, expressly
  rejected by the Stage 27A comparator freeze. Stage 13 exports are an obsolete
  supervised window baseline; Stage 24B exports are DEV_VAL.
- Stage 12 architecture report and Stage 10 CNN test report establish that
  FC/CNN/LSTM share historical test customers/windows, but train on normal
  windows, score reconstruction MSE, primarily report window-level metrics and
  validation-normal P95 operating rules. Customer mean/max/p95 MSE are secondary
  diagnostics, not the Stage 27B preregistered customer HGB comparison. Sharing
  an underlying split does not establish the same scoring unit and protocol.

## Curves, thresholds and boundaries

Repository results inventory contains historical reconstruction training/error,
validation PR and confusion-matrix plots and aggregate paper tables, but no
original Stage 27B ROC/PR coordinates or per-customer comparator exports.
`evaluate_stage27b_terminal.py` computes comparator scores in memory and persists
aggregate/seed/uncertainty outputs. No original final comparator curves will be
shown or approximated; no comparator inference is authorized.

Derive AMP-LDP coordinates offline from the already verified Stage 1B score
export and aligned FLAG only. Preserve reconstructed origin. Independently
check sklearn ROC-AUC and AP against full frozen precision, retain source,
ordered membership/labels and output hashes in a separate Stage 3 manifest.
The dashboard reads immutable coordinates; it never imports sklearn or models.
PR display uses an ordered step curve; AP is the original non-interpolated
recall-increment weighted precision sum, not trapezoidal PR area.

Both comparison axes and curve axes span [0,1]. Absolute differences use metric
units, with no relative-percentage or significance claim. There is no applicable
frozen AMP-LDP binary operating threshold; show no confusion matrix, 0.5 rule
or test-selected threshold. Prevalence is verified positives/customers context.

Retain exact SGCC cohort disclosure, ranking/probability/proof/localization/
causality/generalization limits. Missing curves leave verified aggregate
comparison usable; missing or incompatible aggregate evidence fails closed.
Existing Explorer/Inspector state and histories remain intact. Cache immutable
parses while checking current artifact hashes. No additional dependency.

Before implementation, `app/derived/stage3/research_snapshot.json` records
protected research/source/data and all Stage 1B/2 derived evidence plus existing
catalog bytes. Recheck after generation, tests and before commit. Documentation,
master ledger and stage-specific validation will record actual outcomes.
