# AMP final project freeze checkpoint

Date: **17 October 2026 (Asia/Calcutta)**. **Local checkpoint complete;
GitHub publication blocked pending documented data permission.** This document
does not certify a completed remote release or server-enforced immutability.

AMP's research methodology, fitted models, evaluation results, interface
implementation and demo configuration are frozen. Future work requires an
explicitly authorized new development stage on a separate branch.

## Frozen scope and architecture

ML research is frozen. Interface Stages 1, 1B, 2, 3 and 4 passed their technical
gates; the final user instruction accepts those reported stage outcomes for this
checkpoint. Historical reports retain their original stage-specific status.

AMP-LDP uses non-overlapping 30-day windows, 17 summaries, a frozen 17→32→16
encoder, six statistics per latent dimension (96 customer features), and five
frozen HGB classifiers (seeds 42–46), combined by unweighted arithmetic mean.
Primary models use DEV_TRAIN only. The compatible final comparators are the
original FULL_48 HGB and DIRECT distribution-summary HGB.

The local Streamlit dashboard reads verified frozen artifacts. Overview,
Customer Risk Explorer/Inspector, Model Performance and Demo Guide provide
rankings, original histories, compatible comparisons, supported AMP-LDP curves,
three provenance-bound illustrative presets and an aggregate-only JSON export.
Runtime performs no fitting, prediction generation or threshold selection.

## Authoritative evidence and reproducibility

- Method/protocol freeze: Stage 27A, commit
  `20cd26627b0601965166bab879ab3c46695ac0f0`;
  `results/supervised/stage_27a/final_model_freeze.json`,
  `final_comparator_freeze.json` and `stage27b_preregistration.json`.
- Original terminal evidence: commit
  `dbd978ffeebe7cee772f8ec693243895be7ff2c8`,
  `results/supervised/stage_27b/`, including primary metrics, integrity and
  original access event. The cohort is the **final evaluation cohort with
  disclosed historical baseline access**: 6,356 customers, 542 theft-labelled,
  165,020 usable windows, no new exclusions.
- Frozen AMP-LDP ROC-AUC **0.8117294176037735**, average precision
  **0.42186322254778824**. Definitions remain customer-level sklearn ROC-AUC
  and non-interpolated recall-increment weighted AP, positive label FLAG=1.
- Stage 1B is a separately authorized, inference-only reconstruction, not an
  original persisted Stage 27B prediction export. Its CSV SHA-256 is
  `17a7af4ac32a01981549dd6fe4e99809948aa92c82aa611b05214433b0d182ff`;
  provenance SHA-256 is
  `e38b28f37231acabf2b42bd6912f65ac2eb6b99d5db4aecc372ab75381b582b3`.
  Both metrics reproduce exactly. Do not rerun replay or builders for this freeze.
- Stage 2 history lineage and Stage 3 curve lineage remain bound by their
  catalogs/provenance. Stage 4's exact-byte baseline covers 549 protected files
  in `app/derived/stage4/research_snapshot.json`.
- See [method](paper/AMP_METHOD_FREEZE.md),
  [results](paper/AMP_RESULTS_FREEZE.md),
  [evidence index](paper/AMP_PAPER_EVIDENCE_INDEX.md),
  [master ledger](preprocessing_decisions.md),
  [Stage 1B recovery](../app/RECOVERY_STAGE1B.md), and
  [final interface validation](../app/STAGE4_VALIDATION.md).

Final rerun on 17 October: **98 interface + 8 recovery tests passed (106)**;
**12 preflight checks passed, zero warnings**. Syntax compilation and dependency
checks passed. All **549 protected files were byte-identical before and after
validation** against the Stage 4 baseline. Research
paths `results`, `models`, `src`, `configs`, `data` have no committed differences
from the authoritative Stage 27B evidence commit. No models were fitted, scores
regenerated, thresholds changed or original access records rewritten. The prior
Stage 4 browser/offline rehearsal remains the recorded rehearsal; it was not
repeated for this documentation-only checkpoint.

## Local dependencies and startup

Use the existing validated Python 3.12 environment and pinned
`requirements-interface.txt`. Exact approved local copies are required for
`data/raw/data set.csv`, `data/processed/test_mask.parquet`, and
`data/windows/stride_30/usable/test_windows.parquet`; their hashes and restoration
guidance are in `app/consumption_catalog.json`, Stage 2 provenance and the
[presenter runbook](../app/DEMO_GUIDE.md). These large inputs are ignored locally
and are not bundled by this release checkpoint. Research replay additionally
depends on its original inputs/checkpoints; replay is not required to launch.

```powershell
Set-Location C:\Projects\ML_Project
.\.venv\Scripts\python.exe -m app.preflight
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`; prewarm pages before presenting, stop with Ctrl+C.
For dependency setup and exact recovery, follow `app/README.md` and the runbook.
A clean isolated reinstall and complete offline wheelhouse are not validated.

## Release identity and publication gate

- Intended remote: **origin**, private repository
  [RaghavSingh-CS/ML_Project](https://github.com/RaghavSingh-CS/ML_Project).
- Branch: [study2/mil](https://github.com/RaghavSingh-CS/ML_Project/tree/study2/mil).
  Audited implementation HEAD: `b2ef13307922ca649ab84cd5fde3038cac980bee`.
  Local checkpoint identity is the documentation commit adding this file, with
  subject `docs(amp): finalize project freeze and release checkpoint`; obtain its
  full SHA with `git log -1 --format=%H -- docs/FINAL_PROJECT_FREEZE.md`.
- Fetched/verified remote branch SHA:
  `55035842fa0306a4fafa911472b45db16517f70f`. It is an ancestor of implementation
  HEAD, with 13 outgoing commits before the checkpoint documentation commit.
- Planned annotated tag: **amp-demo-freeze-2026-10-17**, targeting the final
  documentation commit only after a verified safe branch push. The tag was absent
  locally/remotely during audit. **No branch push, tag creation or tag push was
  performed** because the data-publication gate is unresolved; no tag URL exists.
- Full outgoing-history audit: 117 changed paths, 148 blob versions, 3,508,090
  uncompressed blob bytes before this documentation commit. Largest blob:
  `app/derived/stage2/history_index.csv`, 812,696 bytes. The 56,780-byte methodology
  ZIP contains report/template sources, not customer records. No outgoing raw
  consumption dataset, environment/credential file, blob over 50 MiB or LFS
  pointer was found. Credential-pattern scans of all outgoing blobs and ZIP
  members found no matches; this is not an exhaustive secret-free certification.
- **Blocking records:** `app/derived/stage1b/customer_scores.csv` links all 6,356
  original/pseudonymous CONS_NO identifiers to FLAG and risk score;
  `app/derived/stage2/history_index.csv` links the same identifiers/labels to
  record hashes, offsets and consumption-coverage/window metadata. Three real
  identifiers also appear in `app/demo_catalog.json`, `app/DEMO_GUIDE.md`,
  `app/STAGE4_VALIDATION.md`, `tests/interface/test_demo.py` and the master ledger.
  Existing governance does not establish permission to upload/redistribute these
  records. Private visibility and repository administration rights do not supply
  that missing data permission. Dataset licence/data-owner approval covering this
  private repository upload must be established before publication can resume.
  No committed artifacts were removed and no history was rewritten.
- Target branch reports `protected: false`. GitHub ruleset and branch-protection
  APIs returned HTTP 403 requiring GitHub Pro or public visibility. No protection
  changes were made; branch/tag enforcement remains pending and unavailable under
  the current private-repository plan. Do not change visibility to bypass this.
- Three unrelated untracked ZIPs and `report/checkpoints/september_20_v2/` remain
  local and excluded. No GitHub Release or repository archive was created.

## Limits and reopening

Evidence is internal to SGCC with disclosed historical baseline access, not an
independent external/operational validation. Scores are review priorities, not
established calibrated probabilities or proof; no applicable frozen binary
threshold/confusion matrix, verified theft day or causal attribution is supplied.
Illustrative customers are chosen using labels/ranks and are not representative.
Original per-customer Stage 27B comparator scores/curves are unavailable; only
supported AMP-LDP curves are shown. Raw units are not confirmed. Export controls
are presentation safeguards, not access control or anonymization.

Publication may resume only when governing data permission is documented and
the release audit, validation and remote ancestry are rechecked. That is release
administration, not authorization for new ML/interface work. Any methodological,
model, evaluation, interface or demo change requires an explicitly authorized new
stage on a separate branch. The requested final remote-freeze declaration remains
withheld until safe publication and branch/tag verification succeed.
