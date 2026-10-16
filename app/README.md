# AMP interface — Stage 1

**Status: Interface Stage 1B PASS; Stage 1 technical gates closed.** Authorized
offline inference-only replay produced a separate derived export of all 6,356
AMP-LDP customer scores. Both frozen ensemble metrics reproduced exactly. The
Explorer is active and clearly labels the reconstruction. No model was fitted,
no threshold changed, and no frozen research artifact was rewritten. Stage 2
has not been started or authorized. The original Stage 1 partial record remains
in [VALIDATION.md](VALIDATION.md); the resolution is in
[RECOVERY_STAGE1B.md](RECOVERY_STAGE1B.md).

## Start locally in PowerShell

Python 3.12 is verified. From this repository:

```powershell
Set-Location C:\Projects\ML_Project
.\.venv\Scripts\python.exe -m pip install -r requirements-interface.txt
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Visit `http://127.0.0.1:8501` and stop with Ctrl+C. If the environment is activated,
the short command is `python -m streamlit run app/main.py`. The explicit command
above restricts listening to loopback and disables usage telemetry.

For a separate CPU-only environment without research dependencies:

```powershell
python -m venv .venv-interface
.\.venv-interface\Scripts\python.exe -m pip install -r requirements-interface.txt
.\.venv-interface\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The existing `.venv` was used for validation. Direct interface dependencies are
Streamlit 1.66.0, pandas 3.0.5 and numpy 2.5.2. No torch, CUDA, scikit-learn,
LightGBM, pytest or model checkpoint deserialization is required. pandas, numpy,
Pillow and packaging retain their pre-existing versions after installation.
The separate environment command is documented, not claimed as separately tested.

## Structure and behavior

- `main.py`: application identity and native sidebar navigation.
- `config.py`, `artifact_catalog.json`: repository root, paths and reviewed
  provenance bindings. The catalog contains hashes, not copied metric values.
- `data_loader.py`: reusable, read-only evidence and prediction adapter.
- `derived/stage1b/`: verified reconstructed CSV, provenance and research snapshot.
- `recovery/`: separately invoked offline replay and independent metric checks;
  these modules are never imported by the dashboard.
- `views/overview.py`: verified metrics, pipeline, freeze provenance and context.
- `views/explorer.py`: score table/search/sorting when verified scores exist;
  informative unavailable state otherwise. Native scrolling limits table height.
- `../tests/interface/`: unittest adapter and Streamlit AppTest coverage.
- `AUDIT.md`: repository findings and structure recorded before implementation.

Native Streamlit components inherit system/light/dark themes and stack columns
on narrow viewports. Choose a theme from Streamlit's menu. There is no custom
frontend framework or fixed-color stylesheet.

## Source of truth

All catalog paths are repository-relative. Aggregate evidence comes from commit
`dbd978ffeebe7cee772f8ec693243895be7ff2c8` (Stage 27B), with Stage 27A metadata
and Stage 17A governance preserved in that commit:

| Purpose | Frozen source |
|---|---|
| ROC-AUC and AP | `results/supervised/stage_27b/final_primary_metrics.csv`, LATENT row |
| Cohort size, theft-labelled count, ordered ID/label hashes | `results/supervised/stage_27b/final_integrity.json` |
| Model definition and ten checkpoint hashes | `results/supervised/stage_27a/final_model_freeze.json` |
| Final role, cohort expectations, ensemble rule | `results/supervised/stage_27a/stage27b_preregistration.json` |
| Evaluation event and historical-access disclosure | `results/supervised/stage_27b/final_access_event.json` |
| Original identifiers and split membership | `results/governance/stage_17a/research_protocol_manifest.csv` |
| Reconstructed AMP-LDP scores / evaluation labels | `app/derived/stage1b/customer_scores.csv`, bound to `provenance.json`; original Stage 27B export absent |

The LATENT values are ROC-AUC `0.8117294176037735`, AP `0.42186322254778824`,
6,356 evaluation customers and 542 theft-labelled customers. The UI uses the
frozen paper reporting policy of four decimals. These numbers are read from
validated evidence at runtime. They are recorded here as verification results.

The final cohort had disclosed historical Stage 13 baseline access. The UI uses
the required name **final evaluation cohort with disclosed historical baseline
access**. Risk is a ranking score, not an established calibrated probability or
proof of theft. Labels describe customers, not theft dates.

## Integrity and caching

Each evidence file must match the reviewed SHA-256 in `artifact_catalog.json`.
Text hashes normalize only CRLF to LF to support Windows Git checkouts. Original
ID strings are parsed without NA coercion or numeric conversion. Paths cannot
escape the repository; small-file size limits prevent accidental raw-data loads.
The adapter checks schema, finite metric ranges, final cohort counts, model
identity, terminal event, role and split agreement. Missing/modified evidence
produces an unavailable state, with no historical-value fallback.

Parsing of verified immutable bytes is cached with bounded `lru_cache`. Every
request rechecks current file existence and content hash, so cached data cannot
hide changed or deleted evidence. The app opens no research artifact for writing.

## Derived score artifact and verification

`predictions` binds the separate reconstructed Stage 1B artifact and its
hash-verified provenance manifest. Do not point it at a Stage 13, Stage 23B,
DEV_VAL or confirmation export. Stage 23B's customer file
belongs to the refitted Study 1 FULL_48 model, not AMP-LDP. The terminal evaluator
`src/evaluation/evaluate_stage27b_terminal.py` retains its scores only in memory
and persists aggregate/seed metrics. Its main/access writers were not rerun.

The approved derived export supplies UTF-8 CSV columns exactly
`CONS_NO,FLAG,risk_score`, with original identifiers, customer-level binary labels
and the frozen arithmetic five-seed ensemble score. Its rows must retain the
original Stage 27B customer order so the recorded membership and label hashes
are checked before any sorting. Governance contains exactly the same
FINAL_TEST IDs, all with original split `test`; duplicate IDs or protected-role
substitutions fail validation. Expected counts come from frozen evidence.
Scores must be numeric, finite and within [0,1]. No calibration is inferred.

The reviewed catalog record includes `path`, LF-normalized `sha256`,
`model_id=LATENT_DISTRIBUTION_HGB_FIVE_SEED_MEAN`, `source_stage=27B`,
`research_role=FINAL_TEST`, `seeds=[42,43,44,45,46]`,
`ensemble=arithmetic_mean`, and `model_freeze_sha256` matching the frozen model
record. The additional `origin=reconstructed_inference_only` and `provenance`
path/hash bind the reconstruction manifest. It must state VERIFIED status,
correct frozen model/protocol/cohort identity, zero fitting, unchanged research,
and successful independent six-decimal metric verification. These bindings were
reviewed against the executed replay; schema/range alone cannot prove provenance.
No UI upload or arbitrary path override is provided.

The output CSV is 354,095 bytes, SHA-256
`17a7af4ac32a01981549dd6fe4e99809948aa92c82aa611b05214433b0d182ff`.
Git attributes preserve all three derived evidence files byte-for-byte. Their
combined size is approximately 430 KiB; no new checkpoint or consumption dataset
is committed.

## Reproduce the authorized offline replay

This is a separate research-environment command, not a dashboard requirement.
It requires the existing fitted checkpoints, historical processed test inputs
and the package versions recorded in the provenance manifest. The original
environment is available in `.venv`; the minimal interface-only environment
does not install torch or scikit-learn.

```powershell
.\.venv\Scripts\python.exe -m app.recovery.replay_stage1b --output-dir app/derived/stage1b-reproduced
```

The command rejects existing output directories and paths outside `app/derived/`.
It checks every prerequisite before inference, uses only original saved state
and read-only helpers, rejects fitting/backpropagation, and verifies every seed
and the ensemble. It never modifies the original Stage 27B access record, calls
historical evaluation main functions, selects thresholds or loads train/validation
consumption. The initial execution used the default `app/derived/stage1b` output.
The documented `-reproduced` command writes a fresh sibling; it was not separately
executed because one verified replay was sufficient for this delivery.

After validation, global rank is deterministic ordinal descending score, then
original identifier for ties. Literal, case-insensitive identifier search and
ascending score/identifier sorting preserve that global rank. Evaluation labels
are explicitly annotated. Fixture data exists only within tests and is never
used by the runtime application.

## Verify

```powershell
.\.venv\Scripts\python.exe -m compileall -q app tests/interface
.\.venv\Scripts\python.exe -m unittest discover -s tests/interface -v
.\.venv\Scripts\python.exe -m unittest discover -s tests/recovery -v
.\.venv\Scripts\python.exe -m pip check
```

The interface suite has 35 tests; the separately run offline suite has 6 tests.
See [RECOVERY_STAGE1B.md](RECOVERY_STAGE1B.md) for current validation and
[VALIDATION.md](VALIDATION.md) for the initial Stage 1 record.
The master decision record is `../docs/preprocessing_decisions.md`, appended as
**Interface Stage 1**, independently of the existing ML stage numbering.

## Boundaries and next dependency

No live inference, retraining, preprocessing/split changes, threshold, history
charts, expanded comparison pages or deployment infrastructure is implemented.
Stage 2 requires supervisor acceptance and an explicit scope. Stage 1B's narrow
authorization resolved the missing-export dependency through frozen replay.
The original export remains absent, and reconstructed scores must retain that
description. No original per-row score checksum exists for direct byte comparison;
frozen inputs, fitted state, ordered cohort hashes and exact aggregate metric
agreement establish the reconstruction's evidence chain.
