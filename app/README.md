# AMP interface

**Current status: Interface Stage 3 PASS — frozen Model Performance dashboard.**
Open **Model Performance** for verified customer metrics, the protocol-compatible
AMP-LDP / FULL_48 / DIRECT comparison, absolute differences, AMP-LDP ROC and
precision–recall curves, methodology, limitations and source hashes. See
[STAGE3_VALIDATION.md](STAGE3_VALIDATION.md) and the preimplementation
[evidence audit](STAGE3_DESIGN.md). Stage 4 is not started.

**Previous delivery: Interface Stage 2 PASS — customer consumption inspector.**
All 6,356 scored customers have verified original histories. Open the Customer
Risk Explorer, search or select an identifier, and choose **Open customer
inspector**. Back preserves search, sort and selected customer; each customer's
date interval is retained during navigation.
See [STAGE2_VALIDATION.md](STAGE2_VALIDATION.md) for its evidence.

**Previous delivery: Interface Stage 1B PASS.** Authorized
offline inference-only replay produced a separate derived export of all 6,356
AMP-LDP customer scores. Both frozen ensemble metrics reproduced exactly. The
Explorer is active and clearly labels the reconstruction. No model was fitted,
no threshold changed, and no frozen research artifact was rewritten. The original Stage 1 partial record remains
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
Streamlit 1.66.0, pandas 3.0.5, numpy 2.5.2, pyarrow 25.0.1 and Altair 6.3.0.
These packages were already installed; Stages 2 and 3 needed no installation.
No torch, CUDA, scikit-learn,
LightGBM, pytest or model checkpoint deserialization is required. pandas, numpy,
Pillow and packaging retain their pre-existing versions after installation.
The separate environment command is documented, not claimed as separately tested.

## Structure and behavior

- `main.py`: application identity and native sidebar navigation.
- `config.py`, `artifact_catalog.json`: repository root, paths and reviewed
  provenance bindings. The catalog contains hashes, not copied metric values.
- `data_loader.py`: reusable, read-only evidence and prediction adapter.
- `performance_loader.py`, `performance_catalog.json`: hash/schema/protocol
  checks for the three frozen customer methods, optional curve provenance,
  derived prevalence and absolute metric differences.
- `views/performance.py`: full-range comparison/curve charts and interpretation.
- `derived/stage3/`: immutable ROC/PR coordinates, calculation provenance and
  protected research baseline. No new model predictions or input data copies.
- `consumption_loader.py`, `consumption_catalog.json`: original-history access,
  verified source/index/mask/window lineage and descriptive statistics.
- `views/inspector.py`: customer details, interactive source chart and coverage.
- `derived/stage2/`: compact raw-record index, provenance and protected snapshot;
  no duplicate consumption values.
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

The interface suite has 75 tests; the separately run offline suite has 8 tests
(83 total, retaining all 59 previous regressions). New checks cover comparator
compatibility, exact metrics and model ordering, correct axes/labels/deltas,
curve provenance/alignment and AP step semantics, optional curve failures,
malformed evidence, unchanged research and navigation. Offline checks
independently reproduce all coordinates from existing scores; no inference runs.
See [RECOVERY_STAGE1B.md](RECOVERY_STAGE1B.md) for current validation and
[VALIDATION.md](VALIDATION.md) for the initial Stage 1 record.
The master decision record is `../docs/preprocessing_decisions.md`, appended as
the **Interface development workstream**, independently of ML stage numbering.

## Customer consumption inspection

The original, immutable `data/raw/data set.csv` is required for history charts.
The adapter joins by original CONS_NO to the verified Stage 1B scores and checks
FLAG against the held-out evaluation labels. Its compact index records each
selected CSV record's byte offset, length and SHA-256. Every request rechecks
the selected record's hash, own ID and label before displaying values. There
is no assumption that a data row number equals a score row number.

The observed period is 1 January 2014–31 October 2016. The source has 1,034 date
columns; the complete calendar has 1,035 days. Missing 18 September 2016 has no
source column and is explicitly marked missing. The source chart includes
3 October 2014 even though frozen model preprocessing excluded that day.
Original zeros remain observed. Counts/statistics cover the full period and use
observed numeric source values only; there is no interpolation or silent zero
fill. Units were not established, so the UI labels source units without assuming
kWh. Processed/imputed values are not plotted as original readings.

Altair supplies line/point tooltips and horizontal pan/wheel zoom. Lines use
separate observed segments, so they stop across missing days. The coverage
strip distinguishes observed, missing and absent-column days. Choose a date
interval using the range control; **Reset viewed interval** also recreates the
chart to clear browser zoom. Charts inherit the native Streamlit theme.

The inspector shows original identifier, six-decimal risk score plus the exact
stored float value, global rank, evaluation-only FLAG, frozen score provenance,
observed/missing/zero counts, missing percentage and observed-value statistics.
Previous/Next rank supports sequential inspection. Actual evaluated-window
metadata and a concise explanation distinguish raw observations from frozen
imputed/log1p/scaled model inputs. No theft periods or causal explanation are
inferred.

Runtime verifies the raw CSV's full hash once per filesystem identity; it streams
the hash and does not materialize a full consumption table. Missing files and
identity changes are checked before using cached validation. Later requests
seek/read one original record. Small immutable index parses and up to 16
customer window lookups are cached. Parquet hashes use exact binary bytes.
Measured on this machine: approximately 0.54 seconds cold, 0.22 seconds for
another customer and 0.20 seconds repeated. These are observations, not a
performance guarantee.

The shipped index is 812,696 bytes. All three Stage 2 metadata files total
903,135 bytes (about 882 KiB). Git preserves their exact bytes. Dataset inputs
remain local and untracked; a missing source disables the history view while
keeping the frozen score and Back navigation available. The app needs no GPU
or network connectivity once dependencies and local artifacts are present.

The separate metadata builder is:

```powershell
.\.venv\Scripts\python.exe -m app.recovery.build_history_index
```

It was executed once. It requires the shipped protected snapshot and original
inputs and refuses to overwrite the shipped index/provenance. It scans the raw
CSV once, verifies all 6,356 selected histories against frozen test masks and
evaluated-window metadata, and writes only interface metadata/catalog. It never
fits, scores, changes preprocessing or writes research artifacts. Ordinary
dashboard use does not invoke it. See `STAGE2_DESIGN.md` for the preimplementation
design and `derived/stage2/history_provenance.json` for exact source hashes.

## Model Performance evidence

All displayed comparisons come from `results/supervised/stage_27b/final_primary_metrics.csv`.
Stage 27A freezes the original Stage 17B FULL_48 and Stage 25B DIRECT HGBs,
DEV_TRAIN-only training for all methods, and the same customer evaluation
protocol. The table keeps full floats and displays six decimals. Both bar-chart
axes span 0–1; differences are absolute metric units. No significance claim or
relative percentage gain is made. FC/CNN/LSTM reconstruction metrics, historical
Stage 13 / DEV_VAL exports and the different Stage 23B FULL_48 refit are excluded.

The SGCC cohort has 6,356 customers, 542 theft-labelled (8.5274%). It retains the
explicit disclosure of historical Stage 13 baseline access. It is not evidence
of external utility generalization. No applicable frozen AMP-LDP decision
threshold/confusion matrix exists; the page uses ranking metrics only.

No original Stage 27B curve coordinates or matching comparator predictions were
found. Only AMP-LDP curves are supported, using the Stage 1B reconstructed scores
and exact aligned labels. The separate create-only visualization builder is:

```powershell
.\.venv\Scripts\python.exe -m app.recovery.build_performance_curves
```

It was run during Stage 3 and refuses to overwrite the shipped files. It requires
the existing research environment's sklearn only for offline coordinate
calculation and integrity checks. It loads no model, fits nothing, regenerates
no score, chooses no operating threshold and writes no research artifact. The
dashboard never invokes or imports it. All 835 ROC and 6,050 PR points are kept.
Independent sklearn ROC-AUC and `average_precision_score` match the frozen values
exactly. PR uses step-before interpolation on increasing recall; AP is the
recall-increment weighted precision sum, not trapezoidal PR area. FULL_48 and
DIRECT retain aggregate-only displays with an explicit absence explanation.

The curve manifest binds source CSV/provenance, membership and labels, metric
source, protocol, output hash, builder source hash, environment and parameters.
The catalog binds the manifest. Hash checks run on every load with cached
immutable parsing. Missing/corrupt curves leave aggregate comparison available;
unverified aggregate/comparator metadata disables the page. All 543 protected
research/source/data and earlier interface evidence files remain byte-identical.

## Boundaries and next dependency

No live inference, retraining, preprocessing/split changes, thresholds or
deployment infrastructure are implemented. Stage 4 requires explicit scope
and approval. A suitable next scope is demo usability, guided research narrative
and provenance-aware presentation/export, retaining the frozen ranking-only
boundary. New comparator inference, causal explanations, threshold selection
or deployment would need separate authorization and evidence. Stage 1B's narrow
authorization resolved the missing-export dependency through frozen replay.
The original export remains absent, and reconstructed scores must retain that
description. No original per-row score checksum exists for direct byte comparison;
frozen inputs, fitted state, ordered cohort hashes and exact aggregate metric
agreement establish the reconstruction's evidence chain.
