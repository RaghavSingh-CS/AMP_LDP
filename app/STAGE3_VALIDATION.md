# Interface Stage 3 — Model Performance validation

17 October 2026. **PASS — technical delivery; supervising acceptance remains
separate.** Base HEAD `efe2ca7eb8d9ef4447d77d5f686d531a29e37e72`, branch
`study2/mil`. No Stage 4 implementation. No push authorized or performed.

## Evidence and scientific scope

The preimplementation inventory, frozen identities and hashes are recorded in
[STAGE3_DESIGN.md](STAGE3_DESIGN.md); runtime bindings are in
`performance_catalog.json` and the unchanged `artifact_catalog.json`.
Sources are the Stage 27A model/comparator freeze and preregistration, Stage 27B
primary metrics, integrity and access event, original evaluator metric
definitions, frozen method document and Stage 12 architecture comparison.

| Display model | ROC-AUC (full precision) | Average Precision (full precision) | Display ROC / AP |
|---|---:|---:|---|
| AMP-LDP | 0.8117294176037735 | 0.42186322254778824 | 0.811729 / 0.421863 |
| FULL_48 | 0.7978625838889967 | 0.3853256447732082 | 0.797863 / 0.385326 |
| DIRECT | 0.7949673266082506 | 0.3684394274673042 | 0.794967 / 0.368439 |

Verified counts are 6,356 customers, 542 theft-labelled and 5,814 normal;
prevalence derives from 542/6356 = 0.08527375707992448 and displays 8.5274%.
All methods use the same customer-level Stage 27B cohort/metric protocol and
DEV_TRAIN-only training. Compatibility checks bind preregistered model IDs,
checkpoint hashes, training population, feature dimensions, metric units and
no-threshold policy. No estimator is loaded at dashboard runtime.

Absolute AMP-LDP minus FULL_48 differences display +0.013867 ROC / +0.036538 AP;
minus DIRECT displays +0.016762 / +0.053424. Computation retains full precision.
Comparison bars and both curve axes span [0,1]. Ordering is AMP-LDP, FULL_48,
DIRECT for both metrics. No truncated axis, relative-gain or significance claim.

FC/CNN/LSTM share historical splits but use normal-window reconstruction MSE,
primarily window-level metrics and validation-normal P95 decisions; customer
mean/max/p95 diagnostic results do not establish Stage 27B protocol equivalence.
Stage 13 window baseline, Stage 24B DEV_VAL exports, individual seed results
and Stage 23B DEV_TRAIN+DEV_VAL FULL_48 refit are also excluded from the main chart.
FULL_48 retains its chronology/persistence information difference; DIRECT is
six statistics over 17 summaries (102 features). The page explains the frozen
17→32→16 latent representation, six statistics (96 features) and five-HGB mean.

## Curves and provenance

Original Stage 27B evaluator persists aggregate/seed/uncertainty records, not
per-customer scores or ROC/PR coordinates. Existing reconstruction figures use
different evaluation protocols. No matching frozen FULL_48/DIRECT curve or
prediction export was found, so they have aggregate-only displays. No comparator
inference, approximation or bootstrap rerun was performed. This explicitly
permitted absence does not block Stage 3's supported-curve acceptance gate.

AMP-LDP curves were calculated offline solely from verified Stage 1B reconstructed
scores and aligned FLAG. Both independently recalculated sklearn ranking metrics
equal the original frozen values at full precision (difference 0.0). Every
coordinate was independently regenerated in tests from the existing CSV, with
no model access. There are 835 ROC points and 6,050 PR points (all retained).
The PR display increases recall with step-before interpolation. AP follows
`average_precision_score`, the non-interpolated recall-increment weighted
precision sum; it is never reported as trapezoidal PR area.

| Separate Stage 3 artifact | Bytes | Exact SHA-256 |
|---|---:|---|
| `derived/stage3/curves.json` | 277172 | `de051fa9ff50a93d35923650f5e4a537b849e002f86212bb624519df2c1178a7` |
| `derived/stage3/curve_provenance.json` | 2183 | `7cdf785590c09a070a4c2801c889b89dcd759705eab218bb4c0dfe02517ecf78` |
| `derived/stage3/research_snapshot.json` | 70955 | `b39fa1e21b2827fd36ab507fc60be91b7863b1f23922beb45fa485f92b0e3bc1` |

The 350,310-byte addition contains coordinates/provenance and a protected-file
baseline, no new risk scores or copied consumption values. Provenance binds
original score/provenance hashes, ordered customer and label hashes, model ID,
protocol/metric source, exact builder source, Python/sklearn versions, parameters,
point counts and no-inference/no-fit/no-operating-threshold declarations. Git
attributes preserve derived JSON and the exact-hashed builder source bytes.
The create-only builder is never run at app startup and refuses overwrite.

## Automated validation

Commands executed using the existing CPU environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/interface -v
.\.venv\Scripts\python.exe -m unittest discover -s tests/recovery -v
.\.venv\Scripts\python.exe -m compileall -q app tests/interface tests/recovery
.\.venv\Scripts\python.exe -m pip check
```

**75 interface tests + 8 offline tests = 83 passing**, retaining every previous
59 tests. New tests cover exact metrics/prevalence/order/deltas, chart data,
metric labels and [0,1] axes, protocol/checkpoint/training incompatibility,
curve source/label alignment and endpoints/geometry, all coordinates,
non-interpolated AP versus trapezoidal area, explicit comparator-curve absence,
missing/malformed/corrupt evidence after cache, unavailable page/curve states,
create-only output protection, navigation and protected research bytes.
Existing Inspector histories, search/sort/selection, exact frozen scores,
Back/Previous/Next and per-customer interval persistence tests pass unchanged.
Compilation/import checks pass; pip reports no broken requirements.

Initial checks caught Windows CRLF serialization conflicting with the new
manifest's exact hash. The serializer now writes LF explicitly; only the new
Stage 3 files were rebuilt. New AppTest assertions were corrected to use the
current `vega_lite_chart` element and include captions when checking metric
explanations. No existing test was weakened or removed. Final suites pass.
Staged whitespace review also normalized only the new Stage 3 snapshot's line
endings to LF; its 543 path/hash entries are identical. Recorded size/hash above
reflect that serialization, and all protected bytes were rechecked afterward.

## Application and browser validation

Streamlit 1.66.0 starts at `http://127.0.0.1:8501`; `/_stcore/health` and `/`
both return HTTP 200. The existing interface dependencies suffice; no package
installation, GPU, model deserialization or online service is required.
Offline coordinate verification uses the existing sklearn 1.9.0 environment.

Actual in-app browser checks verify Model Performance navigation, exact displayed
metrics, three-model table, separate ROC/AP bars, absolute difference table,
AMP-LDP ROC/PR curves, missing comparator-curve disclosure and method/limit
text. Light and dark themes render without chart errors. Desktop 1280×720 and
mobile 390×844 checked: charts stack, axes remain readable and prose wraps.
The temporary viewport override was reset. Earlier Overview and Explorer /
Inspector remain available; automated navigation/state regressions pass.
Screenshot proof is saved in the task visualization directory as
`amp-stage3-summary-light.png`, `amp-stage3-curves-dark.png` and
`amp-stage3-mobile.png`.

## Integrity, limits and next stage

All **543 protected files** match the exact-byte baseline before calculation,
after calculation and after tests. Coverage includes research/source/data,
Stage 1B and Stage 2 derived evidence and their existing catalogs. Original
models, metrics, input files, governance, preprocessing/splits and access-event
records are unchanged. No research evaluator main, training, inference,
calibration, threshold selection or new methodological experiment was run.
Unrelated untracked ZIPs and checkpoint report directory are preserved.

Scores remain reconstructed Stage 1B evidence rather than recovered original
persisted predictions. Cohort disclosure retains historical Stage 13 access;
results do not establish independent external utility/operational generalization.
No applicable frozen AMP-LDP binary threshold/matching confusion matrix exists;
none is displayed. Scores rank review priorities, are not established as
calibrated theft probabilities or proof, and establish neither theft-day
localization nor causal attribution. Missing curve artifacts do not disable
verified aggregates; missing/incompatible aggregate evidence fails closed.

The existing master ledger `docs/preprocessing_decisions.md` is appended for
Interface Stage 3; previous decisions remain historical. Stage 3 changes alone
are reviewed and committed locally with no push. Recommended Stage 4, requiring
separate approval: demo usability and guided narrative, plus provenance-aware
presentation/export if scoped. It must retain frozen research boundaries.
Comparator inference, threshold decisions, explanations or deployment require
specific authorization and supporting evidence. Stage 4 is not implemented.

## Changed-file inventory

- Navigation/presentation: `app/main.py`, `app/views/performance.py`.
- Evidence loading/bindings: `app/performance_loader.py`, `app/performance_catalog.json`.
- Offline visualization: `app/recovery/build_performance_curves.py`.
- Derived evidence: `app/derived/stage3/curves.json`, `curve_provenance.json`,
  `research_snapshot.json`.
- Tests: `tests/interface/test_performance.py`, `tests/recovery/test_performance_curves.py`.
- Documentation: `README.md`, `app/README.md`, `app/VALIDATION.md`,
  `app/STAGE3_DESIGN.md`, `app/STAGE3_VALIDATION.md`, `docs/preprocessing_decisions.md`.
- Byte preservation: `.gitattributes`.

17 stage-specific files; research output/source and earlier feature tests are
unchanged. The local delivery commit is identified in the execution report and
Git history rather than embedding a self-referential commit hash in this file.
