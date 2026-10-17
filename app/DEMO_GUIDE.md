# AMP university demo runbook

17 October 2026. About five minutes, after preflight. Start on **Overview**;
**Demo Guide** is an optional cue sheet, not a compulsory tutorial. Ordinary
sidebar navigation always works. Selecting a customer reads validated frozen
scores and original indexed observations; it **does not run inference live**.

## Before the audience arrives

1. Use this local repository and its approved original data files. Keep the
   machine on power. Install the pinned interface dependencies before going
   offline; no GPU, torch or sklearn is needed to use the dashboard/preflight.
2. In PowerShell run:

   ```powershell
   Set-Location C:\Projects\ML_Project
   .\.venv\Scripts\python.exe -m app.preflight
   .\.venv\Scripts\python.exe -m pip check
   ```

   Expect **AMP DEMO PREFLIGHT: PASS**, five dependency passes and seven other
   checks. Optional AMP-LDP curves can produce WARN while verified aggregate
   evidence remains available. Investigate required FAIL results before starting.
   Preflight never repairs, fits, scores, selects a threshold or writes evidence.
3. Start the app and open the loopback URL:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false
   ```

   Open `http://127.0.0.1:8501`. Expected fresh landing page: **Overview**, with
   6,356 customers, 542 theft-labelled, ROC-AUC 0.8117 and AP 0.4219. Performance
   displays six decimals. Stop the server with Ctrl+C after the demonstration.
4. Visit each page once and open both main preset histories to warm caches and
   verify the projector/theme. First-process imports and source hash checks take
   longer than repeated interactions. Choose Light or Dark in the top-right
   menu. Do not fit, rebuild indices, replay scores or alter evidence on stage.

## Five-minute sequence and clicks

| Elapsed time | Page and click | Presenter cue |
|---|---|---|
| 0:00–0:45 | Overview; then Demo Guide → Open customer rankings | “AMP prioritizes customers for review using SGCC consumption and customer-level labels. Our final cohort has disclosed historical baseline access; this is an internal governed research evaluation.” |
| 0:45–1:35 | Explorer; search an original ID, change Sort customers, clear search when finished | “These are 6,356 genuine held-out customer rankings. The displayed predictions were reconstructed once by an authorized frozen replay, with both metrics matching exactly. FLAG is evaluation information, not an inference input here.” |
| 1:35–3:00 | Demo Guide → high-ranked theft-labelled preset → Inspect verified example; then Guide → low-ranked normal-labelled preset → Inspect verified example | “This is the original customer's reading history, with gaps and real zeros. Hover a point; choose an interval and Reset viewed interval. The chart does not identify a theft day or explain causality. Units are not established.” |
| 3:00–4:20 | Demo Guide → Open Model Performance; scroll comparison and curves | “ROC-AUC is 0.811729; AP is 0.421863. FULL_48 and DIRECT use the same cohort and customer metrics. Both chart axes start at zero. Only AMP-LDP curves have supported coordinates. AP uses the original non-interpolated definition, not trapezoidal PR area.” |
| 4:20–5:00 | Guide → Conclusion, or remain on Performance limitations | “Learning window representations before distribution aggregation adds customer-level discrimination beyond DIRECT under this protocol. Scores are review priorities, not calibrated probabilities or proof. External utility deployment validity and theft-day localization are not established.” |

Return to **Demo Guide** using the sidebar between steps. **Open customer
rankings** returns to the ranking view without erasing your search/sort. If a
previous search remains, clear its text to show all customers. Inspector's
**Back to customer ranking** preserves search/sort and selection. **Previous
rank / Next rank** follow global order. Each customer's date interval is retained.
The optional difficult example is a substitute for part of the second history
discussion, not extra material needed to finish in five minutes.
If the browser retains a page's scroll position, use the sidebar's **Back to
page start** link before explaining its metrics. Scroll outside the consumption
plot or use the page scrollbar: wheel events over that plot intentionally zoom
its horizontal axis. **Reset viewed interval** clears that zoom.

## Verified illustrative customers

| Preset | Original CONS_NO | Global rank | Frozen score (six decimals) | Evaluation FLAG | Reproducible rule |
|---|---|---:|---:|---:|---|
| High-ranked theft-labelled | `45C64D484FC7B1A767557CF0674330E0` | 1 | 0.879706 | 1 | Smallest global rank among FLAG=1 |
| Low-ranked normal-labelled | `D5F510E7EFD856CDA64770568C25017B` | 6356 | 0.005418 | 0 | Largest global rank among FLAG=0 |
| Difficult high-ranked normal-labelled | `C21A9C87D49E6945FB4570F38F73C023` | 13 | 0.792207 | 0 | Smallest global rank among FLAG=0 |

These are deliberately **label/outcome-conditioned illustrative choices**, not a
representative sample or new classification evaluation. “Lower ranked” does not
mean a frozen binary decision was made. The difficult case shows disagreement
between ranking and the evaluation label without inventing a false-positive
threshold. Do not describe FLAG as an independently verified theft event.

`demo_catalog.json` binds the original IDs and rules to Stage 1B score SHA-256
`17a7af4ac32a01981549dd6fe4e99809948aa92c82aa611b05214433b0d182ff`
and provenance SHA-256
`e38b28f37231acabf2b42bd6912f65ac2eb6b99d5db4aecc372ab75381b582b3`.
Runtime recomputes the choices from validated scores and rejects changed IDs,
rules or source binding. Global ranking uses descending score, identifier for
ties. The unchanged Stage 2 history index and per-record hash/ID/label checks
trace each selection to original observations. No reading or score is altered.

## Main results and local export

| Frozen Stage 27B method | Customer ROC-AUC | Average Precision |
|---|---:|---:|
| AMP-LDP, five-seed mean | 0.811729 | 0.421863 |
| Original Stage 17B FULL_48 | 0.797863 | 0.385326 |
| Stage 25B DIRECT | 0.794967 | 0.368439 |

All use DEV_TRAIN-only fitting and the same 6,356-customer final protocol;
542 positives yield 8.5274% prevalence. AMP-LDP minus FULL_48 is +0.013867 ROC
and +0.036538 AP; minus DIRECT is +0.016762 and +0.053424. These are absolute
metric differences, not relative percentage improvements or significance claims.

On Performance, **Download aggregate evidence summary** creates one local JSON
file with full-precision metrics, cohort/access disclosure, metric definitions,
source hashes, score origin and limits. It contains **no original customer IDs,
preset list, per-customer scores or histories**. No new customer export is
provided. Repository governance does not establish permission to redistribute
individual records, so built-in CSV and read-only clipboard exports are disabled
using supported Streamlit configuration. This is a presentation safeguard, not
security/access control: data are still visible and browser tools/screenshots
can capture them; chart PNG controls remain local. No public sharing permission
or production deployment is implied. Keep example records and screenshots local
to the authorized demonstration.

## Likely evaluator questions

- **What problem does AMP solve?** Customer-level theft-risk discrimination and
  review prioritization from daily smart-meter histories, under SGCC's available
  customer labels. It is research evidence, not a deployed enforcement system.
- **Why learned representations?** The encoder learns a window representation
  before distribution statistics summarize variation across each customer's
  windows. DIRECT controls for the same aggregation without learned encoding.
  The frozen evidence supports this representation's incremental discrimination
  under the evaluated protocol, not a universal neural-network advantage.
- **What are the 17 input summaries?** In frozen order: mean, median, minimum,
  maximum, q10, q90, population standard deviation, range, interquartile range,
  mean absolute deviation, first-to-last change, absolute first-to-last change,
  mean absolute consecutive change, maximum absolute consecutive change,
  linear slope, `fraction_below_minus_one`, second-half minus first-half mean.
  They are computed over the frozen preprocessed 30-position window, not raw
  kWh. The historical fraction name uses values **at or below −1.0** in code.
  Source: Stage 25B `bag_integrity.json` ordered features and Stage 14
  `behavioural_features`; do not recompute or reinterpret them during the demo.
- **Why five HGB models?** Five fitted seed-specific encoders/HGBs (42–46) and
  their unweighted arithmetic mean were frozen before final evaluation and
  positively replicated in confirmation. The UI follows this fixed ensemble,
  rather than selecting a best seed or creating a new predictor.
- **What does ROC-AUC measure?** Ranking a positive customer above a negative
  customer across score thresholds, with half credit for ties. It is not accuracy.
- **Why AP?** Positives are uncommon; AP summarizes precision–recall retrieval
  using the original sklearn recall-increment weighted precision definition.
  Prevalence is context, not a promised performance level.
- **How do the comparators differ?** DIRECT: six statistics over 17 standardized
  summaries, 102 features and HGB. FULL_48: engineered customer profiles including
  chronology/persistence, original Stage 17B HGB. AMP-LDP: 17→32→16 encoder,
  six latent statistics yielding 96 features, then five frozen HGBs and mean.
- **Why no FC/CNN/LSTM in the final comparison?** Those historical models train
  on normal windows and score reconstruction MSE, primarily at window level.
  Their customer aggregates are diagnostics under another protocol. A shared
  source split alone does not make their metrics directly comparable.
- **Can AMP identify the exact theft day?** No. Customer-level FLAG provides no
  verified theft timing. A plotted spike or missing day is not a detected theft.
- **Does a high score confirm theft?** No. It supports review priority, not proof.
- **Why aren't scores necessarily probabilities?** Estimator scores have not
  been established as calibrated theft probabilities. There is also no applicable
  frozen AMP-LDP binary operating threshold or matching confusion matrix.
- **Does selection run inference live?** No. Stage 1B executed one authorized
  frozen inference replay offline. Scores/labels and both metrics were validated;
  the interface now reads that immutable export and original histories.
- **What are the limitations?** Internal SGCC evaluation; disclosed historical
  Stage 13 baseline access; reconstructed export origin; no comparator curve
  outputs available; no calibrated probability, causal attribution, exact theft
  day or independently established external/operational generalization.

## Troubleshooting and exact recovery paths

Preflight prints the failed dependency/artifact, reason and recovery guidance.
Use `python -m app.preflight --json` for a structured report. Do not weaken hash
checks, fabricate fallback data, retrain, replay or silently rebuild on stage.

| Symptom | What to check and restore |
|---|---|
| Missing/wrong dependency | Before going offline, run `.\.venv\Scripts\python.exe -m pip install -r requirements-interface.txt`. For an offline install, obtain an approved complete wheelhouse beforehand; none is bundled. |
| Metrics/comparison unavailable | Restore the exact Stage 27A/B files named by the error and reviewed catalogs. Tracked original evidence is available in source commit `dbd978f`; compare to the reviewed local Git copy before restoring and preserve unrelated local edits. |
| Scores/presets unavailable | Restore `app/derived/stage1b/customer_scores.csv`, `provenance.json` and reviewed `app/artifact_catalog.json`; original Stage 1B delivery `b50956e` is the local recovery reference. Verify the hashes above. Do not rerun the replay. Restore `app/demo_catalog.json` from the Stage 4 delivery if its selection rule/ID changed. |
| History unavailable, score still visible | Restore the approved matching `data/raw/data set.csv` (SHA `99ba49d416238b400771b97aea71975706ce94989dd5440202402441cbcc4579`), `data/processed/test_mask.parquet` and `data/windows/stride_30/usable/test_windows.parquet` from local research backup. Restore Stage 2 index/provenance/catalog from `efe2ca7`. Preflight names the failing file; Stage 2 provenance binds every hash. |
| Curves unavailable | Restore `app/derived/stage3/curves.json`, `curve_provenance.json`, `performance_catalog.json` from `fac1261`. Verified aggregate metrics remain usable; do not approximate comparator curves or rerun builders. |
| Blank customer table | Clear the search field; check Sort customers. No-match results are deliberate, not missing predictions. |
| Incomplete interval | Choose both endpoints, or Reset viewed interval. An all-missing interval legitimately has no observed consumption line. |
| Server/port problem | Keep the original server terminal open; inspect its output. Health check: `(Invoke-WebRequest http://127.0.0.1:8501/_stcore/health).Content`. If 8501 is occupied, choose `--server.port 8502` and open that same port; stop only your own old demo process. |
| First page feels slow | Start and visit pages before the audience arrives. Imports and initial full source hash are expected cold costs. Never disable verification to speed the demo. |

After restoring a reviewed file, rerun preflight and restart the demo server.
Keep the existing validated repository/venv and approved data together. A clean
isolated dependency reinstall was not validated; the repository does not ship a
wheelhouse and the inspected local pip cache contained no installable wheels.

## Validation and final handoff

See `STAGE4_VALIDATION.md` for executed tests, performance/machine measurements,
offline network isolation, browser rehearsal, limitations and changed files.
Stage 4 is the final interface-development stage, subject to supervising
acceptance. No further research or development stage is started automatically.
