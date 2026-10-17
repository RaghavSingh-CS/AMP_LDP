# Interface Stage 4 — final demo validation

17 October 2026. **PASS — technical delivery, subject to supervising acceptance.**
Base `fac126198a464f2e2824f7e13b4705c71268ceed`, branch `study2/mil`.
Stage 4 completes the authorized interface-development scope. No further
research/development stage started; no push authorized or performed.

The audit and plan were recorded before app changes in `STAGE4_DESIGN.md`.
Previous stage reports and master-ledger entries remain historical.

## Completed usability and narrative

- Overview remains the fresh landing page. Optional Demo Guide provides five
  concise timed steps and functioning Overview/ranking/preset/Performance
  callbacks using the same sidebar state. Ordinary navigation is always available.
- Search, sort, original identifiers, selection, Back/Previous/Next and each
  customer's viewed date interval remain functional and tested. Guide entry to
  rankings explicitly returns to the ranking view without erasing filters.
- Sidebar Back to page start is a native same-page anchor, tested from the
  conclusion. This addresses retained scroll positions without a JS framework.
- Genuine history loading has a verification spinner and informative recovery
  instructions on unavailable inputs, while frozen score and Back remain usable.
- Browser review found history SVG overflow at 390 pixels: a layered vconcat was
  given pad autosizing by Streamlit. Explicit fit-x sizing and a stacked coverage
  legend correct it. Final SVG width 348, main scroll width 380, viewport 390;
  no page-level horizontal overflow. Source values, explicit gaps, zeros,
  tooltips, horizontal pan/zoom, shared dates and reset are preserved.
- Precision audit found pandas.to_numeric rounded final digits for 5,846 stored
  score strings. Python float parsing now preserves every CSV roundtrip exactly.
  No score CSV changed; all 6,356 ranks and six-decimal values are identical.
  New tests compare every original string's float and the full global ordering.

## Verified presenter examples and traceability

| Criterion | Original CONS_NO | Rank | Six-decimal score | FLAG |
|---|---|---:|---:|---:|
| Smallest rank among FLAG=1 | `45C64D484FC7B1A767557CF0674330E0` | 1 | 0.879706 | 1 |
| Largest rank among FLAG=0 | `D5F510E7EFD856CDA64770568C25017B` | 6356 | 0.005418 | 0 |
| Smallest rank among FLAG=0, optional difficult case | `C21A9C87D49E6945FB4570F38F73C023` | 13 | 0.792207 | 0 |

`demo_catalog.json` binds IDs/rules to the unchanged Stage 1B source and provenance
hashes. Runtime recomputes choices from validated global ranks and rejects wrong
IDs, source hashes, selection rules or missing label-selection disclosure. All
three histories pass Stage 2 raw record/mask/window/index lineage checks.
These label/outcome-conditioned selections are illustrative, not representative;
FLAG is never an inference input in this interface. No threshold-based correctness
claim is made. Scores remain reconstructed via the single authorized frozen
Stage 1B replay, not original persisted Stage 27B predictions or live inference.

## Export and sharing policy

One explicit local JSON export contains the three full-precision frozen metric
rows, verified cohort/prevalence, original AP/ROC definitions, historical-access
disclosure, source hashes, score reconstruction context and limits. It does not
contain any original customer ID, preset list, per-customer score or history.
Automated tests exclude all 6,356 original IDs and check every exported metric.
The real browser downloaded `amp-frozen-evaluation-summary.json` (4,445 bytes);
the saved file was parsed and checked against frozen metrics/counts and all IDs.
No file was uploaded or shared remotely.

Repository governance establishes research roles, not individual-record
redistribution permission. Built-in dataframe/chart-table CSV and read-only
clipboard exports are hidden using supported Streamlit 1.66.0 configuration.
The browser verified CSV controls absent; explicit aggregate download remains.
This is a presentation safeguard, not access control: browser tools, visible
data/screenshots and local chart PNG controls remain possible. No sharing
permission or public deployment is asserted. No PDF/slide/extra figure system
or dependency was added. Viewer toolbar removes developer/Deploy controls and
retains theme controls.

## Preflight, offline and fresh startup

Executed `.\.venv\Scripts\python.exe -m app.preflight` and `--json`:
**PASS, 12 checks, zero optional warnings**. Five exact pinned dependencies,
local demo configuration, critical imports/main syntax, frozen aggregates and
comparator compatibility, Stage 1B scores/provenance/schema/ordered cohort,
presenter selection, original history inputs/index and all three selected
histories passed. AMP-LDP curve source/coordinate checks passed. Required
failures return status 1; missing optional curves are explicit WARN with verified
aggregates retained. Tests cover missing/wrong dependencies, missing raw source,
wrong provenance/IDs/rules, optional curve absence and loss during a preset click.
No automatic restoration, builders, fitting or inference is invoked.

Fresh preflight, Streamlit server and AppTest processes also ran with the opt-in
`tests/offline/sitecustomize.py` guard blocking external Python TCP, UDP and DNS
access while allowing loopback. Tests explicitly prove those blocks are active
and verify preflight imports neither torch, sklearn nor the replay module.
All views/navigation work in that profile. Health returns `ok`/HTTP200 and the
root returns HTTP200. Browser DOM asset URLs are local Streamlit JS/CSS only;
no browser chart errors were recorded. Telemetry is disabled in repository config.

This is process-level network isolation, not a claim that the OS/browser was
physically disconnected. No essential remote asset/API/service/download is
used by the app. No new packages were installed. The existing CPU environment
was tested; an isolated clean dependency reinstall was not performed because
no complete wheelhouse was supplied and the inspected pip cache had no wheels.
Installation instructions are documented rather than claimed as tested offline.

## Actual performance observations

Reproducible measurement command:

```powershell
.\.venv\Scripts\python.exe -m app.benchmark --offline
```

Windows 11 build 26300, Python 3.12.14, AMD64 Family 23 Model 104 Stepping 1
(AuthenticAMD), 16 logical CPUs; Streamlit1.66.0 / pandas3.0.5 / numpy2.5.2 /
pyarrow25.0.1 / Altair6.3.0. No GPU requirement. Single local samples; fresh
Python processes with OS disk cache uncontrolled. UI timings are AppTest
rendering/serialization, not browser paint or guaranteed end-user latency.

| Measurement, seconds | First recorded run | Second recorded run |
|---|---:|---:|
| New server → health HTTP200 | 7.4946 | 3.4204 |
| Overview fresh process, including library imports | 13.0340 | 12.9994 |
| Explorer first / repeat | 0.3549 / 0.1615 | 0.2789 / 0.1591 |
| Inspector first / repeat | 1.0493 / 0.4784 | 1.4278 / 0.5579 |
| Inspector next rank | 0.5976 | 0.6001 |
| Performance first / repeat | 3.3876 / 2.4912 | 2.7135 / 2.4533 |
| Guide first / repeat | 0.1631 / 0.1561 | 0.1423 / 0.1380 |
| Return to Overview | 0.0342 | 0.0313 |

Health readiness does not mean the first page has finished rendering. Cold
imports, first large-source hash and chart serialization dominate; repeated
index-based interactions are suitable for a local live demonstration after
prewarming. There is no controlled before/after speedup claim. No evidence
justified weakening artifact verification, changing preprocessing or adding
dataset copies. The final chart layout correction does not change its data.

## Automated and browser validation

Final commands executed:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/interface -q
.\.venv\Scripts\python.exe -m unittest discover -s tests/recovery -q
.\.venv\Scripts\python.exe -m compileall -q app tests/interface tests/recovery tests/offline
.\.venv\Scripts\python.exe -m pip check
```

**98 interface + 8 offline tests = 106 passing**, retaining every previous 83.
No old test was weakened or removed. New coverage includes real presets,
selection/provenance enforcement, exact score-string parsing, guide navigation,
search/sort/date state, exports with correct metric definitions and no IDs,
dependency failures, source loss, optional curve warning, offline fresh-process
readiness and unchanged protected research. Syntax/import checks pass; pip
reports no broken requirements. Expected bare-mode AppTest context warnings
are not failures.

Actual browser checked Overview, Explorer search and alternate sorting,
both main authentic preset histories, Performance tables/bars/curves/export,
Guide and conclusion, and Back navigation. A complete simulated walkthrough
ran from approximately 10:45 to 10:50 IST, covering all five steps. Its browser
automation download wait added roughly 98 seconds; that transport observation
is not claimed as application download latency. The timed presenter content is
45+50+85+80+40=300 seconds; the download is optional and can be prepared before
the audience arrives. No broken application transition remained.

Light/dark themes, desktop1280×720, laptop1024×768 and mobile390×844 were
inspected. Guide headings/cues and real IDs wrap/fit, mobile metrics stack and
Performance settles without page overflow. History overflow was corrected as
described above. Mobile sidebar must be dismissed to interact with underlying
content; history-wheel zoom is intentional and the runbook explains reset.
Some automation observations arrived during resize/animation; checks used the
settled page before recording final bounds. Temporary viewport override reset;
original dark theme restored. Loading feedback was observed during cold startup.

Missing-history browser state used **only** the explicit
`tests/interface/demo_failure_app.py` fixture on separate loopback8503, patching
history access to raise a clearly labelled TEST FIXTURE error. Genuine score,
rank, FLAG and Back remained visible; recovery text appeared without a crash.
The fixture tab/server were closed afterward. No original source was moved,
deleted or corrupted for the test. Normal browser remained on loopback8501.

Screenshot proof in the task visualization directory:
`amp-stage4-guide-light.png`, `amp-stage4-mobile-history.png`,
`amp-stage4-missing-history-fixture.png`. Browser-downloaded JSON was verified
locally in Downloads, not added as a redundant committed aggregate artifact.

## Protected integrity, delivery and remaining limits

All **549 protected files** match `derived/stage4/research_snapshot.json` exact
bytes before/after implementation and tests. Coverage includes research/source/
data, all earlier derived scores/histories/curves/catalogs, exact-hashed curve
builder and the unchanged interface dependency declaration. All original metrics,
checkpoints, input files, governance, access event and preprocessing/splits remain
unchanged. Tests may recompute existing-score metrics/coordinates for regression
integrity only; no new evaluation protocol, inference, fit, threshold selection
or research write occurs. Existing unrelated ZIPs/checkpoint report remain intact.

No calibrated probability, theft proof/day localization, causal explanation,
new significance claim, external generalization or production deployment is
established. Original persisted Stage 27B row predictions and comparator curves
remain absent, with reconstructed-origin and historical-access disclosures
retained. Local original artifacts and matching hashes are necessary. No public
record-sharing permission is inferred from the internal research governance.

The existing `docs/preprocessing_decisions.md` gains Interface Stage 4 decisions;
earlier stages are preserved. Only Stage 4 files are reviewed/staged/committed
locally, with no push. The execution report/Git history identify the delivery
commit without a self-referential embedded hash. Recommendation: ready for the
supervised local university demonstration after preflight and prewarming;
supervising acceptance remains separate. Do not begin another stage automatically.

## Changed-file inventory

- Runtime/configuration: `.streamlit/config.toml`, `.gitattributes`, `app/main.py`,
  `app/data_loader.py`, `app/navigation.py`.
- Guide/export/evidence: `app/demo.py`, `app/demo_catalog.json`,
  `app/views/demo_guide.py`, `app/views/explorer.py`, `app/views/inspector.py`,
  `app/views/performance.py`.
- Readiness/measurement: `app/preflight.py`, `app/benchmark.py`.
- Protected baseline: `app/derived/stage4/research_snapshot.json`.
- Tests: `tests/interface/test_demo.py`, `tests/interface/demo_failure_app.py`,
  `tests/offline/sitecustomize.py`.
- Documentation: `README.md`, `app/README.md`, `app/VALIDATION.md`,
  `app/DEMO_GUIDE.md`, `app/STAGE4_DESIGN.md`, `app/STAGE4_VALIDATION.md`,
  `docs/preprocessing_decisions.md`.

24 stage-specific files. Prior stage reports, research source/output and old tests
are unchanged.
