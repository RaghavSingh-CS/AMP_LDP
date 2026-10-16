# Interface Stage 2 — customer inspection (17 October 2026)

**PASS — technical delivery; supervisor acceptance remains separate.**
All 6,356 held-out customers have authentic source histories and the inspector
is active. No score was regenerated, model fitted, threshold changed or research
artifact rewritten. Stage 3 is not started.

## Source and identity proof

Base HEAD: `b50956e0e0f668326f84ee59a94f42dee51a60d8`, branch `study2/mil`.
The tracked tree was clean. Prior untracked ZIPs/report checkpoint files remain
unrelated and unchanged. `STAGE2_DESIGN.md` records the design before code edits.

| Purpose | Authoritative artifact | Exact SHA-256 |
|---|---|---|
| Original consumption | `data/raw/data set.csv` | `99ba49d416238b400771b97aea71975706ce94989dd5440202402441cbcc4579` |
| Frozen missingness | `data/processed/test_mask.parquet` | `315be291c798f26e1db79256c5ba45b00571270f1ad1a0b5f2f52b0a68552327` |
| Evaluated windows | `data/windows/stride_30/usable/test_windows.parquet` | `237672636f627b8e550baac1cb64b328a29a4a4f356d745a30f84bb3df03c1c2` |
| Historical customer split | `data/processed/split_manifest.parquet` | `6ad1132bbaef0c6f6ccc5bdc973844b805c4291612ac7170ec165f273835959c` |
| Existing scores | `app/derived/stage1b/customer_scores.csv` | `17a7af4ac32a01981549dd6fe4e99809948aa92c82aa611b05214433b0d182ff` |
| New metadata index | `app/derived/stage2/history_index.csv` | `0e84bd18a28e462e3a2965f930be0fc82c4953aec08e1d50d38403e02e1fe465` |

Historical input hashes come from frozen research evidence, including
`results/cnn_autoencoder/stride_30/test_evaluation/test_metrics.json`; the full
inspection lineage is recorded in `app/derived/stage2/history_provenance.json`.
Stage 17A governance and Stage 27A/B metadata remain bound through the unchanged
Stage 1B adapter/catalog. Its ordered customer-membership hash is preserved.

The index builder checked all 42,372 raw source records for original-ID
uniqueness and selected customers through CONS_NO, not row position. All 6,356
scored IDs and labels match the historical test split. Every selected source
missingness mask matches its frozen test mask on the processed calendar; every
evaluated window matches its customer, label, bounds, dates, missing count and
eligibility. The index stores exact byte offset/length and per-record hash.
Runtime additionally verifies the record's own ID and label before plotting.

## Reading and model semantics

The plot uses original numeric CSV readings on real dates, 1 January 2014 to
31 October 2016. There are 1,034 source date columns and 1,035 calendar positions.
The absent 18 September 2016 column is explicitly missing, not an observed zero.
The raw chart retains 3 October 2014; frozen preprocessing excluded it. Genuine
zero readings remain observed. Missing days split line segments. Observed-only
mean/median/min/max/population-SD and coverage counts refer to the full period.
No measurement unit was established, so kWh is not assumed.

The UI distinguishes this source chart from the frozen imputed/log1p/scaled
model inputs. It explains 30-position windows, 17 summaries, 16 latent dimensions,
96 customer features and the five-HGB mean. Usable-window counts and metadata
come from the frozen manifest. Thirty processed positions can span the removed
calendar date. Fourteen trailing positions are not windowed; fully missing
candidate windows are excluded. No attribution, theft-day highlight, risk
threshold, probability calibration or causal interpretation is introduced.

## Executed validation

- **53 interface tests passed**, including all previous 35 and 18 new tests.
- **6 existing offline recovery/verification tests passed** without replaying
  inference. Total **59 passed**, retaining all prior 41 regressions.
- New automated coverage: exact ID/label joins, cross-customer rejection even
  with a repinned offset index, chronological sorting, absent dates, genuine
  zeros, missingness, statistics, source validity, corrupt/deleted cached files,
  unavailable histories, original values/window metadata for real customers of
  both evaluation classes, real zero-readings, exact stored score/rank display,
  search/sort/selection preservation, sequential inspection, per-customer date
  intervals across Back/Overview navigation, and reset behavior.
- Python syntax/import compilation passed. `pip check` found no broken
  requirements. Existing Altair/pyarrow packages were pinned explicitly; no
  package installation was performed.
- Streamlit 1.66.0 started on loopback. `/_stcore/health` returned `ok`; `/`
  returned HTTP 200. Existing Overview/Explorer and the new inspector render.
- Actual browser checks verified selection, original values/coverage, chart
  gaps, horizontal zoom and reset for real normal/theft-labelled customers.
  Light/dark views were inspected; controls stacked at 390-pixel width, and the
  viewport override was reset. Score display was refined after the initial
  laptop screenshot showed full precision being visually truncated: six-decimal
  metric plus separate exact stored value now remain readable.
- The first validation run caught use of a text-normalizing hash reader for
  binary Parquet. The adapter now checks Parquet as exact bytes. The corrected
  suite passed; no research file was changed to resolve the check.

Protected snapshot: **538 files byte-identical**, including all tracked frozen
research/source files, all local data files and all three Stage 1B derived files.
The original score catalog/provenance, model checkpoints, split manifests and
Stage 27A/B outputs remain unchanged. `git diff --exit-code -- src results models
data app/derived/stage1b` passed. No earlier research conclusion was edited.

## Performance and artifacts

The source CSV is 163 MB. Cold access streams its hash once per filesystem
identity. Later access seeks and reads one selected record, not the whole CSV.
Every request checks file existence/identity, selected record bytes/hash/ID and
label. Index parsing and up to 16 window lookups are cached; source changes
invalidate the cached full-file verification. Measured local load times were
0.54 seconds cold, 0.22 seconds for a second customer and 0.20 seconds repeated.
These are local measurements, not guaranteed timing targets.

The new metadata is 903,135 bytes total: 812,696-byte index, 20,026-byte lineage
manifest, and 70,413-byte protected snapshot. No consumption values are copied
or committed. Original data files stay local. Git attributes preserve metadata
bytes. The delivered commit contains only Stage 2 code/tests/docs/dependency
declarations and this compact metadata; no push is authorized or performed.

Startup remains:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

## Limitations and Stage 3 prerequisite

Original local input artifacts must be present and match their bound hashes.
Missing/corrupt histories show an unavailable state while retaining the frozen
score and Back control. Units remain unverified; observations and customer
labels do not establish theft timing or causality. The full data hash is checked
once per filesystem identity rather than on every rerun; the selected record's
bytes are checked every time. The metadata builder is create-only and its
one-time complete validation can take several minutes.

Proposed Stage 3 scope, pending separate approval: a restrained comparison view
using already frozen aggregate evidence, with correct cohort/comparator
disclosures. No training, score regeneration or expanded claims are implied.
No Stage 3 implementation has begun.
