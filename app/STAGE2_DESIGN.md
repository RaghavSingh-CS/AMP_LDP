# Interface Stage 2 design — 17 October 2026

Recorded before implementing the inspector. Base HEAD is `b50956e`, branch
`study2/mil`; tracked tree is clean. The three existing untracked ZIPs and report
checkpoint directory are unrelated and preserved. Existing interface README,
audit, validation, Stage 1B recovery and master ledger records were reviewed.

## Findings and interpretation

- Immutable `data/raw/data set.csv` exists (163,119,034 bytes); CONS_NO is unique.
  An explicit ID join finds all 6,356 Stage 1B customers, with matching FLAG.
- Raw headers supply 1,034 actual date columns from 1 January 2014 through
  31 October 2016, lacking 18 September 2016. The complete daily range has 1,035
  positions. The absent source column will be explicitly marked missing.
- Frozen cleaning removes 3 October 2014 because only 15 source customers had a
  reading; inserts 18 September 2016 as missing; preserves genuine zero values;
  saves the original missingness mask; interpolates/fills gaps within customer;
  applies log1p. The training-fitted StandardScaler then produces test.parquet.
- The chart will show source numeric values, including any observed reading on
  the model-excluded date. It will not invert processed values or impute gaps.
  Measurement units were not established by the inspected source metadata, so
  the axis will state source units, not assume kWh.
- Model windows are 30 consecutive positions on the frozen 1,034-position
  processed calendar, stride 30. Eligibility is missing_count < 30. There are 34
  full candidate windows and 14 trailing unused positions. Across the removed
  calendar date, 30 positions need not mean 30 contiguous calendar days.
- Exact Stage 1B scores/provenance remain unchanged; no inference is needed.
- Plotly is absent. Installed Altair 6.3.0 provides native interactive charts
  without installing another plotting package.

## Intended implementation

Use an integrated detail view inside Explorer. Search/sort lead to an explicit
customer selector and Open inspector button. Store view, selected original ID,
query, sort order and per-customer date interval independently of widget keys in
session state. Back returns to the same ranking context; adjacent-rank controls
support sequential inspection. Display score, global rank, evaluation-only FLAG,
cohort and original score provenance without thresholds or explanations.

A read-only `consumption_loader.py` will seek to one raw CSV record using a
small reviewed index under `app/derived/stage2/`. An offline index builder will
scan the source once, bind exact source/hash/offset/length/ID, validate all scored
customers against governance, frozen test masks and usable-window metadata,
and write only interface metadata (no duplicate consumption dataset). Runtime
checks verify lineage and the row's own ID/label, never assume positional joins.
Bounded caches will use file identity plus hash verification; missing/changed
artifacts will fail clearly. Sources and Stage 1B evidence remain immutable.

An Altair line/point chart will use real dates and separate observed segments
so it cannot connect across missing days. A coverage strip will distinguish
observed, missing and absent-column days; genuine zeros remain observed points.
Provide hover values, horizontal pan/zoom, a date interval control and reset.
Full-period counts/statistics will use observed source values only. Explain the
frozen processed pipeline and show actual evaluated-window metadata separately.

Focused tests will cover identity isolation, chronological order, zeros versus
missingness, statistics, corrupt/missing inputs, real customers from both
evaluation classes, exact score/rank and navigation/state. Run all existing 41
tests, syntax/startup/browser checks, and compare protected byte snapshots.
Update the existing ledger and interface docs, review and commit only Stage 2
files, do not push, and do not begin Stage 3.
