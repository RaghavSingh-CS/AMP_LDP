# Interface Stage 1B — recovery assessment (17 October 2026)

Initial repository: `study2/mil`, HEAD
`b7a085b6f917d75908cb159961383dd57b97803c`; tracked tree clean. Existing
untracked archives and report checkpoint directory are unrelated and preserved.

## Recovery search

No original Stage 27B AMP-LDP customer-score export was found in the 631-file
project inventory (including ignored input artifacts; excluding environments,
Git internals and caches), all local/ref branches and their filename history,
the one registered worktree, reflog inspection, or the four project ZIP archives.
No stash is present. The four ZIPs contain manuscript material and no prediction
or score payload. No remote fetch or unrelated-project scan was needed.
Stage 13, Stage 23B FULL_48 and Stage 24B DEV_VAL predictions are ineligible.

## Feasibility established before inference

- All five fitted HGBs passed `check_is_fitted`, and all five fitted encoder
  state dictionaries exist. All ten byte hashes match the Stage 27A freeze.
- Frozen CPU environment is available: torch 2.13.0+cpu, scikit-learn 1.9.0,
  numpy 2.5.2, pandas 3.0.5 and joblib 1.5.3. No installation is required.
- `data/processed/test.parquet` and usable stride-30 `test_windows.parquet`
  match their historical byte hashes recorded in frozen Stage 10/12 evidence.
- Original split manifest matches Stage 17A's recorded hash. Governance and
  Stage 25B fitted standardizer match their frozen bindings. Already scaled
  consumption is read directly; the preprocessing scaler is not refitted.
- Original Stage 27B preparation and latent-feature helpers and their source
  dependencies are unchanged from evidence commit `dbd978f`. Preparation uses
  saved mean/scale and fixed historical FULL_48 references; there is no fitting.
  The original feature, aggregation, batch and arithmetic ensemble logic can
  therefore be reused without substituting train/validation data.

The user's Stage 1B request explicitly authorizes conditional inference-only
replay and metric recomputation solely for integrity. That authorization
supersedes the previous interface restriction for this narrow operation. It
does not reopen ML development, the original terminal access gate, or Stage 2.

## Replay design

`app/recovery/replay_stage1b.py` calls only original read-only helpers, loads
strict fitted state, uses `eval()`/`no_grad()`, and rejects fitting/backpropagation
through runtime tripwires. It never calls the historical evaluator's `main`,
`open_cohort_once`, or access-event writers. Constructors only allocate the
saved architecture; every tensor is replaced and checked against its fitted
checkpoint. No model is recreated from seeds or training data.

All required model/input/code hashes are checked before inference. Original
ordered customer/window/label hashes, historical test membership and labels,
and protected-role separation must match before scoring. An independent
sklearn check must reproduce both ensemble metrics (and each seed's metrics)
at six decimals. Differences stop integration; predictions are never adjusted.

Outputs are create-only under `app/derived/`, outside frozen research directories.
They are explicitly **reconstructed interface scores**, not original persisted
Stage 27B predictions. The manifest records exact input/output hashes, original
and replay code revisions, environment, scoring rule, verification and a
before/after byte snapshot. The dashboard consumes only the resulting small
CSV/manifest plus its existing evidence, with no inference at app startup.

## Executed replay and independent verification

**Stage 1B PASS. Stage 1 technical gates closed; supervisor acceptance remains
separate. No Stage 2 work was started.** Replay was possible and executed once
under the conditional authorization, after the recovery search found no export.

Original customer order and within-customer window order reproduced the frozen
membership, window and label hashes. All 6,356 customers (542 theft-labelled;
165,020 usable windows) matched FINAL_TEST and historical test membership, with
no protected-role overlap. Every fitted seed reproduced its own frozen metrics
to six decimals. The ensemble check used the original FLAG=1 orientation and
sklearn ROC-AUC / average precision definitions, independently of the historical
evaluator's metric wrapper:

| Metric | Frozen | Reconstructed | Difference |
|---|---:|---:|---:|
| ROC-AUC | 0.8117294176037735 | 0.8117294176037735 | 0.0 |
| Average Precision | 0.42186322254778824 | 0.42186322254778824 | 0.0 |

CSV roundtrip preserves the exact float64 scores. No score adjustment, fitting,
threshold selection, comparator inference or evaluation redesign was performed.
No discrepancy is unresolved. Exact agreement of aggregate metrics is supported
by the frozen input/state/code and ordered cohort checks; it is not a direct
comparison against nonexistent original per-row output bytes.

## Authoritative inputs and hashes

The complete inventory, including each of the ten checkpoint hashes, all five
scoring source hashes, frozen evidence, fitted standardizer, historical hash
sources and environment, is in
`app/derived/stage1b/provenance.json` under `inputs`. Key exact-byte hashes are:

| Input | SHA-256 |
|---|---|
| `data/processed/test.parquet` | `da4cfc4ee72ced02760c720bad8cfc26d14ec4cfe07c83dde9ad4b1caf2cfd0e` |
| `data/windows/stride_30/usable/test_windows.parquet` | `237672636f627b8e550baac1cb64b328a29a4a4f356d745a30f84bb3df03c1c2` |
| `data/processed/split_manifest.parquet` | `6ad1132bbaef0c6f6ccc5bdc973844b805c4291612ac7170ec165f273835959c` |
| `results/governance/stage_17a/research_protocol_manifest.csv` | `dae7dba53260a3af6b8e2ddd904b5c7867c3321f27cdff5af5e4060b97c348c0` |
| `results/supervised/stage_25b/bag_integrity.json` | `fd78f87c85c33c54ba5fce9083a50ea4c0827247a22d0f457be8218e4b00149e` |
| `results/supervised/stage_27a/final_model_freeze.json` | `a5ce3e6db799e3f4053a69d457a7d22a9f45db27bda6e6f08ec87c69b232c14e` |

The preprocessing scaler is also historically hash-verified, but not applied
again: the processed test input is already scaled. No training/validation
consumption file is loaded. Membership metadata includes all roles solely to
verify separation.

## Derived output and activated Explorer

- Scores: `app/derived/stage1b/customer_scores.csv`, 354,095 bytes, SHA-256
  `17a7af4ac32a01981549dd6fe4e99809948aa92c82aa611b05214433b0d182ff`.
- Provenance: `app/derived/stage1b/provenance.json`, with its reviewed hash in
  `app/artifact_catalog.json`. It records the base HEAD at replay time and exact
  new replay/verification script hashes; the delivery commit versions those
  scripts. This avoids claiming an uncommitted script existed at the base HEAD.
- Research byte snapshot: `app/derived/stage1b/research_snapshot.json`, 512
  protected tracked research/source files and required ignored test inputs.
  All were byte-identical before and after replay.
- Total derived evidence size: 440,033 bytes. No checkpoints or large datasets
  were added. Scoped Git attributes preserve the derived evidence's exact bytes.

Only after verification passed was the catalog activated. The Explorer shows
original identifiers, 6,356 scored customers, global descending rank, sortable
risk scores, search, interactive scrolling, and explicitly marked evaluation
FLAG labels. Its artifact path, reconstruction disclosure and expandable
provenance identify the source. It never describes the scores as calibrated
probabilities or invents explanations/theft dates. Missing or corrupted CSV or
provenance returns an unavailable state, including after cached reads.

## Tests executed

- `python -m unittest discover -s tests/interface -v`: **35 passed**, including
  the existing Stage 1 cases, actual 6,356-row integration/search/ranking, schema,
  unique original IDs, finite/range checks, role/label hashes, altered manifests,
  wrong model/output binding, metric mismatch, and missing/corrupted artifacts.
- `python -m unittest discover -s tests/recovery -v`: **6 passed**, including
  independent metric recomputation from the actual CSV, six-decimal agreement,
  error reporting, invalid input/orientation, fitting/backpropagation tripwires,
  and output path/overwrite rejection. These tests do not replay inference.
- Syntax/import compilation passed. Existing research dependencies were reused.
- Streamlit 1.66.0 launched on loopback; health returned `ok` and root HTTP 200.
- Actual browser verified Overview, Explorer count/table, original-ID search and
  alternate risk sorting. The existing native UI and theme behavior are retained.
- Protected files matched the 512-entry byte snapshot after validation; tracked
  research/source directories have no Git diff. Original Stage 27B metrics and
  access event remain unchanged.

Exact dashboard startup and fresh-directory replay commands are documented in
`app/README.md`. The initial sandbox torch inspection did not complete promptly;
CPU checkpoint/replay and runtime checks completed outside that restriction.
No additional package installation, fitting or protocol change was needed.

The existing master ledger has an appended Interface Stage 1B entry. The local
delivery commit includes only relevant interface code/tests/docs and the small
derived evidence; existing untracked archives/report files remain excluded.
No push is authorized or performed. There are no remaining Stage 1B blockers;
future methodological work and Stage 2 still require explicit approval.
