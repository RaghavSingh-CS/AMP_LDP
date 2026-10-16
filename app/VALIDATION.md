# Interface Stage 1 validation — 17 October 2026

Historical Stage 1 record. Interface Stage 1B subsequently closed the score gap
through explicitly authorized, verified inference-only replay. Current status
and executed Stage 2 inspection checks are in [STAGE2_VALIDATION.md](STAGE2_VALIDATION.md).
The earlier score-recovery record is in [RECOVERY_STAGE1B.md](RECOVERY_STAGE1B.md).

**Technical delivery status: PARTIAL; supervisor acceptance remains separate.**
The missing canonical Stage 27B AMP-LDP score export prevents authentic
customer-score display. No substitute model outputs or generated predictions
were used. Stage 2 remains unstarted.

## Executed checks

| Check | Result |
|---|---|
| Syntax/imports | `python -m compileall -q app tests/interface` passed; app and adapter imported successfully |
| Automated suite | `python -m unittest discover -s tests/interface -v`: 24 tests passed |
| Adapter-only suite in restricted environment | 21 tests passed after placing temporary fixtures inside the workspace |
| Dependencies | `python -m pip check`: no broken requirements |
| Runtime | Streamlit 1.66.0 started on Windows Python 3.12, loopback port 8501, no CUDA |
| HTTP | `/_stcore/health` returned `ok`; `/` returned HTTP 200 |
| Actual browser | Overview showed 6,356 / 542 / 0.8117 / 0.4219; navigation selected Explorer and showed unavailable state |
| Theme/layout | Actual browser screenshots inspected in dark and light themes at desktop size; 390×844 viewport stacked the metric columns with readable text; viewport override reset |
| Interactive score table | Streamlit AppTest passed table, search, literal bracket query, alternate sorting and empty-results checks using explicitly mocked synthetic test records only |
| Frozen source agreement | All six catalog bindings match SHA-256 of LF-normalized blobs from committed Stage 27B evidence |
| Research regression safety | 463 tracked `results/`, `models/`, `data/` artifacts matched Git object hashes; no content differences; all ten Stage 25B encoder/HGB byte hashes matched Stage 27A freeze |
| Research source safety | `git diff --exit-code -- src results models data` passed |
| Existing relevant suite | No existing unit-test suite was present; historical evaluation entry points were not executed because they would reopen evaluation/write research outputs |

Adapter coverage includes missing evidence after a successful cached read,
changed content/hash, CRLF/LF compatibility, malformed JSON/CSV schema,
invalid metrics/counts, duplicate model rows, path traversal, absent canonical
scores, and genuine FINAL_TEST governance counts. Prediction-contract coverage
includes schema, duplicate/empty/non-string IDs, nonnumeric/nonfinite/out-of-range
scores, incorrect cohort size, training-member substitution, ordered membership
hash, label validity/count/hash, deterministic ties, original IDs (including
leading zeros and literal `NA`), search/sort, and complete fixture export loading
with rejected alternative-model provenance and missing-file handling.

The first complete test run exposed a fixture-construction error: pandas 3's
string dtype refused an integer before the validator could receive it. The test
now explicitly uses an object column for invalid-ID cases. The corrected suite
passed. Early restricted-sandbox attempts encountered denied temporary-directory
operations and blocked local sockets; runtime tests and HTTP checks were rerun
successfully outside that restriction. AppTest's expected bare-mode
`missing ScriptRunContext` warning did not cause failures.

## Acceptance gates

| Gate | State |
|---|---|
| 1. Streamlit launches | Met |
| 2. Verified Overview metrics | Met |
| 3. Functional navigation | Met, AppTest and actual browser |
| 4. Genuine customer scores | Unmet: canonical Stage 27B export absent; unavailable state implemented |
| 5. Missing/malformed data safety | Met |
| 6. Automated validation | Met, 24 tests |
| 7. Research artifacts untouched | Met |
| 8. Existing master ledger updated | Met, separate Interface Stage 1 section |
| 9. Reproducible startup docs | Met, interface README and minimal requirements |
| 10. Reviewed safe Git commit | Included in the Interface Stage 1 delivery commit; commit ID reported to the user |

No actual AMP-LDP customer-score table can be browser-validated until its
authoritative export is recovered and approved. Real-data integration is complete
for aggregate evidence only. Hash bindings cannot establish the origin of a
future export without a reviewed chain of custody. These limitations prevent PASS.
