# Interface Stage 4 — audit and implementation plan

Recorded before application changes, 17 October 2026. Base HEAD
`fac126198a464f2e2824f7e13b4705c71268ceed`, branch `study2/mil`, tracked diff
empty. Three untracked ZIPs and `report/checkpoints/september_20_v2/` are
unrelated and will remain untouched. No applicable AGENTS.md was found.
Read the interface README, audit, Stages 1/1B/2/3 reports, recent master ledger,
Stage 1B provenance, Stage 27A/B evidence and frozen method/feature source.

## Findings

- Current native navigation, original-ID Explorer, indexed consumption Inspector
  and verified Performance views already work; preserve their layout/state.
  Existing suites have 83 tests. No research method change is authorized.
- Every artifact adapter checks current hashes and caches immutable parses.
  In a fresh Python process, baseline adapter seconds (first / repeat):
  Overview 0.1982 / 0.0278; Explorer 0.3778 / 0.1623; Performance aggregate
  0.1632 / 0.1058; Performance curves 0.3011 after other loads. Original-history
  first customer 1.5804 / 0.3073, then other customers 0.2986–0.3346 first and
  0.2687–0.2783 repeated. These are local warm-disk observations, not targets.
  The main first-history costs are verifying the 163 MB source and loading
  immutable metadata. No evidence justifies weaker hash checks or dataset copies.
- Missing scores/history/curves already have guarded states; add visible loading
  cues and link recovery instructions to the preflight/runbook.
- Installed Streamlit 1.66.0 supports `client.disableDataExport`: this hides
  built-in dataframe/chart-data CSV and read-only clipboard controls. It is a
  presentation safeguard, not access control. Use it because dataset
  redistribution permission is not established in repository governance.
  Retain local inspection, charts and table search/fullscreen. No customer-level
  download will be added. A single explicit aggregate JSON summary is useful.
- No essential remote assets or API calls found in app source. Set telemetry
  off and viewer toolbar locally, retain theme controls. Dependencies are already
  installed. The sandbox pip cache has no wheel files; a clean isolated reinstall
  without obtaining packages is not presently practical. Do not claim it tested.

## Planned delivery

1. Add optional Demo Guide with five concise timed steps, working callbacks into
   existing pages and authentic preset inspection. Keep Overview as landing page
   and ordinary sidebar navigation available. Retain Explorer query/sort/date
   state; opening the guide's ranking step explicitly returns to ranking.
2. Bind presets to existing Stage 1B score/provenance hashes and recompute their
   identity from validated scores: highest-ranked FLAG=1, lowest-ranked FLAG=0,
   and optional highest-ranked FLAG=0. Disclose label/outcome-conditioned,
   illustrative selection; FLAG never enters inference (none runs here).
   Join all three to the existing history loader. No score or history changes.
3. Add one local aggregate evidence JSON download with full frozen metrics,
   cohort/access disclosure, AP/ROC definitions, source hashes, reconstructed
   score origin and limits. Exclude original customer IDs, examples and histories.
   No PDF/slides/figure dependency or customer export.
4. Add stdlib-entry preflight command with per-check PASS/FAIL, dependency pins,
   critical imports, aggregates, source/provenance/count/schema checks, all
   history input hashes/index membership, preset histories and optional curves.
   Never infer, repair or rewrite evidence. Failures give exact local restoration
   guidance; missing optional curves are explicit warnings.
5. Measure fresh server health/startup and AppTest page transitions/repeated
   interactions with machine context. Check offline operation in a fresh process
   with outbound connections blocked. No speculative optimization or performance
   improvement claims. Retain efficient indexed lookup and current hash caches.
6. Add meaningful tests for guide/preset identity/state, summary export,
   preflight dependencies/missing evidence, offline execution and protected bytes;
   retain all old tests. Browser-check all pages, full five-minute rehearsal,
   themes and laptop/mobile layouts; record actual issues/results.
7. Write DEMO_GUIDE.md with exact preflight/startup, timed clicks, real examples,
   grounded questions/answers, limitations and artifact-specific recovery.
   Update current README/validation pointer and append the existing master ledger
   while preserving prior stages. Review and commit only Stage 4 files; no push.

Before implementation, `derived/stage4/research_snapshot.json` records protected
research/source/data and all previous interface derived evidence/catalogs,
the exact-hashed curve builder and unchanged dependency declaration. Compare
all bytes after validation and before/after commit. Final technical completion
remains subject to supervising acceptance; no further stage is started.
