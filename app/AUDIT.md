# Interface Stage 1 audit — 17 October 2026

Recorded before implementation. Repository root is `C:/Projects/ML_Project`,
branch `study2/mil`, initial HEAD `034a2d1`. Remote is
`https://github.com/RaghavSingh-CS/ML_Project.git`. The tracked tree was clean;
three ZIPs and `report/checkpoints/september_20_v2/` were already untracked and
must remain untouched. No applicable AGENTS.md was found. The existing master
ledger is `docs/preprocessing_decisions.md`; its ML history ends at Stage 28.
Later manuscript work is reflected in Git and `report/`.

## Evidence and provenance concerns

- Stage 27A `final_model_freeze.json` verifies SUMMARY_17, encoder 17→32→16,
  six statistics per latent dimension (96 features), five frozen HGBs, seeds
  42–46, arithmetic ensemble, DEV_TRAIN-only fitting. All ten checkpoint
  hashes match the freeze. No checkpoint needs loading in the interface.
- Stage 27B `final_primary_metrics.csv` supplies LATENT ROC-AUC
  0.8117294176037735 and AP 0.42186322254778824. `final_integrity.json`
  supplies 6,356 eligible customers, including 542 theft-labelled customers.
- Stage 27A preregistration, Stage 27B access event and integrity agree on the
  final cohort. Required wording is “final evaluation cohort with disclosed
  historical baseline access”; historical Stage 13 access must be disclosed.
- A full workspace inventory (excluding environment and raw/processed/window
  input directories) found no Stage 27B customer predictions. Its evaluator
  computes scores in memory and writes aggregate/seed metrics only. Stage 23B
  customer predictions are refitted FULL_48, Stage 13 predictions are an obsolete
  baseline, and Stage 24B predictions are DEV_VAL. None can stand in for AMP-LDP.
- Governance IDs and roles are in
  `results/governance/stage_17a/research_protocol_manifest.csv`. The final
  integrity artifact supplies ordered membership/label hashes, not individual
  labels. Thus no canonical per-customer AMP-LDP score/label join is available.
- `dashboard/` is empty; no existing application or unit-test suite was found.
- Windows Python is 3.12; `.venv` has numpy 2.5.2 and pandas 3.0.5, no Streamlit
  or pytest. Existing `requirements.txt` includes GPU-capable research packages;
  the interface will have its own minimal CPU dependency list and use unittest.
- Historical Stage 28 documentation explicitly does not authorize pushing.
  Commit locally; do not push.

## Proposed implementation

`app/main.py` will provide native Streamlit sidebar navigation. `app/config.py`
and `app/artifact_catalog.json` will hold paths and reviewed SHA-256 bindings;
`app/data_loader.py` will validate small immutable evidence without importing
training code. `app/views/` will separate Overview and Explorer presentation;
using `views` avoids Streamlit's automatic `pages/` navigation. Tests belong in
`tests/interface/`. `app/README.md` will document reproduction and the export
contract. Append a separate Interface Stage 1 section to the existing ledger.

Overview can display authentic aggregate evidence immediately. Explorer must
show an unavailable state until a canonical, approved Stage 27B export and its
provenance binding exist. No inference, metrics recomputation, substitute model,
threshold, raw consumption loading or research-output writes are needed.
