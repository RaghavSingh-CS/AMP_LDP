# Anomaly-Based Electricity Theft Detection Using Autoencoder and GAN Models

This project studies anomaly-based electricity theft detection from SGCC daily
smart-meter consumption histories. The planned workflow compares temporal
autoencoder and GAN-based approaches while preserving customer traceability and
preventing preprocessing leakage.

## Repository structure

    data/raw/                  Immutable source data
    data/interim/              Validated Stage 1 outputs
    data/processed/            Customer-level Stage 2 split outputs
    data/windows/              Reproducible Stage 3 window manifests
    docs/                      Master technical decision record
    src/preprocessing/         Reproducible preprocessing stages
    models/preprocessing/      Fitted train-only preprocessing artifacts
    notebooks/                 Future exploratory work
    results/                   Future evaluation outputs
    dashboard/                 Future presentation layer

## Pipeline status

- Stage 1 — Cleaning: complete
- Stage 2 — Customer split and scaling: complete
- Stage 3 — Configurable temporal windowing: implemented; stride 30 executed
- Windowing does not perform class balancing or model training

Stage 3 creates reference-only manifests under data/windows/stride_30/. Each row
identifies a 30-position window by customer, split, and feature indices while
retaining missingness metadata. The authoritative customer assignments remain
in data/processed/split_manifest.parquet, and the training-fitted scaler remains
at models/preprocessing/scaler.joblib. See
[the preprocessing decision log](docs/preprocessing_decisions.md) for the full
methodology and stage boundaries.

Files under data/raw/ are immutable source data and must never be overwritten.

Run the implemented stages from the repository root:

    python -m src.preprocessing.clean_data
    python -m src.preprocessing.split_and_scale
    python -m src.preprocessing.generate_windows --stride 30

Planned comparative window runs, not yet executed:

    python -m src.preprocessing.generate_windows --stride 15
    python -m src.preprocessing.generate_windows --stride 7

