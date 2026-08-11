# Anomaly-Based Electricity Theft Detection Using Autoencoder and GAN Models

This project studies anomaly-based electricity theft detection from SGCC daily
smart-meter consumption histories. The planned workflow will compare temporal
autoencoder and GAN-based approaches while preserving customer-level
traceability and preventing preprocessing leakage.

## Repository structure

    data/raw/                  Immutable source data
    data/interim/              Validated Stage 1 outputs
    data/processed/            Customer-level Stage 2 split outputs
    docs/                      Master technical decision record
    src/preprocessing/         Reproducible Stage 1 and Stage 2 code
    models/preprocessing/      Fitted train-only preprocessing artifacts
    notebooks/                 Future exploratory work
    results/                   Future evaluation outputs
    dashboard/                 Future presentation layer

## Current status

Stage 1 cleaning is complete: the calendar was repaired, completely empty
customers were removed, the missingness mask was retained, values were imputed
customer-wise, and log1p was applied.

Stage 2 customer splitting and scaling is complete: customers were split
70/15/15 using stratification and random state 42, then one feature-wise
StandardScaler was fitted only on training consumption and reused for validation
and test data. No windowing, class balancing, or model training has been
performed.

Processed outputs are under data/processed/. The authoritative assignments are
in data/processed/split_manifest.parquet, and the fitted scaler is stored at
models/preprocessing/scaler.joblib. See
[the preprocessing decision log](docs/preprocessing_decisions.md) for the
methodology and stage boundaries.

Files under data/raw/ are immutable source data and must never be overwritten.

Run the stages from the repository root:

    python -m src.preprocessing.clean_data
    python -m src.preprocessing.split_and_scale
