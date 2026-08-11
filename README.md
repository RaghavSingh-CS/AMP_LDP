# Anomaly-Based Electricity Theft Detection Using Autoencoder and GAN Models

This project studies anomaly-based electricity theft detection from SGCC daily
smart-meter consumption histories. The planned workflow will compare temporal
autoencoder and GAN-based approaches while preserving customer-level
traceability and preventing preprocessing leakage.

## Repository structure

    data/raw/          Immutable source data
    data/interim/      Validated first-stage cleaning outputs
    docs/              Technical decision records
    src/preprocessing/ Reproducible preprocessing code
    models/            Future trained-model artifacts
    notebooks/         Future exploratory work
    results/           Future evaluation outputs
    dashboard/         Future presentation layer

## Current status

The first cleaning stage repairs the daily calendar, removes completely empty
customers, preserves a missingness mask, imputes each customer independently,
and applies a log1p transformation. It intentionally does not split, scale,
balance, window, or train the data. See
[the preprocessing decision log](docs/preprocessing_decisions.md) for the full
rationale and stage boundaries.

Files under data/raw/ are immutable source data and must never be overwritten.

Run the cleaning stage from the repository root:

    python -m src.preprocessing.clean_data
