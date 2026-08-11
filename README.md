# Anomaly-Based Electricity Theft Detection Using Autoencoder and GAN Models

This project studies anomaly-based electricity theft detection from SGCC daily
smart-meter consumption histories. The planned workflow compares temporal
autoencoder and GAN-based approaches while preserving customer traceability and
preventing preprocessing leakage.

## Repository structure

    data/raw/                    Immutable source data
    data/interim/                Validated Stage 1 outputs
    data/processed/              Customer-level Stage 2 split outputs
    data/windows/                Reproducible Stage 3 window manifests
    data/windows/stride_30/usable/ Stage 3C baseline-eligible manifests
    data/windows/stride_15/usable/ Stage 7 overlap-ablation manifests
    results/missingness_analysis/ Stage 3B quality evidence and recommendation
    docs/                        Master technical decision record
    src/preprocessing/           Reproducible preprocessing stages
    src/analysis/                Reproducible descriptive analyses
    src/data/                    Lazy model-input datasets
    src/models/                  Neural-network architectures
    src/training/                Reproducible training entry points
    models/preprocessing/        Fitted train-only preprocessing artifacts

## Pipeline status

- Stage 1 — Cleaning: complete
- Stage 2 — Customer split and scaling: complete
- Stage 3 — Configurable temporal windowing: complete for stride 30
- Stage 3B — Missingness quality analysis: complete
- Stage 3C — Window quality filtering: complete for stride 30
- Stage 4 — Baseline Autoencoder: trained
- Stage 5 — Validation threshold: frozen
- Stage 6 — Frozen test evaluation: complete
- Stage 7 — Stride-15 Autoencoder ablation: preparation and manual training complete
- Stage 8 — Stride-15 threshold, frozen test, and controlled comparison: complete

Stage 3B compares candidate missingness thresholds without changing the
canonical manifests. See the
[missingness quality report](results/missingness_analysis/report.md) for the
window, customer, class, temporal-position, and consecutive-run findings.

Stage 3C adopts the minimum baseline eligibility rule
`missing_fraction < 1.0` (equivalently `missing_count < 30`): a window is
retained when at least one of its 30 source readings was observed. The filtered
manifests are exact subsets under `data/windows/stride_30/usable/`; the original
Stage 3 manifests remain canonical and unchanged. The baseline model input will
be the scaled consumption sequence only. The missingness mask,
`missing_fraction`, `FLAG`, and `CONS_NO` are excluded from model inputs.

Stage 4 implements a normal-only, fully connected Autoencoder with architecture
`30 → 16 → 8 → 4 → 8 → 16 → 30`. It uses usable stride-30 `FLAG=0` training
windows, normal-only validation MSE for early stopping, Adam at `0.001`, and
MSE loss. Training completed on CPU and saved the best checkpoint at
`models/autoencoder/ae_stride30_baseline.pt` without using the test set.

Stage 5 selects the baseline threshold from validation reconstruction errors
only. The adopted normal-validation 95th-percentile threshold is stored at
`models/autoencoder/ae_stride30_threshold.json`; detailed evidence is under
`results/autoencoder/stride_30/threshold_analysis/`.

Stage 6 applies that frozen threshold once to the reserved test split without
retraining or tuning. The final stride-30 Autoencoder baseline record is under
`results/autoencoder/stride_30/test_evaluation/`.

The stride-30 fully connected Autoencoder is the complete frozen baseline.
The stride-15 Autoencoder has completed normal-only training, validation-only
P95 threshold selection, and one-time frozen test evaluation under the same
methodology. The controlled comparison is at
`results/autoencoder/stride_comparison/stride_comparison_report.md`. Stride 15
is effectively equivalent and slightly worse at the window level; lower
reconstruction loss did not improve core anomaly discrimination. Stride 7 is
not recommended next and remains unexecuted.

Files under data/raw/ are immutable source data and must never be overwritten.

Run the implemented stages from the repository root:

    python -m src.preprocessing.clean_data
    python -m src.preprocessing.split_and_scale
    python -m src.preprocessing.generate_windows --stride 30
    python -m src.analysis.analyze_window_missingness
    python -m src.preprocessing.filter_windows --stride 30
    python -m src.training.train_autoencoder --stride 30
    python -m src.analysis.analyze_autoencoder_threshold
    python -m src.evaluation.evaluate_autoencoder_test

Stride-15 reproduction commands are:

    python -m src.training.train_autoencoder --stride 15

    python -m src.analysis.analyze_autoencoder_threshold --stride 15
    python -m src.evaluation.evaluate_autoencoder_test --stride 15
    python -m src.analysis.compare_autoencoder_strides
Stride 7 remains optional, ungenerated, and not recommended for this baseline.

