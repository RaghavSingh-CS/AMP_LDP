"""Train the normal-only stride-30 LSTM Autoencoder baseline.

The user runs this module manually. It never loads the test split and never
selects an anomaly threshold.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.window_dataset import ProcessedSplitData, WindowDataset
from src.models.lstm_autoencoder import (
    DECODER_HIDDEN_SIZE,
    ENCODER_HIDDEN_SIZE,
    INPUT_SIZE,
    LATENT_DIM,
    NUM_LAYERS,
    SEQUENCE_LENGTH,
    LSTMAutoencoder,
    architecture_config,
    count_trainable_parameters,
)
from src.training.train_autoencoder import (
    atomic_torch_save,
    make_loader,
    nonnegative_float,
    positive_float,
    positive_int,
    save_csv,
    save_json,
    save_parquet,
    set_reproducibility,
    summarize_errors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_LENGTH = 30
DEFAULT_STRIDE = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 1e-5
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 0
EXPECTED_PARAMETER_COUNT = 3177


def as_lstm_input(inputs: Tensor) -> Tensor:
    """Convert a model-agnostic `(B, 30)` batch into `(B, 30, 1)`."""
    if inputs.ndim != 2 or inputs.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected Dataset batch shape (batch, {SEQUENCE_LENGTH}); "
            f"received {tuple(inputs.shape)}"
        )
    return inputs.unsqueeze(-1)


def train_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader[Tensor],
    optimizer: Adam,
    criterion: nn.MSELoss,
    device: torch.device,
) -> float:
    """Run one normal-only optimization epoch."""
    model.train()
    weighted_loss = 0.0
    examples = 0
    for flat_inputs in loader:
        inputs = as_lstm_input(
            flat_inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        )
        optimizer.zero_grad(set_to_none=True)
        reconstructions = model(inputs)
        if reconstructions.shape != inputs.shape:
            raise RuntimeError("LSTM input and reconstruction shapes differ")
        loss = criterion(reconstructions, inputs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LSTM training loss encountered")
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        weighted_loss += float(loss.detach().item()) * batch_size
        examples += batch_size
    if examples == 0:
        raise RuntimeError("LSTM training DataLoader produced no examples")
    return weighted_loss / examples


@torch.no_grad()
def validation_loss(
    model: LSTMAutoencoder,
    loader: DataLoader[Tensor],
    criterion: nn.MSELoss,
    device: torch.device,
) -> float:
    """Calculate normal-validation MSE for early stopping."""
    model.eval()
    weighted_loss = 0.0
    examples = 0
    for flat_inputs in loader:
        inputs = as_lstm_input(
            flat_inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        )
        reconstructions = model(inputs)
        if reconstructions.shape != inputs.shape:
            raise RuntimeError("LSTM validation input and output shapes differ")
        loss = criterion(reconstructions, inputs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LSTM validation loss encountered")
        batch_size = inputs.shape[0]
        weighted_loss += float(loss.item()) * batch_size
        examples += batch_size
    if examples == 0:
        raise RuntimeError("LSTM validation DataLoader produced no examples")
    return weighted_loss / examples


def checkpoint_payload(
    model: LSTMAutoencoder,
    optimizer: Adam,
    args: argparse.Namespace,
    best_epoch: int,
    best_validation_loss: float,
) -> dict[str, object]:
    """Build the portable LSTM state-dictionary checkpoint."""
    architecture = architecture_config()
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": architecture["type"],
        "architecture_configuration": architecture,
        "input_size": INPUT_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "decoder_hidden_size": DECODER_HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "latent_dim": LATENT_DIM,
        "window_length": WINDOW_LENGTH,
        "stride": args.stride,
        "loss": "MSELoss",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


def model_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the full recurrent architecture, data, and training policy."""
    return {
        "stage": "Stage 11 — LSTM Autoencoder",
        "model": architecture_config(),
        "trainable_parameters": EXPECTED_PARAMETER_COUNT,
        "input_shape": [SEQUENCE_LENGTH, INPUT_SIZE],
        "output_shape": [SEQUENCE_LENGTH, INPUT_SIZE],
        "window_length": WINDOW_LENGTH,
        "stride": args.stride,
        "loss": "MSELoss",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "early_stopping": {
            "monitor": "normal_validation_mse",
            "patience": args.patience,
            "minimum_delta": args.min_delta,
        },
        "seed": args.seed,
        "num_workers": args.num_workers,
        "decoder_policy": {
            "input": "latent vector repeated over 30 timesteps",
            "initial_hidden": "ReLU Linear 8 -> 16",
            "initial_cell": "zeros",
            "teacher_forcing": False,
            "output_projection": "Linear 16 -> 1 without output activation",
        },
        "input_policy": {
            "included": ["one StandardScaled consumption value per timestep"],
            "excluded": [
                "CONS_NO",
                "FLAG",
                "missingness mask",
                "missing_count",
                "missing_fraction",
                "dates",
                "window_id",
            ],
        },
        "training_population": "usable stride-30 train windows with FLAG == 0 only",
        "early_stopping_population": (
            "usable stride-30 validation windows with FLAG == 0 only"
        ),
        "scaling_assumption": (
            "Stage 1 log1p consumption transformed by the Stage 2 "
            "training-fitted feature-wise StandardScaler"
        ),
        "future_threshold_policy": "normal_validation_95th_percentile",
        "test_set_policy": "not loaded or used during LSTM training",
        "controlled_ablation_note": (
            "The stride-30 data and training protocol remain fixed; the model "
            "architecture changes to recurrent sequence memory. Latent dimension "
            "8 matches the CNN bottleneck and is not tuned."
        ),
    }


def load_expected_counts(stride: int) -> dict[str, int]:
    """Derive normal train/validation counts from the filtering artifact."""
    path = (
        PROJECT_ROOT
        / "data"
        / "windows"
        / f"stride_{stride}"
        / "usable"
        / "filter_summary.json"
    )
    with path.open("r", encoding="utf-8") as source:
        summary = json.load(source)
    return {
        "train": int(summary["splits"]["train"]["retained_normal_windows"]),
        "val": int(summary["splits"]["val"]["retained_normal_windows"]),
    }


def save_training_curve(history: pd.DataFrame, path: Path) -> None:
    """Plot LSTM training and normal-validation MSE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_mse"], label="Train MSE")
    axis.plot(
        history["epoch"],
        history["val_normal_mse"],
        label="Normal validation MSE",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("LSTM Autoencoder Training Curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


@torch.no_grad()
def reconstruction_errors(
    model: LSTMAutoencoder,
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate per-window LSTM reconstruction MSE in manifest order."""
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        num_workers=num_workers,
        seed=seed,
    )
    model.eval()
    score_batches: list[np.ndarray] = []
    for flat_inputs in loader:
        inputs = as_lstm_input(
            flat_inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        )
        reconstructions = model(inputs)
        scores = torch.mean((reconstructions - inputs) ** 2, dim=(1, 2))
        if not torch.isfinite(scores).all() or torch.any(scores < 0):
            raise FloatingPointError("Invalid LSTM reconstruction MSE encountered")
        score_batches.append(scores.cpu().numpy())
    scores = np.concatenate(score_batches).astype(np.float32, copy=False)
    metadata = dataset.metadata
    if len(scores) != len(metadata):
        raise RuntimeError("LSTM scores and manifest metadata are misaligned")
    metadata["reconstruction_mse"] = scores
    return metadata.loc[
        :,
        ["window_id", "CONS_NO", "FLAG", "reconstruction_mse", "missing_fraction"],
    ]


def save_error_distribution(errors: pd.DataFrame, path: Path) -> None:
    """Plot validation LSTM MSE without changing saved values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure, axis = plt.subplots(figsize=(8, 5))
    for flag, label in ((0, "Normal validation"), (1, "Theft-labelled validation")):
        values = errors.loc[errors["FLAG"].eq(flag), "reconstruction_mse"].to_numpy()
        transformed = np.log10(values.astype(np.float64) + np.finfo(np.float32).eps)
        axis.hist(transformed, bins=80, density=True, alpha=0.45, label=label)
    axis.set_xlabel("log10(reconstruction MSE + float32 epsilon)")
    axis.set_ylabel("Density")
    axis.set_title("LSTM Validation Reconstruction-Error Distribution")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    try:
        figure.savefig(temporary, dpi=150)
        temporary.replace(path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def load_best_model(checkpoint_path: Path, device: torch.device) -> LSTMAutoencoder:
    """Reload the best LSTM checkpoint for validation-error extraction."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["stride"] != 30 or checkpoint["window_length"] != WINDOW_LENGTH:
        raise ValueError("LSTM checkpoint window configuration is inconsistent")
    if checkpoint["architecture_configuration"] != architecture_config():
        raise ValueError("LSTM checkpoint architecture differs from source")
    model = LSTMAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def run_training(args: argparse.Namespace) -> None:
    """Execute the user-run LSTM workflow without loading test data."""
    started = time.perf_counter()
    set_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if args.num_workers > 0 and os.name == "nt":
        print("Warning: num_workers > 0 can duplicate Dataset memory on Windows.")

    model_dir = PROJECT_ROOT / "models" / "lstm_autoencoder"
    results_dir = PROJECT_ROOT / "results" / "lstm_autoencoder" / "stride_30"
    checkpoint_path = model_dir / "lstm_ae_stride30_baseline.pt"
    config_path = model_dir / "lstm_ae_stride30_baseline_config.json"
    history_path = results_dir / "training_history.csv"
    summary_path = results_dir / "training_summary.json"
    curve_path = results_dir / "training_curve.png"
    errors_path = results_dir / "val_reconstruction_errors.parquet"
    distribution_path = results_dir / "validation_error_distribution.png"

    save_json(model_config(args), config_path)
    expected = load_expected_counts(args.stride)
    train_store = ProcessedSplitData.load("train")
    validation_store = ProcessedSplitData.load("val")
    train_dataset = WindowDataset(
        "train", 0, stride=args.stride, store=train_store, include_metadata=False
    )
    validation_normal_dataset = WindowDataset(
        "val", 0, stride=args.stride, store=validation_store, include_metadata=True
    )
    if (
        len(train_dataset) != expected["train"]
        or len(validation_normal_dataset) != expected["val"]
    ):
        raise ValueError("LSTM Dataset counts disagree with Stage 3C")
    if train_dataset.flag != 0 or validation_normal_dataset.flag != 0:
        raise RuntimeError("LSTM training and early stopping must use FLAG == 0")
    print(f"Normal training windows: {len(train_dataset):,}")
    print(f"Normal validation windows: {len(validation_normal_dataset):,}")
    print(
        "Primary split-matrix memory: "
        f"{(train_store.memory_bytes + validation_store.memory_bytes) / 1024**2:.1f} MiB"
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    validation_loader = make_loader(
        validation_normal_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = LSTMAutoencoder().to(device)
    if count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("LSTM architecture parameter count changed")
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    history_rows: list[dict[str, float | int]] = []
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse = validation_loss(model, validation_loader, criterion, device)
        history_rows.append(
            {"epoch": epoch, "train_mse": train_mse, "val_normal_mse": val_mse}
        )
        if val_mse < best_validation_mse - args.min_delta:
            best_validation_mse = val_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                checkpoint_payload(
                    model, optimizer, args, best_epoch, best_validation_mse
                ),
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train MSE: {train_mse:.8f}")
        print(f"Val MSE: {val_mse:.8f}")
        print(f"Best Val MSE: {best_validation_mse:.8f}")
        if epochs_without_improvement >= args.patience:
            early_stopped = True
            print(f"Early stopping after epoch {epoch}.")
            break

    history = pd.DataFrame(
        history_rows, columns=["epoch", "train_mse", "val_normal_mse"]
    )
    save_csv(history, history_path)
    save_training_curve(history, curve_path)
    training_window_count = len(train_dataset)
    validation_normal_window_count = len(validation_normal_dataset)
    del train_loader, validation_loader, train_dataset, train_store
    gc.collect()

    best_model = load_best_model(checkpoint_path, device)
    normal_errors = reconstruction_errors(
        best_model,
        validation_normal_dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    validation_theft_dataset = WindowDataset(
        "val", 1, stride=args.stride, store=validation_store, include_metadata=True
    )
    theft_errors = reconstruction_errors(
        best_model,
        validation_theft_dataset,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    errors = pd.concat([normal_errors, theft_errors], ignore_index=True)
    save_parquet(errors, errors_path)
    save_error_distribution(errors, distribution_path)
    save_json(
        {
            "stage": "Stage 11 — LSTM Autoencoder",
            "device": str(device),
            "training_normal_window_count": training_window_count,
            "validation_normal_window_count": validation_normal_window_count,
            "validation_theft_window_count": len(validation_theft_dataset),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs_executed": len(history),
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation_mse,
            "early_stopping_occurred": early_stopped,
            "runtime_seconds": time.perf_counter() - started,
            "reconstruction_error_summary": summarize_errors(errors),
            "anomaly_threshold_selected": False,
            "test_set_used": False,
        },
        summary_path,
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training results: {results_dir}")
    print("Anomaly threshold selected: False")
    print("Test set used: False")


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse the fixed stride-30 LSTM baseline and safe resource controls."""
    parser = argparse.ArgumentParser(
        description="Train the normal-only stride-30 LSTM Autoencoder."
    )
    parser.add_argument("--stride", type=int, choices=(30,), default=DEFAULT_STRIDE)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=positive_int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=positive_float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument("--patience", type=positive_int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--min-delta", type=nonnegative_float, default=DEFAULT_MIN_DELTA
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num-workers", type=int, choices=range(0, 65), default=DEFAULT_NUM_WORKERS
    )
    return parser.parse_args(arguments)


def main() -> None:
    """CLI entry point for the user-executed full LSTM run."""
    run_training(parse_args())


if __name__ == "__main__":
    main()
