"""Train the Stage 4 normal-only baseline Autoencoder.

The module implements the complete real-run pipeline but is intentionally not
executed as part of repository setup. The test split is never loaded here.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
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
from src.models.autoencoder import (
    INPUT_DIM,
    LATENT_DIM,
    Autoencoder,
    architecture_config,
    count_trainable_parameters,
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


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Derive deterministic NumPy/Python seeds for optional DataLoader workers."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: WindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> DataLoader[Tensor]:
    """Create a deterministic DataLoader with Windows-safe defaults."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def train_epoch(
    model: Autoencoder,
    loader: DataLoader[Tensor],
    optimizer: Adam,
    criterion: nn.MSELoss,
    device: torch.device,
) -> float:
    """Run one optimization epoch and return mean per-window batch MSE."""
    model.train()
    weighted_loss = 0.0
    examples = 0
    for inputs in loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, inputs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered")
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        weighted_loss += float(loss.detach().item()) * batch_size
        examples += batch_size
    if examples == 0:
        raise RuntimeError("Training DataLoader produced no examples")
    return weighted_loss / examples


@torch.no_grad()
def validation_loss(
    model: Autoencoder,
    loader: DataLoader[Tensor],
    criterion: nn.MSELoss,
    device: torch.device,
) -> float:
    """Evaluate normal validation windows without gradient tracking."""
    model.eval()
    weighted_loss = 0.0
    examples = 0
    for inputs in loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        reconstructions = model(inputs)
        loss = criterion(reconstructions, inputs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss encountered")
        batch_size = inputs.shape[0]
        weighted_loss += float(loss.item()) * batch_size
        examples += batch_size
    if examples == 0:
        raise RuntimeError("Validation DataLoader produced no examples")
    return weighted_loss / examples


def checkpoint_payload(
    model: Autoencoder,
    optimizer: Adam,
    args: argparse.Namespace,
    best_epoch: int,
    best_validation_loss: float,
) -> dict[str, object]:
    """Build a portable state-dictionary checkpoint with full baseline metadata."""
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture_configuration": architecture_config(),
        "input_dimension": INPUT_DIM,
        "latent_dimension": LATENT_DIM,
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


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    """Write a checkpoint without exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(payload: dict[str, object], path: Path) -> None:
    """Atomically write indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a CSV without its pandas index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write compressed reconstruction-error metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def model_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the complete real-run model and preprocessing configuration."""
    return {
        "stage": "Stage 4 — Baseline Autoencoder",
        "model": architecture_config(),
        "trainable_parameters": 1362,
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
        "input_policy": {
            "included": ["30 StandardScaled consumption values"],
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
        "training_population": (
            f"usable stride-{args.stride} train windows with FLAG == 0 only"
        ),
        "early_stopping_population": (
            f"usable stride-{args.stride} validation windows with FLAG == 0 only"
        ),
        "scaling_assumption": (
            "Stage 1 log1p consumption transformed by the Stage 2 "
            "training-fitted feature-wise StandardScaler"
        ),
        "test_set_policy": "not loaded or used during Stage 4",
        "determinism_note": (
            "Seeds and deterministic modes are set, but exact reproducibility "
            "can still depend on device, backend, and PyTorch operations."
        ),
    }


def save_training_curve(history: pd.DataFrame, path: Path) -> None:
    """Plot training and normal-validation MSE by epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_mse"], label="Train MSE")
    axis.plot(
        history["epoch"], history["val_normal_mse"], label="Normal validation MSE"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Baseline Autoencoder Training Curve")
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
    model: Autoencoder,
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate per-window MSE in deterministic manifest order."""
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
    for inputs in loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        reconstructions = model(inputs)
        scores = torch.mean((reconstructions - inputs) ** 2, dim=1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite reconstruction MSE encountered")
        score_batches.append(scores.cpu().numpy())
    scores = np.concatenate(score_batches).astype(np.float32, copy=False)
    metadata = dataset.metadata
    if len(scores) != len(metadata):
        raise RuntimeError("Reconstruction errors and manifest metadata are misaligned")
    metadata["reconstruction_mse"] = scores
    return metadata.loc[
        :, ["window_id", "CONS_NO", "FLAG", "reconstruction_mse", "missing_fraction"]
    ]


def summarize_errors(errors: pd.DataFrame) -> dict[str, object]:
    """Summarize reconstruction MSE separately for normal and theft labels."""
    summaries: dict[str, object] = {}
    medians: dict[int, float] = {}
    for flag in (0, 1):
        values = errors.loc[errors["FLAG"].eq(flag), "reconstruction_mse"].to_numpy(
            dtype=np.float64
        )
        if values.size == 0:
            raise ValueError(f"No FLAG={flag} reconstruction errors available")
        median = float(np.median(values))
        medians[flag] = median
        summaries[f"FLAG_{flag}"] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": median,
            "standard_deviation": float(np.std(values, ddof=1)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
        }
    summaries["theft_to_normal_median_ratio"] = (
        medians[1] / medians[0] if medians[0] != 0 else None
    )
    summaries["theft_minus_normal_median"] = medians[1] - medians[0]
    return summaries


def save_error_distribution(errors: pd.DataFrame, path: Path) -> None:
    """Plot log-transformed MSE for visibility without altering saved scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure, axis = plt.subplots(figsize=(8, 5))
    for flag, label in ((0, "Normal validation"), (1, "Theft-labelled validation")):
        values = errors.loc[errors["FLAG"].eq(flag), "reconstruction_mse"].to_numpy()
        transformed = np.log10(values.astype(np.float64) + np.finfo(np.float32).eps)
        axis.hist(transformed, bins=80, density=True, alpha=0.45, label=label)
    axis.set_xlabel("log10(reconstruction MSE + float32 epsilon)")
    axis.set_ylabel("Density")
    axis.set_title("Validation Reconstruction-Error Distribution")
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


def load_best_model(checkpoint_path: Path, device: torch.device) -> Autoencoder:
    """Reload the best state-dictionary checkpoint for error extraction."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Autoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_training(args: argparse.Namespace) -> None:
    """Execute the real Stage 4 training and validation-error workflow."""
    started = time.perf_counter()
    set_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if args.num_workers > 0 and os.name == "nt":
        print("Warning: num_workers > 0 can duplicate Dataset memory on Windows.")

    model_dir = PROJECT_ROOT / "models" / "autoencoder"
    results_dir = PROJECT_ROOT / "results" / "autoencoder" / f"stride_{args.stride}"
    checkpoint_path = model_dir / f"ae_stride{args.stride}_baseline.pt"
    config_path = model_dir / f"ae_stride{args.stride}_baseline_config.json"
    history_path = results_dir / "training_history.csv"
    summary_path = results_dir / "training_summary.json"
    curve_path = results_dir / "training_curve.png"
    errors_path = results_dir / "val_reconstruction_errors.parquet"
    distribution_path = results_dir / "validation_error_distribution.png"

    config = model_config(args)
    save_json(config, config_path)

    train_store = ProcessedSplitData.load("train")
    validation_store = ProcessedSplitData.load("val")
    train_dataset = WindowDataset(
        "train", 0, stride=args.stride, store=train_store, include_metadata=False
    )
    validation_normal_dataset = WindowDataset(
        "val", 0, stride=args.stride, store=validation_store, include_metadata=True
    )
    if train_dataset.flag != 0 or validation_normal_dataset.flag != 0:
        raise RuntimeError("Training and early-stopping populations must be FLAG == 0")
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

    model = Autoencoder().to(device)
    if count_trainable_parameters(model) != 1362:
        raise RuntimeError("Baseline architecture parameter count changed")
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
        improved = val_mse < best_validation_mse - args.min_delta
        if improved:
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
    error_summary = summarize_errors(errors)
    save_parquet(errors, errors_path)
    save_error_distribution(errors, distribution_path)

    runtime_seconds = time.perf_counter() - started
    training_summary: dict[str, object] = {
        "stage": "Stage 4 — Baseline Autoencoder",
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
        "runtime_seconds": runtime_seconds,
        "reconstruction_error_summary": error_summary,
        "anomaly_threshold_selected": False,
        "test_set_used": False,
    }
    save_json(training_summary, summary_path)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training results: {results_dir}")
    print("Anomaly threshold selected: False")
    print("Test set used: False")


def positive_int(value: str) -> int:
    """Argparse converter for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """Argparse converter for positive floating-point values."""
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def nonnegative_float(value: str) -> float:
    """Argparse converter for non-negative floating-point values."""
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative finite number")
    return parsed


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse the fixed baseline plus safe resource controls."""
    parser = argparse.ArgumentParser(
        description="Train the normal-only baseline Autoencoder for one prepared stride."
    )
    parser.add_argument("--stride", type=int, choices=(30, 15), default=DEFAULT_STRIDE)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=positive_int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=positive_float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument("--patience", type=positive_int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=nonnegative_float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num-workers", type=int, choices=range(0, 65), default=DEFAULT_NUM_WORKERS
    )
    return parser.parse_args(arguments)


def main() -> None:
    """CLI entry point for the user-executed real training run."""
    run_training(parse_args())


if __name__ == "__main__":
    main()
