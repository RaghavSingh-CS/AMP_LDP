"""Shared, leakage-safe utilities for the Stage 13 supervised diagnostic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "supervised" / "stage_13"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised" / "stage_13"
ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
WINDOW_LENGTH = 30
STRIDE = 30
RANDOM_STATE = 42
SPLIT_NAMES = ("train", "val", "test")


@dataclass(slots=True)
class WindowMatrix:
    """Dense model input plus audit-only metadata kept outside the feature matrix."""

    X: np.ndarray
    y: np.ndarray
    customer_ids: np.ndarray
    window_ids: np.ndarray


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the whole file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def load_window_matrix(split: str) -> WindowMatrix:
    """Materialize existing usable windows without introducing new features."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    processed_path = PROJECT_ROOT / "data" / "processed" / f"{split}.parquet"
    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "windows"
        / f"stride_{STRIDE}"
        / "usable"
        / f"{split}_windows.parquet"
    )
    processed = pd.read_parquet(processed_path, engine="pyarrow")
    manifest = pd.read_parquet(
        manifest_path,
        columns=[
            "window_id",
            ID_COLUMN,
            LABEL_COLUMN,
            "start_feature_index",
            "end_feature_index",
            "missing_fraction",
        ],
        engine="pyarrow",
    )

    feature_columns = [
        column for column in processed.columns if column not in {ID_COLUMN, LABEL_COLUMN}
    ]
    parsed_dates = pd.to_datetime(feature_columns, format="%m/%d/%Y", errors="coerce")
    if len(feature_columns) != 1034 or parsed_dates.isna().any():
        raise ValueError("Processed split does not contain the established 1,034 dates")
    if not parsed_dates.is_unique or not parsed_dates.is_monotonic_increasing:
        raise ValueError("Processed consumption dates are not uniquely chronological")
    if processed[ID_COLUMN].isna().any() or not processed[ID_COLUMN].is_unique:
        raise ValueError(f"{split} processed customer IDs are invalid")
    if manifest["window_id"].duplicated().any():
        raise ValueError(f"{split} usable window IDs are not unique")
    if not np.isin(manifest[LABEL_COLUMN], [0, 1]).all():
        raise ValueError(f"{split} window labels are not binary")
    if not manifest["missing_fraction"].lt(1.0).all():
        raise ValueError(f"{split} contains a window outside the Stage 3C rule")

    processed_ids = pd.Index(processed[ID_COLUMN].astype(str), name=ID_COLUMN)
    manifest_ids = pd.Index(manifest[ID_COLUMN].astype(str), name=ID_COLUMN)
    customer_rows = processed_ids.get_indexer(manifest_ids)
    if np.any(customer_rows < 0):
        raise ValueError(f"{split} manifest references an unknown customer")
    customer_labels = processed[LABEL_COLUMN].to_numpy(dtype=np.int8, copy=False)
    labels = manifest[LABEL_COLUMN].to_numpy(dtype=np.int8, copy=True)
    if not np.array_equal(customer_labels[customer_rows], labels):
        raise ValueError(f"{split} customer and propagated window labels disagree")

    starts = manifest["start_feature_index"].to_numpy(dtype=np.int32, copy=True)
    ends = manifest["end_feature_index"].to_numpy(dtype=np.int32, copy=False)
    if np.any(starts < 0) or np.any(ends >= len(feature_columns)):
        raise ValueError(f"{split} contains out-of-range window indices")
    if not np.all(ends - starts + 1 == WINDOW_LENGTH):
        raise ValueError(f"{split} contains a non-{WINDOW_LENGTH}-value window")

    values = np.ascontiguousarray(
        processed[feature_columns].to_numpy(dtype=np.float32, copy=False)
    )
    X = np.empty((len(manifest), WINDOW_LENGTH), dtype=np.float32)
    offsets = np.arange(WINDOW_LENGTH, dtype=np.int32)
    block_size = 100_000
    for first in range(0, len(manifest), block_size):
        last = min(first + block_size, len(manifest))
        rows = customer_rows[first:last, None]
        columns = starts[first:last, None] + offsets[None, :]
        X[first:last] = values[rows, columns]
    if not np.isfinite(X).all():
        raise ValueError(f"{split} model matrix contains non-finite values")

    return WindowMatrix(
        X=X,
        y=labels,
        customer_ids=manifest_ids.to_numpy(dtype=str),
        window_ids=manifest["window_id"].astype(str).to_numpy(),
    )


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose the validation threshold with maximum positive-class F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        raise ValueError("No classification threshold is available")
    numerators = 2.0 * precision[:-1] * recall[:-1]
    denominators = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators),
        where=denominators > 0,
    )
    best = np.flatnonzero(f1_values == np.nanmax(f1_values))
    # A deterministic highest-threshold tie break avoids gratuitous positives.
    return float(thresholds[int(best[-1])])


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Compute discrimination and operating-point metrics for FLAG=1."""
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "count": int(len(y_true)),
        "normal_count": int(np.sum(y_true == 0)),
        "theft_count": int(np.sum(y_true == 1)),
        "positive_prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "threshold_rule": "predicted_probability >= threshold",
        "ROC_AUC": float(roc_auc_score(y_true, scores)),
        "PR_AUC_average_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def aggregate_customer_mean(
    y_true: np.ndarray, scores: np.ndarray, customer_ids: np.ndarray
) -> pd.DataFrame:
    """Aggregate window probabilities by arithmetic mean for each customer."""
    frame = pd.DataFrame(
        {ID_COLUMN: customer_ids, LABEL_COLUMN: y_true, "mean_probability": scores}
    )
    label_counts = frame.groupby(ID_COLUMN, sort=False)[LABEL_COLUMN].nunique()
    if not label_counts.eq(1).all():
        raise ValueError("A customer has inconsistent propagated window labels")
    customers = (
        frame.groupby(ID_COLUMN, sort=False, as_index=False)
        .agg(
            FLAG=(LABEL_COLUMN, "first"),
            mean_probability=("mean_probability", "mean"),
            usable_window_count=("mean_probability", "size"),
        )
        .reset_index(drop=True)
    )
    # Prevent floating summation order from creating fictitious rank signal when
    # a reference classifier assigns exactly the same score to every window.
    if np.ptp(scores) == 0.0:
        customers["mean_probability"] = float(scores[0])
    return customers


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
    temporary.replace(path)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
