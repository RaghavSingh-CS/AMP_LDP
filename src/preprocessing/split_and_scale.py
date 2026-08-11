"""Stage 2: customer-level stratified splitting and train-only scaling.

Inputs are the immutable Stage 1 Parquet artifacts. This module creates
customer-disjoint train, validation, and test datasets; splits the missingness
mask with the same assignments; fits one feature-wise StandardScaler on
training consumption only; and persists the split manifest and scaler.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "data set.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
CLEANED_PATH = INTERIM_DIR / "cleaned_data.parquet"
MASK_PATH = INTERIM_DIR / "missing_mask.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSING_MODEL_DIR = PROJECT_ROOT / "models" / "preprocessing"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
TRAIN_MASK_PATH = PROCESSED_DIR / "train_mask.parquet"
VAL_MASK_PATH = PROCESSED_DIR / "val_mask.parquet"
TEST_MASK_PATH = PROCESSED_DIR / "test_mask.parquet"
MANIFEST_PATH = PROCESSED_DIR / "split_manifest.parquet"
SCALER_PATH = PREPROCESSING_MODEL_DIR / "scaler.joblib"

ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
SPLIT_COLUMN = "split"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
TEMP_FRACTION = 0.30
VALIDATION_SHARE_OF_TEMP = 0.50


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes() -> dict[Path, str]:
    """Hash raw and Stage 1 inputs so Stage 2 can prove they were not changed."""
    paths = (RAW_PATH, CLEANED_PATH, MASK_PATH)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Stage 2 inputs are missing: {missing}")
    return {path: file_sha256(path) for path in paths}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cleaned customer table and its Stage 1 missingness mask."""
    cleaned = pd.read_parquet(CLEANED_PATH, engine="pyarrow")
    mask = pd.read_parquet(MASK_PATH, engine="pyarrow")
    return cleaned, mask


def identify_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return chronological date features after validating their headers."""
    columns = list(columns)
    required = {ID_COLUMN, LABEL_COLUMN}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Cleaned data is missing required columns: {sorted(missing)}")

    features = [column for column in columns if column not in required]
    parsed = pd.to_datetime(features, format="%m/%d/%Y", errors="coerce")
    invalid = [name for name, date in zip(features, parsed) if pd.isna(date)]
    if invalid:
        raise ValueError(f"Non-date consumption feature columns found: {invalid}")
    if not parsed.is_unique:
        raise ValueError("Duplicate parsed consumption dates were found")
    if not parsed.is_monotonic_increasing:
        raise ValueError("Consumption feature columns are not chronological")
    return features


def validate_and_align_inputs(
    cleaned: pd.DataFrame,
    mask: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Validate Stage 1 invariants and align mask rows to cleaned customer order."""
    if cleaned[ID_COLUMN].isna().any() or not cleaned[ID_COLUMN].is_unique:
        raise ValueError("Cleaned CONS_NO values must be non-missing and unique")
    if mask[ID_COLUMN].isna().any() or not mask[ID_COLUMN].is_unique:
        raise ValueError("Mask CONS_NO values must be non-missing and unique")
    if set(cleaned[LABEL_COLUMN].unique()) != {0, 1}:
        raise ValueError("FLAG must contain exactly the binary values {0, 1}")
    if list(mask.columns) != [ID_COLUMN, *features]:
        raise ValueError("Mask feature columns do not exactly match cleaned features")

    cleaned_ids = set(cleaned[ID_COLUMN])
    mask_ids = set(mask[ID_COLUMN])
    if cleaned_ids != mask_ids:
        raise ValueError(
            "Cleaned and mask customer sets differ: "
            f"cleaned-only={len(cleaned_ids - mask_ids)}, "
            f"mask-only={len(mask_ids - cleaned_ids)}"
        )

    aligned_mask = (
        mask.set_index(ID_COLUMN, drop=False)
        .loc[cleaned[ID_COLUMN].to_numpy()]
        .reset_index(drop=True)
    )
    if not cleaned[ID_COLUMN].reset_index(drop=True).equals(
        aligned_mask[ID_COLUMN].reset_index(drop=True)
    ):
        raise ValueError("Mask could not be aligned to cleaned customer ordering")

    consumption = cleaned[features].to_numpy(copy=False)
    if not np.isfinite(consumption).all():
        raise ValueError("Cleaned consumption contains NaN or infinite values")
    mask_values = aligned_mask[features].to_numpy(copy=False)
    if not np.isin(mask_values, [0, 1]).all():
        raise ValueError("Missingness mask contains values outside {0, 1}")
    if not all(dtype == np.dtype("uint8") for dtype in aligned_mask[features].dtypes):
        raise TypeError("All mask features must be uint8")
    return aligned_mask


def create_stratified_split(cleaned: pd.DataFrame) -> dict[str, np.ndarray]:
    """Create deterministic 70/15/15 customer indices stratified by FLAG."""
    all_indices = np.arange(len(cleaned))
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=TEMP_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=cleaned[LABEL_COLUMN].to_numpy(),
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=VALIDATION_SHARE_OF_TEMP,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=cleaned.iloc[temp_indices][LABEL_COLUMN].to_numpy(),
    )
    return {
        "train": np.asarray(train_indices),
        "val": np.asarray(val_indices),
        "test": np.asarray(test_indices),
    }


def build_split_manifest(
    cleaned: pd.DataFrame, split_indices: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Build the authoritative one-row-per-customer split assignment."""
    parts: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        part = cleaned.iloc[split_indices[split_name]][
            [ID_COLUMN, LABEL_COLUMN]
        ].copy()
        part[SPLIT_COLUMN] = split_name
        parts.append(part)
    manifest = pd.concat(parts, ignore_index=True)
    manifest[LABEL_COLUMN] = manifest[LABEL_COLUMN].astype("int8")
    manifest[SPLIT_COLUMN] = manifest[SPLIT_COLUMN].astype("string")
    return manifest


def split_data_and_mask(
    cleaned: pd.DataFrame,
    aligned_mask: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Apply the exact same customer-index assignments to data and mask."""
    data_splits: dict[str, pd.DataFrame] = {}
    mask_splits: dict[str, pd.DataFrame] = {}
    for split_name, indices in split_indices.items():
        data_split = cleaned.iloc[indices].reset_index(drop=True).copy()
        mask_split = aligned_mask.iloc[indices].reset_index(drop=True).copy()
        if not data_split[ID_COLUMN].equals(mask_split[ID_COLUMN]):
            raise ValueError(f"{split_name} data and mask customer order differs")
        data_splits[split_name] = data_split
        mask_splits[split_name] = mask_split
    return data_splits, mask_splits


def fit_scaler(
    training_data: pd.DataFrame, features: list[str]
) -> StandardScaler:
    """Fit feature-wise statistics using training consumption and nothing else."""
    scaler = StandardScaler(copy=True)
    scaler.fit(training_data[features])
    if scaler.n_features_in_ != len(features):
        raise ValueError("Scaler feature count differs from consumption feature count")
    if not np.array_equal(scaler.feature_names_in_, np.asarray(features, dtype=object)):
        raise ValueError("Scaler saw columns other than the ordered consumption features")
    return scaler


def transform_split(
    data: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Transform one split with the train-fitted scaler and restore float32."""
    scaled_values = scaler.transform(data[features]).astype(np.float32, copy=False)
    if not np.isfinite(scaled_values).all():
        raise ValueError("Scaled consumption contains NaN or infinite values")
    scaled_features = pd.DataFrame(scaled_values, columns=features)
    output = pd.concat(
        [
            data[[ID_COLUMN]].reset_index(drop=True),
            data[[LABEL_COLUMN]].reset_index(drop=True).astype("int8"),
            scaled_features,
        ],
        axis=1,
    )
    return output


def validate_split_integrity(
    cleaned: pd.DataFrame,
    scaled_splits: dict[str, pd.DataFrame],
    mask_splits: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    features: list[str],
) -> dict[str, int]:
    """Validate customer exclusivity, manifest completeness, alignment, and types."""
    expected_names = {"train", "val", "test"}
    if set(scaled_splits) != expected_names or set(mask_splits) != expected_names:
        raise ValueError("Expected exactly train, val, and test splits")

    id_sets = {
        name: set(frame[ID_COLUMN]) for name, frame in scaled_splits.items()
    }
    overlaps = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Customer leakage detected: {overlaps}")
    if sum(len(frame) for frame in scaled_splits.values()) != len(cleaned):
        raise ValueError("Split row counts do not sum to the cleaned dataset")

    if len(manifest) != len(cleaned) or not manifest[ID_COLUMN].is_unique:
        raise ValueError("Manifest must contain every customer exactly once")
    if set(manifest[ID_COLUMN]) != set(cleaned[ID_COLUMN]):
        raise ValueError("Manifest customer set differs from cleaned data")
    if set(manifest[SPLIT_COLUMN].unique()) != expected_names:
        raise ValueError("Manifest contains an invalid split label")

    for name in ("train", "val", "test"):
        data = scaled_splits[name]
        mask = mask_splits[name]
        if data[ID_COLUMN].isna().any() or not data[ID_COLUMN].is_unique:
            raise ValueError(f"{name} data IDs must be non-missing and unique")
        if mask[ID_COLUMN].isna().any() or not mask[ID_COLUMN].is_unique:
            raise ValueError(f"{name} mask IDs must be non-missing and unique")
        if not data[ID_COLUMN].equals(mask[ID_COLUMN]):
            raise ValueError(f"{name} data and mask IDs are not aligned")
        if list(data.columns) != [ID_COLUMN, LABEL_COLUMN, *features]:
            raise ValueError(f"{name} data feature order changed")
        if list(mask.columns) != [ID_COLUMN, *features]:
            raise ValueError(f"{name} mask feature order changed")
        if not all(dtype == np.dtype("float32") for dtype in data[features].dtypes):
            raise TypeError(f"{name} scaled features must be float32")
        if not all(dtype == np.dtype("uint8") for dtype in mask[features].dtypes):
            raise TypeError(f"{name} mask features must be uint8")
        if data[features].isna().any().any() or mask[features].isna().any().any():
            raise ValueError(f"{name} output contains missing values")
        if not np.isfinite(data[features].to_numpy(copy=False)).all():
            raise ValueError(f"{name} scaled output contains infinite values")
        if not np.isin(mask[features].to_numpy(copy=False), [0, 1]).all():
            raise ValueError(f"{name} mask is not binary")
    return overlaps


def save_outputs(
    scaled_splits: dict[str, pd.DataFrame],
    mask_splits: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    scaler: StandardScaler,
) -> dict[str, Path]:
    """Atomically persist processed tables, aligned masks, manifest, and scaler."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    table_outputs = {
        "train": (scaled_splits["train"], TRAIN_PATH),
        "val": (scaled_splits["val"], VAL_PATH),
        "test": (scaled_splits["test"], TEST_PATH),
        "train_mask": (mask_splits["train"], TRAIN_MASK_PATH),
        "val_mask": (mask_splits["val"], VAL_MASK_PATH),
        "test_mask": (mask_splits["test"], TEST_MASK_PATH),
        "manifest": (manifest, MANIFEST_PATH),
    }
    temporary_paths: list[Path] = []
    try:
        for frame, destination in table_outputs.values():
            temporary = destination.with_name(destination.name + ".tmp")
            frame.to_parquet(
                temporary, engine="pyarrow", compression="zstd", index=False
            )
            temporary_paths.append(temporary)
        scaler_temporary = SCALER_PATH.with_name(SCALER_PATH.name + ".tmp")
        joblib.dump(scaler, scaler_temporary, compress=3)
        temporary_paths.append(scaler_temporary)

        for _, destination in table_outputs.values():
            destination.with_name(destination.name + ".tmp").replace(destination)
        scaler_temporary.replace(SCALER_PATH)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()

    paths = {name: path for name, (_, path) in table_outputs.items()}
    paths["scaler"] = SCALER_PATH
    return paths


def class_summary(frame: pd.DataFrame) -> tuple[int, int, float]:
    """Return normal count, theft count, and theft percentage."""
    counts = frame[LABEL_COLUMN].value_counts()
    normal = int(counts.get(0, 0))
    theft = int(counts.get(1, 0))
    theft_percentage = 100.0 * theft / len(frame)
    return normal, theft, theft_percentage


def scaling_summary(
    scaled_splits: dict[str, pd.DataFrame],
    features: list[str],
    scaler: StandardScaler,
) -> dict[str, object]:
    """Summarize feature-wise train centering and scale plus held-out ranges."""
    train_values = scaled_splits["train"][features].to_numpy(dtype=np.float64)
    feature_means = train_values.mean(axis=0)
    feature_stds = train_values.std(axis=0, ddof=0)
    zero_variance = np.flatnonzero(scaler.var_ == 0.0)
    nonconstant = scaler.var_ != 0.0
    summary: dict[str, object] = {
        "zero_variance_count": int(len(zero_variance)),
        "zero_variance_features": [features[index] for index in zero_variance],
        "train_feature_mean_min": float(feature_means.min()),
        "train_feature_mean_max": float(feature_means.max()),
        "train_max_abs_feature_mean": float(np.abs(feature_means).max()),
        "train_nonconstant_std_min": (
            float(feature_stds[nonconstant].min()) if nonconstant.any() else None
        ),
        "train_nonconstant_std_max": (
            float(feature_stds[nonconstant].max()) if nonconstant.any() else None
        ),
        "train_nonconstant_max_abs_std_error": (
            float(np.abs(feature_stds[nonconstant] - 1.0).max())
            if nonconstant.any()
            else None
        ),
    }
    for name in ("val", "test"):
        values = scaled_splits[name][features].to_numpy(copy=False)
        summary[f"{name}_min"] = float(values.min())
        summary[f"{name}_max"] = float(values.max())
    return summary


def main() -> None:
    """Execute Stage 2 and print a concise, auditable validation report."""
    hashes_before = input_hashes()
    cleaned, mask = load_inputs()
    features = identify_feature_columns(cleaned.columns)
    aligned_mask = validate_and_align_inputs(cleaned, mask, features)

    split_indices = create_stratified_split(cleaned)
    manifest = build_split_manifest(cleaned, split_indices)
    data_splits, mask_splits = split_data_and_mask(
        cleaned, aligned_mask, split_indices
    )

    scaler = fit_scaler(data_splits["train"], features)
    scaled_splits = {
        name: transform_split(frame, features, scaler)
        for name, frame in data_splits.items()
    }
    overlaps = validate_split_integrity(
        cleaned, scaled_splits, mask_splits, manifest, features
    )
    scale_report = scaling_summary(scaled_splits, features, scaler)
    output_paths = save_outputs(scaled_splits, mask_splits, manifest, scaler)

    hashes_after = input_hashes()
    if hashes_before != hashes_after:
        changed = [
            str(path)
            for path in hashes_before
            if hashes_before[path] != hashes_after[path]
        ]
        raise RuntimeError(f"Raw or Stage 1 input artifacts changed: {changed}")

    print("\nSGCC Stage 2 split-and-scale report")
    print(f"Total customers: {len(cleaned):,}")
    for name in ("train", "val", "test"):
        frame = scaled_splits[name]
        normal, theft, theft_pct = class_summary(frame)
        print(
            f"{name}: {len(frame):,} ({100 * len(frame) / len(cleaned):.4f}%) | "
            f"normal={normal:,}, theft={theft:,}, theft={theft_pct:.4f}%"
        )
    print(
        "Customer overlaps: "
        f"train/val={overlaps['train_val']}, "
        f"train/test={overlaps['train_test']}, "
        f"val/test={overlaps['val_test']}"
    )
    print("Mask IDs aligned: train=True, val=True, test=True")
    print(f"Consumption features: {len(features):,}")
    print("Feature ordering chronological: True")
    print("Scaler fitted on training consumption features only: True")
    print(f"Scaler features: {scaler.n_features_in_:,}")
    print(
        "Train per-feature means: "
        f"min={scale_report['train_feature_mean_min']:.8f}, "
        f"max={scale_report['train_feature_mean_max']:.8f}, "
        f"max_abs={scale_report['train_max_abs_feature_mean']:.8f}"
    )
    print(
        "Train non-constant feature stds: "
        f"min={scale_report['train_nonconstant_std_min']}, "
        f"max={scale_report['train_nonconstant_std_max']}, "
        f"max_abs_error_from_1="
        f"{scale_report['train_nonconstant_max_abs_std_error']}"
    )
    print(
        f"Zero-variance training features: "
        f"{scale_report['zero_variance_count']} "
        f"{scale_report['zero_variance_features']}"
    )
    print(
        f"Validation scaled range: "
        f"[{scale_report['val_min']:.6f}, {scale_report['val_max']:.6f}]"
    )
    print(
        f"Test scaled range: "
        f"[{scale_report['test_min']:.6f}, {scale_report['test_max']:.6f}]"
    )
    print("NaN values: 0; infinite values: 0")
    print("Dtypes: consumption=float32, FLAG=int8, mask=uint8")
    for name, path in output_paths.items():
        print(f"{name} output: {path} ({path.stat().st_size / 1024**2:.2f} MiB)")
    print("Raw and Stage 1 input hashes unchanged: True")


if __name__ == "__main__":
    main()
