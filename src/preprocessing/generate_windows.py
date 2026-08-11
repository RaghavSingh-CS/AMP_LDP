"""Stage 3: configurable, memory-conscious temporal window manifests.

The generator processes one Stage 2 split and one stride at a time. It stores
window references and missingness metadata rather than duplicating dense
30-value consumption or mask sequences.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
WINDOWS_DIR = DATA_DIR / "windows"
SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing" / "scaler.joblib"

ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
SPLIT_COLUMN = "split"
WINDOW_LENGTH = 30
STRIDE_OPTIONS = (30, 15, 7)
DEFAULT_STRIDE = 30
SPLIT_NAMES = ("train", "val", "test")

DATA_PATHS = {
    split: PROCESSED_DIR / f"{split}.parquet" for split in SPLIT_NAMES
}
MASK_PATHS = {
    split: PROCESSED_DIR / f"{split}_mask.parquet" for split in SPLIT_NAMES
}
SPLIT_MANIFEST_PATH = PROCESSED_DIR / "split_manifest.parquet"

IMMUTABLE_INPUT_PATHS = (
    DATA_DIR / "raw" / "data set.csv",
    DATA_DIR / "interim" / "cleaned_data.parquet",
    DATA_DIR / "interim" / "missing_mask.parquet",
    DATA_PATHS["train"],
    DATA_PATHS["val"],
    DATA_PATHS["test"],
    MASK_PATHS["train"],
    MASK_PATHS["val"],
    MASK_PATHS["test"],
    SPLIT_MANIFEST_PATH,
    SCALER_PATH,
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes() -> dict[str, str]:
    """Hash every Stage 1/2 artifact that Stage 3 must leave unchanged."""
    missing = [str(path) for path in IMMUTABLE_INPUT_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Stage 3 inputs are missing: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in IMMUTABLE_INPUT_PATHS
    }


def load_split_manifest() -> pd.DataFrame:
    """Load and validate the authoritative Stage 2 customer assignments."""
    manifest = pd.read_parquet(SPLIT_MANIFEST_PATH, engine="pyarrow")
    if list(manifest.columns) != [ID_COLUMN, LABEL_COLUMN, SPLIT_COLUMN]:
        raise ValueError("Stage 2 split manifest has an unexpected schema")
    if manifest[ID_COLUMN].isna().any() or not manifest[ID_COLUMN].is_unique:
        raise ValueError("Stage 2 split manifest IDs must be non-missing and unique")
    if set(manifest[LABEL_COLUMN].unique()) != {0, 1}:
        raise ValueError("Stage 2 split manifest FLAG must be binary")
    if set(manifest[SPLIT_COLUMN].unique()) != set(SPLIT_NAMES):
        raise ValueError("Stage 2 split manifest contains unexpected split names")
    return manifest


def load_split_data(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one processed split and its aligned mask."""
    data = pd.read_parquet(DATA_PATHS[split_name], engine="pyarrow")
    mask = pd.read_parquet(MASK_PATHS[split_name], engine="pyarrow")
    return data, mask


def identify_feature_columns(columns: Iterable[str]) -> list[str]:
    """Identify and validate chronological date-named consumption features."""
    columns = list(columns)
    required = {ID_COLUMN, LABEL_COLUMN}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Processed data is missing required columns: {missing}")
    features = [column for column in columns if column not in required]
    dates = pd.to_datetime(features, format="%m/%d/%Y", errors="coerce")
    invalid = [feature for feature, date in zip(features, dates) if pd.isna(date)]
    if invalid:
        raise ValueError(f"Non-date consumption feature columns found: {invalid}")
    if not dates.is_unique:
        raise ValueError("Duplicate chronological feature dates found")
    if not dates.is_monotonic_increasing:
        raise ValueError("Consumption feature columns are not chronological")
    return features


def validate_alignment(
    split_name: str,
    data: pd.DataFrame,
    mask: pd.DataFrame,
    stage2_manifest: pd.DataFrame,
    features: list[str],
) -> None:
    """Validate data, mask, and authoritative Stage 2 assignment alignment."""
    if list(data.columns) != [ID_COLUMN, LABEL_COLUMN, *features]:
        raise ValueError(f"{split_name} processed feature order changed")
    if list(mask.columns) != [ID_COLUMN, *features]:
        raise ValueError(f"{split_name} mask feature order changed")
    if len(data) != len(mask) or not data[ID_COLUMN].equals(mask[ID_COLUMN]):
        raise ValueError(f"{split_name} data and mask rows are not exactly aligned")
    if data[ID_COLUMN].isna().any() or not data[ID_COLUMN].is_unique:
        raise ValueError(f"{split_name} data IDs must be non-missing and unique")
    if mask[ID_COLUMN].isna().any() or not mask[ID_COLUMN].is_unique:
        raise ValueError(f"{split_name} mask IDs must be non-missing and unique")
    if not set(data[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{split_name} FLAG contains non-binary values")

    assigned = stage2_manifest.loc[
        stage2_manifest[SPLIT_COLUMN].eq(split_name), [ID_COLUMN, LABEL_COLUMN]
    ].reset_index(drop=True)
    if not data[ID_COLUMN].reset_index(drop=True).equals(assigned[ID_COLUMN]):
        raise ValueError(
            f"{split_name} customer order differs from the Stage 2 manifest"
        )
    if not np.array_equal(
        data[LABEL_COLUMN].to_numpy(), assigned[LABEL_COLUMN].to_numpy()
    ):
        raise ValueError(f"{split_name} labels differ from the Stage 2 manifest")

    values = data[features].to_numpy(copy=False)
    mask_values = mask[features].to_numpy(copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{split_name} processed consumption is not finite")
    if not np.isin(mask_values, [0, 1]).all():
        raise ValueError(f"{split_name} mask is not binary")
    if not all(dtype == np.dtype("uint8") for dtype in mask[features].dtypes):
        raise TypeError(f"{split_name} mask features must remain uint8")


def calculate_window_positions(
    number_of_features: int, stride: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return inclusive start/end indices and trailing unused feature count."""
    if number_of_features < WINDOW_LENGTH:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            number_of_features,
        )
    starts = np.arange(
        0,
        number_of_features - WINDOW_LENGTH + 1,
        stride,
        dtype=np.int32,
    )
    ends = starts + WINDOW_LENGTH - 1
    trailing_unused = int(number_of_features - (int(ends[-1]) + 1))
    return starts, ends, trailing_unused


def generate_manifest_for_split(
    split_name: str,
    data: pd.DataFrame,
    mask: pd.DataFrame,
    features: list[str],
    starts: np.ndarray,
    ends: np.ndarray,
) -> pd.DataFrame:
    """Generate reference-only windows and per-window missingness metadata."""
    customer_count = len(data)
    windows_per_customer = len(starts)
    if windows_per_customer == 0:
        raise ValueError("No complete windows can be generated")

    mask_values = mask[features].to_numpy(dtype=np.uint8, copy=False)
    missing_by_window = np.column_stack(
        [
            mask_values[:, start : end + 1].sum(axis=1, dtype=np.uint16)
            for start, end in zip(starts, ends)
        ]
    )
    missing_counts = missing_by_window.reshape(-1).astype(np.uint8, copy=False)
    missing_fractions = (missing_counts.astype(np.float32) / WINDOW_LENGTH).astype(
        np.float32, copy=False
    )

    customer_ids = data[ID_COLUMN].astype("string").to_numpy()
    labels = data[LABEL_COLUMN].to_numpy(dtype=np.int8, copy=False)
    window_numbers = np.tile(
        np.arange(windows_per_customer, dtype=np.int16), customer_count
    )
    repeated_ids = np.repeat(customer_ids, windows_per_customer)
    repeated_labels = np.repeat(labels, windows_per_customer)
    tiled_starts = np.tile(starts.astype(np.int16), customer_count)
    tiled_ends = np.tile(ends.astype(np.int16), customer_count)
    start_dates = np.tile(
        np.asarray([features[index] for index in starts], dtype=object),
        customer_count,
    )
    end_dates = np.tile(
        np.asarray([features[index] for index in ends], dtype=object),
        customer_count,
    )
    window_ids = [
        f"{split_name}_{customer_id}_{window_number:03d}"
        for customer_id in customer_ids
        for window_number in range(windows_per_customer)
    ]

    manifest = pd.DataFrame(
        {
            "window_id": pd.Series(window_ids, dtype="string"),
            ID_COLUMN: pd.Series(repeated_ids, dtype="string"),
            LABEL_COLUMN: repeated_labels,
            SPLIT_COLUMN: pd.Series(
                np.repeat(split_name, len(window_ids)), dtype="string"
            ),
            "window_number_within_customer": window_numbers,
            "start_feature_index": tiled_starts,
            "end_feature_index": tiled_ends,
            "start_date": pd.Series(start_dates, dtype="string"),
            "end_date": pd.Series(end_dates, dtype="string"),
            "missing_count": missing_counts,
            "missing_fraction": missing_fractions,
        }
    )
    return manifest


def validate_manifest(
    split_name: str,
    source_data: pd.DataFrame,
    manifest: pd.DataFrame,
    features: list[str],
    starts: np.ndarray,
    ends: np.ndarray,
    stride: int,
) -> dict[str, int]:
    """Validate deterministic IDs, positions, dates, labels, and row counts."""
    windows_per_customer = len(starts)
    expected_rows = len(source_data) * windows_per_customer
    invalid_windows = 0

    if len(manifest) != expected_rows:
        raise ValueError(
            f"{split_name} manifest has {len(manifest)} rows; "
            f"expected {expected_rows}"
        )
    if manifest["window_id"].isna().any() or not manifest["window_id"].is_unique:
        raise ValueError(f"{split_name} window IDs must be non-missing and unique")
    if set(manifest[SPLIT_COLUMN].unique()) != {split_name}:
        raise ValueError(f"{split_name} manifest contains another split label")
    if not set(manifest[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{split_name} window labels are not binary")

    actual_starts = manifest["start_feature_index"].to_numpy().reshape(
        len(source_data), windows_per_customer
    )
    actual_ends = manifest["end_feature_index"].to_numpy().reshape(
        len(source_data), windows_per_customer
    )
    expected_start_matrix = np.broadcast_to(starts, actual_starts.shape)
    expected_end_matrix = np.broadcast_to(ends, actual_ends.shape)
    if not np.array_equal(actual_starts, expected_start_matrix):
        raise ValueError(f"{split_name} start indices do not advance by {stride}")
    if not np.array_equal(actual_ends, expected_end_matrix):
        raise ValueError(f"{split_name} end indices are incorrect")
    if not np.all(actual_ends - actual_starts + 1 == WINDOW_LENGTH):
        invalid_windows += int(
            np.count_nonzero(actual_ends - actual_starts + 1 != WINDOW_LENGTH)
        )
    if actual_starts.min() < 0 or actual_ends.max() >= len(features):
        raise ValueError(f"{split_name} manifest contains out-of-range positions")

    expected_ids = np.repeat(
        source_data[ID_COLUMN].astype("string").to_numpy(), windows_per_customer
    )
    expected_labels = np.repeat(
        source_data[LABEL_COLUMN].to_numpy(dtype=np.int8), windows_per_customer
    )
    if not np.array_equal(manifest[ID_COLUMN].to_numpy(), expected_ids):
        raise ValueError(f"{split_name} windows crossed customer boundaries")
    if not np.array_equal(manifest[LABEL_COLUMN].to_numpy(), expected_labels):
        raise ValueError(f"{split_name} window labels differ from customer labels")

    expected_start_dates = np.tile(
        np.asarray([features[index] for index in starts], dtype=object),
        len(source_data),
    )
    expected_end_dates = np.tile(
        np.asarray([features[index] for index in ends], dtype=object),
        len(source_data),
    )
    if not np.array_equal(manifest["start_date"].to_numpy(), expected_start_dates):
        raise ValueError(f"{split_name} start dates do not match feature indices")
    if not np.array_equal(manifest["end_date"].to_numpy(), expected_end_dates):
        raise ValueError(f"{split_name} end dates do not match feature indices")

    missing_counts = manifest["missing_count"].to_numpy()
    missing_fractions = manifest["missing_fraction"].to_numpy()
    if (missing_counts < 0).any() or (missing_counts > WINDOW_LENGTH).any():
        raise ValueError(f"{split_name} missing_count is outside [0, 30]")
    if (missing_fractions < 0).any() or (missing_fractions > 1).any():
        raise ValueError(f"{split_name} missing_fraction is outside [0, 1]")
    if not np.allclose(
        missing_fractions,
        missing_counts.astype(np.float32) / WINDOW_LENGTH,
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError(f"{split_name} missing fractions are inconsistent")
    if invalid_windows:
        raise ValueError(f"{split_name} has {invalid_windows} invalid windows")
    return {
        "expected_rows": expected_rows,
        "invalid_windows": invalid_windows,
        "duplicate_window_ids": int(manifest["window_id"].duplicated().sum()),
    }


def summarize_missingness(fractions: np.ndarray) -> dict[str, float | int]:
    """Summarize original/constructed missingness represented by each window."""
    total = len(fractions)

    def count_and_percentage(condition: np.ndarray) -> dict[str, float | int]:
        count = int(np.count_nonzero(condition))
        return {"count": count, "percentage": 100.0 * count / total}

    return {
        "average": float(np.mean(fractions)),
        "median": float(np.median(fractions)),
        "p90": float(np.quantile(fractions, 0.90)),
        "p95": float(np.quantile(fractions, 0.95)),
        "p99": float(np.quantile(fractions, 0.99)),
        "maximum": float(np.max(fractions)),
        "zero_missing": count_and_percentage(fractions == 0),
        "over_25_percent": count_and_percentage(fractions > 0.25),
        "over_50_percent": count_and_percentage(fractions > 0.50),
        "over_75_percent": count_and_percentage(fractions > 0.75),
        "all_missing": count_and_percentage(fractions == 1.0),
    }


def save_manifest(manifest: pd.DataFrame, path: Path) -> None:
    """Atomically save one split manifest using compressed Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        manifest.to_parquet(
            temporary, engine="pyarrow", compression="zstd", index=False
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(payload: dict[str, object], path: Path) -> None:
    """Atomically save the configuration and validation summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    """Parse the supported comparative stride configuration."""
    parser = argparse.ArgumentParser(
        description="Generate reference-only temporal window manifests."
    )
    parser.add_argument(
        "--stride",
        type=int,
        choices=STRIDE_OPTIONS,
        default=DEFAULT_STRIDE,
        help="Window stride in days; supported values: 30, 15, 7.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and validate manifests for one configured stride."""
    args = parse_args()
    stride = args.stride
    hashes_before = input_hashes()
    stage2_manifest = load_split_manifest()
    output_dir = WINDOWS_DIR / f"stride_{stride}"

    reference_features: list[str] | None = None
    starts: np.ndarray | None = None
    ends: np.ndarray | None = None
    trailing_unused: int | None = None
    split_reports: dict[str, dict[str, object]] = {}
    all_missing_fractions: list[np.ndarray] = []
    customer_sets: dict[str, set[str]] = {}

    for split_name in SPLIT_NAMES:
        data, mask = load_split_data(split_name)
        features = identify_feature_columns(data.columns)
        if reference_features is None:
            reference_features = features
            starts, ends, trailing_unused = calculate_window_positions(
                len(features), stride
            )
        elif features != reference_features:
            raise ValueError(f"{split_name} feature columns differ from train")
        assert starts is not None and ends is not None

        validate_alignment(
            split_name, data, mask, stage2_manifest, reference_features
        )
        manifest = generate_manifest_for_split(
            split_name, data, mask, reference_features, starts, ends
        )
        integrity = validate_manifest(
            split_name,
            data,
            manifest,
            reference_features,
            starts,
            ends,
            stride,
        )
        fractions = manifest["missing_fraction"].to_numpy(dtype=np.float32, copy=True)
        all_missing_fractions.append(fractions)
        missing_summary = summarize_missingness(fractions)

        output_path = output_dir / f"{split_name}_windows.parquet"
        save_manifest(manifest, output_path)
        normal_customers = int(data[LABEL_COLUMN].eq(0).sum())
        theft_customers = int(data[LABEL_COLUMN].eq(1).sum())
        windows_per_customer = len(starts)
        split_reports[split_name] = {
            "customers": len(data),
            "total_windows": len(manifest),
            "normal_customer_windows": normal_customers * windows_per_customer,
            "theft_customer_windows": theft_customers * windows_per_customer,
            "missingness": missing_summary,
            "duplicate_window_ids": integrity["duplicate_window_ids"],
            "invalid_windows": integrity["invalid_windows"],
            "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            "output_size_bytes": output_path.stat().st_size,
        }
        customer_sets[split_name] = set(data[ID_COLUMN].astype(str))

        del data, mask, manifest, fractions
        gc.collect()

    assert reference_features is not None
    assert starts is not None and ends is not None and trailing_unused is not None
    overlaps = {
        "train_val": len(customer_sets["train"] & customer_sets["val"]),
        "train_test": len(customer_sets["train"] & customer_sets["test"]),
        "val_test": len(customer_sets["val"] & customer_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Stage 2 customer leakage detected: {overlaps}")

    combined_fractions = np.concatenate(all_missing_fractions)
    overall_missingness = summarize_missingness(combined_fractions)
    expected_total_windows = len(stage2_manifest) * len(starts)
    actual_total_windows = sum(
        int(report["total_windows"]) for report in split_reports.values()
    )
    if actual_total_windows != expected_total_windows:
        raise ValueError(
            f"Total manifest rows {actual_total_windows} do not match "
            f"expected {expected_total_windows}"
        )

    hashes_after = input_hashes()
    if hashes_before != hashes_after:
        changed = [
            path
            for path in hashes_before
            if hashes_before[path] != hashes_after[path]
        ]
        raise RuntimeError(f"Stage 1/2 artifacts changed during Stage 3: {changed}")

    summary: dict[str, object] = {
        "configuration": {
            "window_length": WINDOW_LENGTH,
            "stride": stride,
            "stride_options": list(STRIDE_OPTIONS),
            "chronological_features": len(reference_features),
            "windows_per_customer": len(starts),
            "trailing_unused_positions": trailing_unused,
            "end_feature_index_is_inclusive": True,
            "dense_windows_materialized": False,
        },
        "splits": split_reports,
        "overall": {
            "customers": len(stage2_manifest),
            "total_windows": actual_total_windows,
            "missingness": overall_missingness,
        },
        "integrity": {
            "customer_overlaps": overlaps,
            "duplicate_window_ids": sum(
                int(report["duplicate_window_ids"])
                for report in split_reports.values()
            ),
            "invalid_windows": sum(
                int(report["invalid_windows"]) for report in split_reports.values()
            ),
            "stage1_stage2_hashes_unchanged": True,
            "input_sha256": hashes_after,
        },
    }
    summary_path = output_dir / "window_summary.json"
    save_json(summary, summary_path)

    print("\nSGCC Stage 3 temporal-window report")
    print(f"WINDOW_LENGTH: {WINDOW_LENGTH}")
    print(f"STRIDE: {stride}")
    print(f"Chronological features: {len(reference_features):,}")
    print(f"Windows per customer: {len(starts):,}")
    print(f"Trailing unused positions per customer: {trailing_unused:,}")
    for split_name in SPLIT_NAMES:
        report = split_reports[split_name]
        print(
            f"{split_name}: customers={report['customers']:,}, "
            f"windows={report['total_windows']:,}, "
            f"normal-customer windows={report['normal_customer_windows']:,}, "
            f"theft-customer windows={report['theft_customer_windows']:,}"
        )
    print(f"Total windows: {actual_total_windows:,}")
    print(
        "Missingness: "
        f"mean={overall_missingness['average']:.6f}, "
        f"median={overall_missingness['median']:.6f}, "
        f"p90={overall_missingness['p90']:.6f}, "
        f"p95={overall_missingness['p95']:.6f}, "
        f"p99={overall_missingness['p99']:.6f}, "
        f"max={overall_missingness['maximum']:.6f}"
    )
    for label, key in (
        ("0% missing", "zero_missing"),
        (">25% missing", "over_25_percent"),
        (">50% missing", "over_50_percent"),
        (">75% missing", "over_75_percent"),
        ("100% missing", "all_missing"),
    ):
        metric = overall_missingness[key]
        print(
            f"{label}: {metric['count']:,} windows "
            f"({metric['percentage']:.4f}%)"
        )
    print(
        "Integrity: "
        f"duplicate window IDs=0, invalid windows=0, "
        f"customer overlaps={overlaps}"
    )
    for split_name in SPLIT_NAMES:
        path = output_dir / f"{split_name}_windows.parquet"
        print(f"{split_name} manifest: {path} ({path.stat().st_size / 1024**2:.2f} MiB)")
    print(
        f"summary: {summary_path} ({summary_path.stat().st_size / 1024**2:.4f} MiB)"
    )
    print("Stage 1 and Stage 2 artifact hashes unchanged: True")


if __name__ == "__main__":
    main()
