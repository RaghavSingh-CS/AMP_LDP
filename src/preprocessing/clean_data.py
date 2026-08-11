"""First-stage cleaning for the SGCC smart-meter dataset.

This module preserves the raw CSV and produces an imputed, log-transformed
customer table plus a separate missingness mask. It deliberately performs no
splitting, scaling, sampling, windowing, or model-related work.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "data set.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
CLEANED_PATH = INTERIM_DIR / "cleaned_data.parquet"
MASK_PATH = INTERIM_DIR / "missing_mask.parquet"

ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
DROP_DATE = pd.Timestamp("2014-10-03")
INSERT_DATE = pd.Timestamp("2016-09-18")
START_DATE = pd.Timestamp("2014-01-01")
END_DATE = pd.Timestamp("2016-10-31")
EXPECTED_EMPTY_CUSTOMERS = 5


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_date_columns(columns: Iterable[str]) -> dict[pd.Timestamp, str]:
    """Identify date headers and map parsed dates back to their source names."""
    columns = list(columns)
    required = {ID_COLUMN, LABEL_COLUMN}
    missing_metadata = required.difference(columns)
    if missing_metadata:
        raise ValueError(f"Missing required columns: {sorted(missing_metadata)}")

    date_map: dict[pd.Timestamp, str] = {}
    unexpected: list[str] = []
    for column in columns:
        if column in required:
            continue
        parsed = pd.to_datetime(column, format="%m/%d/%Y", errors="coerce")
        if pd.isna(parsed):
            unexpected.append(column)
            continue
        timestamp = pd.Timestamp(parsed)
        if timestamp in date_map:
            raise ValueError(
                f"Duplicate date headers after parsing: {date_map[timestamp]!r} "
                f"and {column!r}"
            )
        date_map[timestamp] = column

    if unexpected:
        raise ValueError(f"Unexpected non-date feature columns: {unexpected}")
    if not date_map:
        raise ValueError("No date-based consumption columns were found")
    return date_map


def load_data(path: Path) -> tuple[pd.DataFrame, dict[pd.Timestamp, str]]:
    """Load the CSV with memory-conscious types after inspecting its header."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    header = pd.read_csv(path, nrows=0)
    date_map = identify_date_columns(header.columns)
    dtype_map: dict[str, str] = {name: "float32" for name in date_map.values()}
    dtype_map.update({ID_COLUMN: "string", LABEL_COLUMN: "int8"})
    data = pd.read_csv(path, dtype=dtype_map, low_memory=False)
    return data, date_map


def clean_calendar(
    data: pd.DataFrame, date_map: dict[pd.Timestamp, str]
) -> tuple[pd.DataFrame, list[str]]:
    """Drop the unusable date, insert the absent date, and sort chronologically."""
    if DROP_DATE not in date_map:
        raise ValueError(f"Required drop date {DROP_DATE.date()} is not present")
    if INSERT_DATE in date_map:
        raise ValueError(f"Expected absent date {INSERT_DATE.date()} is already present")

    cleaned = data.drop(columns=[date_map[DROP_DATE]]).copy()
    inserted_name = "9/18/2016"
    cleaned[inserted_name] = np.float32(np.nan)

    revised_map = {
        date: name for date, name in date_map.items() if date != DROP_DATE
    }
    revised_map[INSERT_DATE] = inserted_name

    expected_dates = set(pd.date_range(START_DATE, END_DATE, freq="D"))
    expected_dates.remove(DROP_DATE)
    actual_dates = set(revised_map)
    if actual_dates != expected_dates:
        missing = sorted(expected_dates - actual_dates)
        extra = sorted(actual_dates - expected_dates)
        raise ValueError(
            "Calendar differs from the required daily range. "
            f"Missing: {[d.date() for d in missing]}; "
            f"extra: {[d.date() for d in extra]}"
        )

    ordered_dates = sorted(revised_map)
    date_columns = [revised_map[date] for date in ordered_dates]
    cleaned = cleaned[[ID_COLUMN, LABEL_COLUMN, *date_columns]]
    return cleaned, date_columns


def remove_empty_customers(
    data: pd.DataFrame, date_columns: list[str]
) -> tuple[pd.DataFrame, int, dict[int, int]]:
    """Remove customers with no observed consumption after calendar cleaning."""
    empty_mask = data[date_columns].isna().all(axis=1)
    removed = data.loc[empty_mask]
    removed_count = int(empty_mask.sum())
    removed_flags = {
        int(flag): int(count)
        for flag, count in removed[LABEL_COLUMN].value_counts().sort_index().items()
    }
    if removed_count != EXPECTED_EMPTY_CUSTOMERS:
        raise ValueError(
            "Completely empty customer count differs from the inspected dataset: "
            f"expected {EXPECTED_EMPTY_CUSTOMERS}, found {removed_count}."
        )
    return data.loc[~empty_mask].reset_index(drop=True), removed_count, removed_flags


def create_missing_mask(
    data: pd.DataFrame, date_columns: list[str]
) -> pd.DataFrame:
    """Create a compact mask: 1 for missing and 0 for originally observed."""
    values = data[date_columns].isna().astype("uint8")
    mask = pd.concat([data[[ID_COLUMN]].copy(), values], axis=1)
    if not mask["9/18/2016"].eq(1).all():
        raise AssertionError("Inserted 9/18/2016 must be missing for every customer")
    return mask


def impute_customer_series(consumption: pd.DataFrame) -> pd.DataFrame:
    """Interpolate within each row, then fill leading/trailing gaps by proximity."""
    imputed = consumption.interpolate(
        method="linear", axis=1, limit_area="inside"
    )
    imputed = imputed.ffill(axis=1).bfill(axis=1)
    if imputed.isna().any().any():
        raise ValueError("Missing consumption values remain after imputation")
    return imputed


def transform_consumption(imputed: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p and return float32 features without clipping outliers."""
    values = imputed.to_numpy(dtype=np.float32, copy=False)
    if (values < 0).any():
        raise ValueError("Negative consumption values detected before transformation")
    transformed_values = np.log1p(values).astype(np.float32, copy=False)
    if not np.isfinite(transformed_values).all():
        raise ValueError("Non-finite values detected after log1p transformation")
    return pd.DataFrame(
        transformed_values, columns=imputed.columns, index=imputed.index
    )


def validate_output(
    cleaned: pd.DataFrame,
    mask: pd.DataFrame,
    date_columns: list[str],
) -> None:
    """Validate identifiers, labels, dates, data types, alignment, and values."""
    if cleaned[ID_COLUMN].isna().any() or not cleaned[ID_COLUMN].is_unique:
        raise ValueError("CONS_NO must remain non-missing and unique")
    if not set(cleaned[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("FLAG contains values outside {0, 1}")
    if cleaned[date_columns].isna().any().any():
        raise ValueError("Cleaned consumption contains missing values")
    if not all(dtype == np.dtype("float32") for dtype in cleaned[date_columns].dtypes):
        raise TypeError("All transformed consumption columns must be float32")
    if not cleaned[ID_COLUMN].reset_index(drop=True).equals(
        mask[ID_COLUMN].reset_index(drop=True)
    ):
        raise ValueError("Cleaned data and mask customer ordering do not match")
    if list(mask.columns) != [ID_COLUMN, *date_columns]:
        raise ValueError("Missing-mask columns do not match the cleaned calendar")
    mask_values = mask[date_columns].to_numpy(copy=False)
    if not np.isin(mask_values, [0, 1]).all():
        raise ValueError("Missing mask must contain only 0 and 1")


def save_outputs(cleaned: pd.DataFrame, mask: pd.DataFrame) -> None:
    """Write Parquet outputs atomically so failed runs do not leave partial files."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for frame, destination in ((cleaned, CLEANED_PATH), (mask, MASK_PATH)):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        temporary.replace(destination)


def main() -> None:
    """Run the complete first-stage cleaning pipeline and print its audit report."""
    raw_hash_before = file_sha256(RAW_PATH)
    data, original_date_map = load_data(RAW_PATH)
    original_rows = len(data)
    original_consumption_columns = len(original_date_map)

    data, date_columns = clean_calendar(data, original_date_map)
    data, removed_count, removed_flags = remove_empty_customers(data, date_columns)

    raw_consumption = data[date_columns]
    if (raw_consumption < 0).any().any():
        raise ValueError("Negative values found in raw consumption data")
    missing_before = int(raw_consumption.isna().to_numpy().sum())
    zero_count = int(raw_consumption.eq(0).to_numpy().sum())
    mask = create_missing_mask(data, date_columns)

    imputed = impute_customer_series(raw_consumption)
    missing_after = int(imputed.isna().to_numpy().sum())
    transformed = transform_consumption(imputed)

    cleaned = pd.concat(
        [
            data[[ID_COLUMN]].copy(),
            data[[LABEL_COLUMN]].astype("int8"),
            transformed,
        ],
        axis=1,
    )
    validate_output(cleaned, mask, date_columns)
    save_outputs(cleaned, mask)

    raw_hash_after = file_sha256(RAW_PATH)
    if raw_hash_before != raw_hash_after:
        raise RuntimeError("Raw input changed during preprocessing")

    final_flags = {
        int(flag): int(count)
        for flag, count in cleaned[LABEL_COLUMN].value_counts().sort_index().items()
    }
    cleaned_size = CLEANED_PATH.stat().st_size
    mask_size = MASK_PATH.stat().st_size

    print("\nSGCC first-stage preprocessing report")
    print(f"Original rows: {original_rows:,}")
    print(f"Final rows: {len(cleaned):,}")
    print(f"Completely empty customers removed: {removed_count:,}")
    print(f"Removed FLAG distribution: {removed_flags}")
    print(f"Original consumption columns: {original_consumption_columns:,}")
    print(f"Final consumption columns: {len(date_columns):,}")
    print(f"10/3/2014 removed: {'10/3/2014' not in date_columns}")
    print(f"9/18/2016 inserted: {'9/18/2016' in date_columns}")
    print(f"Missing consumption cells before imputation: {missing_before:,}")
    print(f"Missing consumption cells after imputation: {missing_after:,}")
    print(f"Zero values before transformation: {zero_count:,}")
    print(f"Minimum transformed value: {transformed.to_numpy().min():.6f}")
    print(f"Maximum transformed value: {transformed.to_numpy().max():.6f}")
    print(f"Final FLAG distribution: {final_flags}")
    print(f"Cleaned output: {CLEANED_PATH} ({cleaned_size / 1024**2:.2f} MiB)")
    print(f"Mask output: {MASK_PATH} ({mask_size / 1024**2:.2f} MiB)")
    print(f"Raw SHA-256 unchanged: {raw_hash_after}")


if __name__ == "__main__":
    main()
