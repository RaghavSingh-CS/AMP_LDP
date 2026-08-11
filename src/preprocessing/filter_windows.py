"""Stage 3C: retain supported-stride windows with observed readings.

This stage filters only manifest rows. It does not materialize sequences,
impute values, change labels, rebalance classes, or alter customer splits.
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
WINDOWS_DIR = DATA_DIR / "windows"
ANALYSIS_ROOT = PROJECT_ROOT / "results" / "missingness_analysis"

ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
SPLIT_COLUMN = "split"
WINDOW_LENGTH = 30
SUPPORTED_STRIDES = (30, 15)
DEFAULT_STRIDE = 30
SPLIT_NAMES = ("train", "val", "test")

PROTECTED_FIXED_PATHS = (
    DATA_DIR / "raw" / "data set.csv",
    DATA_DIR / "interim" / "cleaned_data.parquet",
    DATA_DIR / "interim" / "missing_mask.parquet",
    DATA_DIR / "processed" / "train.parquet",
    DATA_DIR / "processed" / "val.parquet",
    DATA_DIR / "processed" / "test.parquet",
    DATA_DIR / "processed" / "train_mask.parquet",
    DATA_DIR / "processed" / "val_mask.parquet",
    DATA_DIR / "processed" / "test_mask.parquet",
    DATA_DIR / "processed" / "split_manifest.parquet",
    PROJECT_ROOT / "models" / "preprocessing" / "scaler.joblib",
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading a complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_paths(stride: int) -> tuple[Path, ...]:
    """Return canonical and Stage 3B artifacts that this stage must not alter."""
    stride_dir = WINDOWS_DIR / f"stride_{stride}"
    stage3_paths = tuple(
        stride_dir / f"{split_name}_windows.parquet"
        for split_name in SPLIT_NAMES
    ) + (stride_dir / "window_summary.json",)
    analysis_dir = ANALYSIS_ROOT if stride == 30 else ANALYSIS_ROOT / f"stride_{stride}"
    analysis_paths = tuple(
        sorted(path for path in analysis_dir.rglob("*") if path.is_file())
    )
    return PROTECTED_FIXED_PATHS + stage3_paths + analysis_paths


def protected_hashes(stride: int) -> dict[str, str]:
    """Hash all protected artifacts and fail if any expected input is absent."""
    paths = protected_paths(stride)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required protected artifacts are missing: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in paths
    }


def load_manifest(split_name: str, stride: int) -> pd.DataFrame:
    """Load one canonical, unfiltered Stage 3 manifest."""
    path = WINDOWS_DIR / f"stride_{stride}" / f"{split_name}_windows.parquet"
    manifest = pd.read_parquet(path, engine="pyarrow")
    required = {
        "window_id",
        ID_COLUMN,
        LABEL_COLUMN,
        SPLIT_COLUMN,
        "missing_count",
        "missing_fraction",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{split_name} manifest is missing columns: {sorted(missing)}")
    if manifest["window_id"].isna().any() or not manifest["window_id"].is_unique:
        raise ValueError(f"{split_name} window_id must be non-missing and unique")
    if set(manifest[SPLIT_COLUMN].unique()) != {split_name}:
        raise ValueError(f"{split_name} manifest contains unexpected split labels")
    if not set(manifest[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{split_name} FLAG is not binary")
    return manifest


def retain_usable_windows(original: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Apply the adopted rule and return an exact, order-preserving subset."""
    counts = original["missing_count"].to_numpy()
    fractions = original["missing_fraction"].to_numpy(dtype=np.float64, copy=False)
    if np.any((counts < 0) | (counts > WINDOW_LENGTH)):
        raise ValueError(f"{split_name} missing_count is outside [0, {WINDOW_LENGTH}]")
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError(f"{split_name} missing_fraction is outside [0, 1]")
    if not np.allclose(
        fractions,
        counts.astype(np.float64) / WINDOW_LENGTH,
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError(f"{split_name} missing_count and missing_fraction disagree")

    fully_missing_by_count = counts == WINDOW_LENGTH
    fully_missing_by_fraction = np.isclose(fractions, 1.0, rtol=0, atol=1e-7)
    if not np.array_equal(fully_missing_by_count, fully_missing_by_fraction):
        raise ValueError(f"{split_name} eligibility rules are not equivalent")

    retained = original.loc[~fully_missing_by_count].reset_index(drop=True).copy()
    validate_exact_subset(original, retained, split_name)
    return retained


def validate_exact_subset(
    original: pd.DataFrame, retained: pd.DataFrame, split_name: str
) -> None:
    """Prove the output is exactly the eligible input rows in original order."""
    expected = original.loc[
        original["missing_count"].lt(WINDOW_LENGTH)
    ].reset_index(drop=True)
    if list(retained.columns) != list(original.columns):
        raise ValueError(f"{split_name} output column order changed")
    if not retained.dtypes.equals(original.dtypes):
        raise TypeError(f"{split_name} output dtypes changed")
    if not retained.equals(expected):
        raise ValueError(f"{split_name} output is not the exact eligible subset")
    if retained["window_id"].duplicated().any():
        raise ValueError(f"{split_name} output contains duplicate window IDs")
    if not retained["window_id"].isin(original["window_id"]).all():
        raise ValueError(f"{split_name} output introduced unknown windows")
    if retained["missing_count"].ge(WINDOW_LENGTH).any():
        raise ValueError(f"{split_name} output retained a fully missing window")
    if retained["missing_fraction"].ge(1.0).any():
        raise ValueError(f"{split_name} output retained missing_fraction >= 1")


def count_and_percentage(condition: np.ndarray, denominator: int) -> dict[str, float | int]:
    """Summarize a Boolean condition against an explicit denominator."""
    count = int(np.count_nonzero(condition))
    return {
        "count": count,
        "percentage": 100.0 * count / denominator if denominator else 0.0,
    }


def missingness_summary(values: Iterable[float]) -> dict[str, object]:
    """Summarize retained-window missing fractions."""
    fractions = np.asarray(list(values), dtype=np.float64)
    if fractions.size == 0:
        raise ValueError("Cannot summarize an empty retained manifest")
    total = int(fractions.size)
    return {
        "mean": float(np.mean(fractions)),
        "median": float(np.median(fractions)),
        "p75": float(np.quantile(fractions, 0.75)),
        "p90": float(np.quantile(fractions, 0.90)),
        "p95": float(np.quantile(fractions, 0.95)),
        "p99": float(np.quantile(fractions, 0.99)),
        "maximum": float(np.max(fractions)),
        "zero_missing": count_and_percentage(fractions == 0, total),
        "over_25_percent": count_and_percentage(fractions > 0.25, total),
        "over_50_percent": count_and_percentage(fractions > 0.50, total),
        "over_75_percent": count_and_percentage(fractions > 0.75, total),
        "fully_missing": count_and_percentage(fractions >= 1.0, total),
    }


def customer_coverage(
    original: pd.DataFrame, retained: pd.DataFrame, split_name: str
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """Summarize customer retention and enumerate customers losing all windows."""
    before = original[[ID_COLUMN, LABEL_COLUMN]].drop_duplicates()
    after = retained[[ID_COLUMN, LABEL_COLUMN]].drop_duplicates()
    if before[ID_COLUMN].duplicated().any() or after[ID_COLUMN].duplicated().any():
        raise ValueError(f"{split_name} has inconsistent labels within customer")

    original_counts = original.groupby(ID_COLUMN, sort=False).size()
    retained_counts = retained.groupby(ID_COLUMN, sort=False).size()
    lost_ids = before.loc[~before[ID_COLUMN].isin(after[ID_COLUMN]), ID_COLUMN]
    labels = before.set_index(ID_COLUMN)[LABEL_COLUMN]
    lost_records = [
        {
            ID_COLUMN: str(customer_id),
            LABEL_COLUMN: int(labels.loc[customer_id]),
            SPLIT_COLUMN: split_name,
            "original_window_count": int(original_counts.loc[customer_id]),
            "retained_window_count": int(retained_counts.get(customer_id, 0)),
        }
        for customer_id in lost_ids
    ]
    coverage = {
        "original_customers": int(len(before)),
        "retained_customers": int(len(after)),
        "customers_losing_all_windows": int(len(lost_records)),
        "normal_customers_losing_all_windows": int(
            sum(record[LABEL_COLUMN] == 0 for record in lost_records)
        ),
        "theft_customers_losing_all_windows": int(
            sum(record[LABEL_COLUMN] == 1 for record in lost_records)
        ),
    }
    return coverage, lost_records


def split_summary(
    original: pd.DataFrame, retained: pd.DataFrame, split_name: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build the required split-level filtering report."""
    original_total = len(original)
    retained_total = len(retained)
    original_theft = int(original[LABEL_COLUMN].eq(1).sum())
    retained_theft = int(retained[LABEL_COLUMN].eq(1).sum())
    coverage, lost_records = customer_coverage(original, retained, split_name)
    report: dict[str, object] = {
        "original_windows": original_total,
        "retained_windows": retained_total,
        "removed_fully_missing_windows": original_total - retained_total,
        "retained_percentage": 100.0 * retained_total / original_total,
        "original_normal_windows": original_total - original_theft,
        "original_theft_windows": original_theft,
        "retained_normal_windows": retained_total - retained_theft,
        "retained_theft_windows": retained_theft,
        "theft_percentage_before": 100.0 * original_theft / original_total,
        "theft_percentage_after": 100.0 * retained_theft / retained_total,
        "customer_coverage": coverage,
        "retained_missingness": missingness_summary(retained["missing_fraction"]),
    }
    return report, lost_records


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one filtered manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(payload: dict[str, object], path: Path) -> None:
    """Atomically write the filtering summary."""
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
    """Parse the deliberately narrow Stage 3C configuration."""
    parser = argparse.ArgumentParser(
        description="Retain manifest rows with at least one observed reading."
    )
    parser.add_argument(
        "--stride",
        type=int,
        choices=SUPPORTED_STRIDES,
        default=DEFAULT_STRIDE,
        help="Manifest stride to filter with the fixed missing_count < 30 rule.",
    )
    return parser.parse_args()


def main() -> None:
    """Filter one supported stride and write a validated audit summary."""
    args = parse_args()
    stride = args.stride
    hashes_before = protected_hashes(stride)
    usable_dir = WINDOWS_DIR / f"stride_{stride}" / "usable"
    split_reports: dict[str, dict[str, object]] = {}
    lost_customers: list[dict[str, object]] = []
    combined_original: list[pd.DataFrame] = []
    combined_retained: list[pd.DataFrame] = []

    for split_name in SPLIT_NAMES:
        original = load_manifest(split_name, stride)
        retained = retain_usable_windows(original, split_name)
        report, lost_records = split_summary(original, retained, split_name)
        output_path = usable_dir / f"{split_name}_windows.parquet"
        save_parquet(retained, output_path)
        reopened = pd.read_parquet(output_path, engine="pyarrow")
        validate_exact_subset(original, reopened, split_name)
        report["output_path"] = str(output_path.relative_to(PROJECT_ROOT))
        report["output_size_bytes"] = output_path.stat().st_size
        split_reports[split_name] = report
        lost_customers.extend(lost_records)
        combined_original.append(original[[ID_COLUMN, LABEL_COLUMN, "missing_count", "missing_fraction"]])
        combined_retained.append(retained[[ID_COLUMN, LABEL_COLUMN, "missing_count", "missing_fraction"]])
        del original, retained, reopened
        gc.collect()

    original_all = pd.concat(combined_original, ignore_index=True)
    retained_all = pd.concat(combined_retained, ignore_index=True)
    overall_report, overall_lost = split_summary(original_all, retained_all, "overall")
    if sorted(
        (record[ID_COLUMN], record[LABEL_COLUMN]) for record in lost_customers
    ) != sorted(
        (record[ID_COLUMN], record[LABEL_COLUMN]) for record in overall_lost
    ):
        raise ValueError("Per-split and overall lost-customer records disagree")
    hashes_after = protected_hashes(stride)
    if hashes_before != hashes_after:
        changed = [
            path for path in hashes_before if hashes_before[path] != hashes_after[path]
        ]
        raise RuntimeError(f"Protected artifacts changed during Stage 3C: {changed}")

    summary: dict[str, object] = {
        "stage": "Stage 3C — Window Quality Filtering",
        "configuration": {
            "window_length": WINDOW_LENGTH,
            "stride": stride,
            "eligibility_rule": "missing_fraction < 1.0",
            "equivalent_rule": "missing_count < 30",
            "interpretation": "retain windows with at least one observed reading",
            "other_filters_applied": False,
            "rows_transformed": False,
        },
        "baseline_model_input_policy": {
            "included": ["scaled consumption sequence"],
            "excluded": [
                "missingness mask",
                "missing_fraction",
                LABEL_COLUMN,
                ID_COLUMN,
            ],
        },
        "splits": split_reports,
        "overall": overall_report,
        "customers_losing_all_windows": lost_customers,
        "integrity": {
            "outputs_are_exact_order_preserving_subsets": True,
            "fully_missing_windows_remaining": int(
                retained_all["missing_count"].eq(WINDOW_LENGTH).sum()
            ),
            "protected_artifacts_unchanged": True,
            "protected_sha256": hashes_after,
        },
    }
    summary_path = usable_dir / "filter_summary.json"
    save_json(summary, summary_path)

    print("\nSGCC Stage 3C window-quality filtering report")
    print("Rule: missing_fraction < 1.0 (equivalently missing_count < 30)")
    for split_name in SPLIT_NAMES:
        report = split_reports[split_name]
        coverage = report["customer_coverage"]
        print(
            f"{split_name}: {report['original_windows']:,} -> "
            f"{report['retained_windows']:,} windows "
            f"({report['retained_percentage']:.4f}% retained); "
            f"customers {coverage['original_customers']:,} -> "
            f"{coverage['retained_customers']:,}"
        )
    print(
        f"overall: {overall_report['original_windows']:,} -> "
        f"{overall_report['retained_windows']:,} windows "
        f"({overall_report['retained_percentage']:.4f}% retained)"
    )
    print(
        f"retained classes: normal={overall_report['retained_normal_windows']:,}, "
        f"theft={overall_report['retained_theft_windows']:,}; "
        f"theft share={overall_report['theft_percentage_after']:.4f}%"
    )
    for record in lost_customers:
        print(
            "customer losing all windows: "
            f"{record[ID_COLUMN]}, FLAG={record[LABEL_COLUMN]}, "
            f"split={record[SPLIT_COLUMN]}, "
            f"{record['original_window_count']} -> "
            f"{record['retained_window_count']}"
        )
    print(f"summary: {summary_path}")
    print("Protected artifact hashes unchanged: True")


if __name__ == "__main__":
    main()
