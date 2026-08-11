"""Stage 3B: descriptive missingness-quality analysis for stride-30 windows.

This module never filters or rewrites canonical window manifests. It quantifies
candidate threshold effects at window, customer, class, temporal-position, and
consecutive-run levels, then writes reproducible analysis reports.
"""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_DIR = PROJECT_ROOT / "data" / "windows" / "stride_30"
OUTPUT_DIR = PROJECT_ROOT / "results" / "missingness_analysis"

ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
SPLIT_COLUMN = "split"
WINDOW_LENGTH = 30
WINDOWS_PER_CUSTOMER = 34
SUPPORTED_STRIDES = (30, 15)
DEFAULT_STRIDE = 30
SPLIT_NAMES = ("train", "val", "test")

MANIFEST_PATHS = {
    split: WINDOW_DIR / f"{split}_windows.parquet" for split in SPLIT_NAMES
}

THRESHOLDS = (
    {"label": "0.00", "maximum_missing_fraction": 0.00, "maximum_missing_count": 0},
    {"label": "0.10", "maximum_missing_fraction": 0.10, "maximum_missing_count": 3},
    {"label": "0.25", "maximum_missing_fraction": 0.25, "maximum_missing_count": 7},
    {"label": "0.50", "maximum_missing_fraction": 0.50, "maximum_missing_count": 15},
    {"label": "0.75", "maximum_missing_fraction": 0.75, "maximum_missing_count": 22},
    {
        "label": "<1.00 (remove fully missing only)",
        "maximum_missing_fraction": None,
        "maximum_missing_count": 29,
    },
)

CANONICAL_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "data set.csv",
    PROJECT_ROOT / "data" / "interim" / "cleaned_data.parquet",
    PROJECT_ROOT / "data" / "interim" / "missing_mask.parquet",
    PROJECT_ROOT / "data" / "processed" / "train.parquet",
    PROJECT_ROOT / "data" / "processed" / "val.parquet",
    PROJECT_ROOT / "data" / "processed" / "test.parquet",
    PROJECT_ROOT / "data" / "processed" / "train_mask.parquet",
    PROJECT_ROOT / "data" / "processed" / "val_mask.parquet",
    PROJECT_ROOT / "data" / "processed" / "test_mask.parquet",
    PROJECT_ROOT / "data" / "processed" / "split_manifest.parquet",
    PROJECT_ROOT / "models" / "preprocessing" / "scaler.joblib",
    MANIFEST_PATHS["train"],
    MANIFEST_PATHS["val"],
    MANIFEST_PATHS["test"],
    WINDOW_DIR / "window_summary.json",
)

def configure_stride(stride: int) -> None:
    """Route one supported stride to isolated inputs and outputs."""
    global WINDOW_DIR, OUTPUT_DIR, WINDOWS_PER_CUSTOMER
    global MANIFEST_PATHS, CANONICAL_PATHS

    if stride not in SUPPORTED_STRIDES:
        raise ValueError(f"Supported strides: {SUPPORTED_STRIDES}; got {stride}")
    window_dir = PROJECT_ROOT / "data" / "windows" / f"stride_{stride}"
    summary_path = window_dir / "window_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Window summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as source:
        summary = json.load(source)
    configuration = summary["configuration"]
    if int(configuration["stride"]) != stride:
        raise ValueError("Window summary stride does not match the requested stride")
    if int(configuration["window_length"]) != WINDOW_LENGTH:
        raise ValueError("Window summary length differs from the fixed model input")

    WINDOW_DIR = window_dir
    OUTPUT_DIR = (
        PROJECT_ROOT / "results" / "missingness_analysis"
        if stride == 30
        else PROJECT_ROOT / "results" / "missingness_analysis" / f"stride_{stride}"
    )
    WINDOWS_PER_CUSTOMER = int(configuration["windows_per_customer"])
    MANIFEST_PATHS = {
        split: WINDOW_DIR / f"{split}_windows.parquet" for split in SPLIT_NAMES
    }
    CANONICAL_PATHS = CANONICAL_PATHS[:11] + (
        MANIFEST_PATHS["train"],
        MANIFEST_PATHS["val"],
        MANIFEST_PATHS["test"],
        summary_path,
    )


ANALYSIS_COLUMNS = [
    ID_COLUMN,
    LABEL_COLUMN,
    SPLIT_COLUMN,
    "window_number_within_customer",
    "start_feature_index",
    "end_feature_index",
    "start_date",
    "end_date",
    "missing_count",
    "missing_fraction",
]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the complete file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hashes() -> dict[str, str]:
    """Hash every Stage 1, 2, and canonical Stage 3 artifact."""
    missing = [str(path) for path in CANONICAL_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Canonical artifacts are missing: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in CANONICAL_PATHS
    }


def load_manifests() -> dict[str, pd.DataFrame]:
    """Load required columns and validate each canonical stride-30 manifest."""
    manifests: dict[str, pd.DataFrame] = {}
    seen_customers: dict[str, set[str]] = {}
    for split, path in MANIFEST_PATHS.items():
        frame = pd.read_parquet(path, columns=ANALYSIS_COLUMNS, engine="pyarrow")
        if set(frame[SPLIT_COLUMN].unique()) != {split}:
            raise ValueError(f"{split} manifest contains an unexpected split label")
        if not set(frame[LABEL_COLUMN].unique()).issubset({0, 1}):
            raise ValueError(f"{split} manifest contains a non-binary FLAG")
        if frame[ID_COLUMN].isna().any():
            raise ValueError(f"{split} manifest contains missing customer IDs")
        if not frame["missing_count"].between(0, WINDOW_LENGTH).all():
            raise ValueError(f"{split} missing_count is outside [0, 30]")
        if not frame["missing_fraction"].between(0, 1).all():
            raise ValueError(f"{split} missing_fraction is outside [0, 1]")
        if not np.allclose(
            frame["missing_fraction"].to_numpy(),
            frame["missing_count"].to_numpy() / WINDOW_LENGTH,
            rtol=0,
            atol=1e-7,
        ):
            raise ValueError(f"{split} missing_count and fraction disagree")

        customer_sizes = frame.groupby(ID_COLUMN, sort=False).size()
        if not customer_sizes.eq(WINDOWS_PER_CUSTOMER).all():
            raise ValueError(
                f"{split} customers do not all have {WINDOWS_PER_CUSTOMER} windows"
            )
        numbers = frame["window_number_within_customer"].to_numpy().reshape(
            len(customer_sizes), WINDOWS_PER_CUSTOMER
        )
        expected = np.broadcast_to(
            np.arange(WINDOWS_PER_CUSTOMER), numbers.shape
        )
        if not np.array_equal(numbers, expected):
            raise ValueError(f"{split} window positions are not deterministic")

        seen_customers[split] = set(frame[ID_COLUMN].astype(str))
        manifests[split] = frame

    overlaps = {
        "train_val": seen_customers["train"] & seen_customers["val"],
        "train_test": seen_customers["train"] & seen_customers["test"],
        "val_test": seen_customers["val"] & seen_customers["test"],
    }
    if any(overlaps.values()):
        raise ValueError(
            "Customer overlap found across canonical manifests: "
            f"{ {name: len(ids) for name, ids in overlaps.items()} }"
        )
    return manifests


def scopes(manifests: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return split-specific frames plus a combined overall frame."""
    return {
        **manifests,
        "overall": pd.concat(
            [manifests[name] for name in SPLIT_NAMES], ignore_index=True
        ),
    }


def threshold_is_retained(
    missing_count: pd.Series, threshold: dict[str, object]
) -> pd.Series:
    """Apply explicit integer-count semantics for a threshold."""
    return missing_count <= int(threshold["maximum_missing_count"])


def evaluate_thresholds(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate window- and customer-level impact for every threshold and scope."""
    window_records: list[dict[str, object]] = []
    customer_records: list[dict[str, object]] = []

    for scope_name, frame in frames.items():
        original_windows = len(frame)
        original_theft = int(frame[LABEL_COLUMN].eq(1).sum())
        theft_percentage_before = 100.0 * original_theft / original_windows
        customer_labels = frame.groupby(ID_COLUMN, sort=False)[LABEL_COLUMN].first()
        total_per_customer = frame.groupby(ID_COLUMN, sort=False).size()
        total_customers = len(total_per_customer)

        for threshold in THRESHOLDS:
            retained_mask = threshold_is_retained(
                frame["missing_count"], threshold
            )
            retained = frame.loc[retained_mask]
            removed = frame.loc[~retained_mask]
            retained_windows = len(retained)
            removed_windows = len(removed)
            retained_normal = int(retained[LABEL_COLUMN].eq(0).sum())
            retained_theft = int(retained[LABEL_COLUMN].eq(1).sum())
            removed_normal = int(removed[LABEL_COLUMN].eq(0).sum())
            removed_theft = int(removed[LABEL_COLUMN].eq(1).sum())
            rule = (
                "missing_fraction < 1.00"
                if threshold["maximum_missing_fraction"] is None
                else "missing_fraction <= "
                f"{float(threshold['maximum_missing_fraction']):.2f}"
            )

            window_records.append(
                {
                    "scope": scope_name,
                    "threshold_label": threshold["label"],
                    "retention_rule": rule,
                    "maximum_allowed_missing_count": int(
                        threshold["maximum_missing_count"]
                    ),
                    "original_windows": original_windows,
                    "retained_windows": retained_windows,
                    "removed_windows": removed_windows,
                    "retained_percentage": 100.0
                    * retained_windows
                    / original_windows,
                    "removed_percentage": 100.0
                    * removed_windows
                    / original_windows,
                    "retained_normal_windows": retained_normal,
                    "retained_theft_windows": retained_theft,
                    "removed_normal_windows": removed_normal,
                    "removed_theft_windows": removed_theft,
                    "theft_percentage_before": theft_percentage_before,
                    "theft_percentage_after": (
                        100.0 * retained_theft / retained_windows
                        if retained_windows
                        else np.nan
                    ),
                }
            )

            retained_counts = (
                retained.groupby(ID_COLUMN, sort=False)
                .size()
                .reindex(total_per_customer.index, fill_value=0)
            )
            losing_any = retained_counts < total_per_customer
            losing_all = retained_counts.eq(0)
            losing_some_not_all = losing_any & ~losing_all
            retain_at_least_one = retained_counts.gt(0)
            normal_customers = customer_labels.eq(0)
            theft_customers = customer_labels.eq(1)

            customer_records.append(
                {
                    "scope": scope_name,
                    "threshold_label": threshold["label"],
                    "retention_rule": rule,
                    "maximum_allowed_missing_count": int(
                        threshold["maximum_missing_count"]
                    ),
                    "total_customers": total_customers,
                    "customers_retaining_at_least_one_window": int(
                        retain_at_least_one.sum()
                    ),
                    "customers_losing_any_windows": int(losing_any.sum()),
                    "customers_losing_some_but_not_all_windows": int(
                        losing_some_not_all.sum()
                    ),
                    "customers_losing_all_windows": int(losing_all.sum()),
                    "percentage_customers_losing_all_windows": 100.0
                    * int(losing_all.sum())
                    / total_customers,
                    "normal_customers_total": int(normal_customers.sum()),
                    "normal_customers_retaining_at_least_one": int(
                        (normal_customers & retain_at_least_one).sum()
                    ),
                    "normal_customers_losing_all": int(
                        (normal_customers & losing_all).sum()
                    ),
                    "theft_customers_total": int(theft_customers.sum()),
                    "theft_customers_retaining_at_least_one": int(
                        (theft_customers & retain_at_least_one).sum()
                    ),
                    "theft_customers_losing_all": int(
                        (theft_customers & losing_all).sum()
                    ),
                }
            )

    return pd.DataFrame(window_records), pd.DataFrame(customer_records)


def summarize_by_class(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Describe missing_fraction by FLAG within every split and overall."""
    records: list[dict[str, object]] = []
    for scope_name, frame in frames.items():
        for flag in (0, 1):
            values = frame.loc[
                frame[LABEL_COLUMN].eq(flag), "missing_fraction"
            ].to_numpy(dtype=np.float64)
            records.append(
                {
                    "scope": scope_name,
                    "FLAG": flag,
                    "windows": len(values),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std_population": float(values.std(ddof=0)),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p90": float(np.quantile(values, 0.90)),
                    "p95": float(np.quantile(values, 0.95)),
                    "p99": float(np.quantile(values, 0.99)),
                    "fraction_zero_missing": float(np.mean(values == 0)),
                    "fraction_over_25_percent": float(np.mean(values > 0.25)),
                    "fraction_over_50_percent": float(np.mean(values > 0.50)),
                    "fraction_over_75_percent": float(np.mean(values > 0.75)),
                    "fraction_fully_missing": float(np.mean(values == 1.0)),
                }
            )
    return pd.DataFrame(records)


def analyze_window_positions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize whether missingness concentrates at temporal positions."""
    records: list[dict[str, object]] = []
    for scope_name, frame in frames.items():
        grouped = frame.groupby("window_number_within_customer", sort=True)
        for position, group in grouped:
            values = group["missing_fraction"].to_numpy(dtype=np.float64)
            records.append(
                {
                    "scope": scope_name,
                    "window_number_within_customer": int(position),
                    "start_feature_index": int(group["start_feature_index"].iloc[0]),
                    "end_feature_index": int(group["end_feature_index"].iloc[0]),
                    "start_date": group["start_date"].iloc[0],
                    "end_date": group["end_date"].iloc[0],
                    "windows": len(group),
                    "average_missing_fraction": float(values.mean()),
                    "median_missing_fraction": float(np.median(values)),
                    "percentage_fully_missing": 100.0
                    * float(np.mean(values == 1.0)),
                    "percentage_zero_missing": 100.0
                    * float(np.mean(values == 0.0)),
                }
            )
    return pd.DataFrame(records)


def customer_full_missing_details(
    manifests: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return one row per customer with fully-missing count and maximum run."""
    parts: list[pd.DataFrame] = []
    for split_name, frame in manifests.items():
        customer_ids = (
            frame[ID_COLUMN].to_numpy().reshape(-1, WINDOWS_PER_CUSTOMER)[:, 0]
        )
        flags = (
            frame[LABEL_COLUMN].to_numpy().reshape(-1, WINDOWS_PER_CUSTOMER)[:, 0]
        )
        full = frame["missing_count"].eq(WINDOW_LENGTH).to_numpy().reshape(
            -1, WINDOWS_PER_CUSTOMER
        )
        current = np.zeros(len(customer_ids), dtype=np.int16)
        maximum = np.zeros(len(customer_ids), dtype=np.int16)
        for position in range(WINDOWS_PER_CUSTOMER):
            current = np.where(full[:, position], current + 1, 0)
            maximum = np.maximum(maximum, current)
        parts.append(
            pd.DataFrame(
                {
                    ID_COLUMN: pd.Series(customer_ids, dtype="string"),
                    LABEL_COLUMN: flags.astype(np.int8),
                    SPLIT_COLUMN: split_name,
                    "fully_missing_windows": full.sum(axis=1).astype(np.int16),
                    "maximum_consecutive_fully_missing_run": maximum,
                }
            )
        )
    details = pd.concat(parts, ignore_index=True)
    if not details[ID_COLUMN].is_unique:
        raise ValueError("Customer detail analysis found duplicate customers")
    return details


def summarize_fully_missing_windows(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize fully-missing window counts and percentages by class."""
    records: list[dict[str, object]] = []
    for scope_name, frame in frames.items():
        for flag in (0, 1):
            subset = frame.loc[frame[LABEL_COLUMN].eq(flag)]
            count = int(subset["missing_count"].eq(WINDOW_LENGTH).sum())
            records.append(
                {
                    "scope": scope_name,
                    "FLAG": flag,
                    "total_windows": len(subset),
                    "fully_missing_windows": count,
                    "percentage_fully_missing": 100.0 * count / len(subset),
                }
            )
    return pd.DataFrame(records)


def customer_detail_scopes(details: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return customer-detail views for each split and overall."""
    return {
        "train": details.loc[details[SPLIT_COLUMN].eq("train")],
        "val": details.loc[details[SPLIT_COLUMN].eq("val")],
        "test": details.loc[details[SPLIT_COLUMN].eq("test")],
        "overall": details,
    }


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    """Return requested percentiles for a per-customer integer metric."""
    return {
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def summarize_fully_missing_customers(
    details: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize counts, distributions, and consecutive fully-missing runs."""
    summary_records: list[dict[str, object]] = []
    count_distribution_records: list[dict[str, object]] = []
    run_summary_records: list[dict[str, object]] = []
    run_distribution_records: list[dict[str, object]] = []

    for scope_name, scoped in customer_detail_scopes(details).items():
        for flag_label, subset in (
            ("all", scoped),
            ("0", scoped.loc[scoped[LABEL_COLUMN].eq(0)]),
            ("1", scoped.loc[scoped[LABEL_COLUMN].eq(1)]),
        ):
            counts = subset["fully_missing_windows"].to_numpy()
            runs = subset["maximum_consecutive_fully_missing_run"].to_numpy()
            count_stats = percentile_summary(counts)
            run_stats = percentile_summary(runs)
            summary_records.append(
                {
                    "scope": scope_name,
                    "FLAG": flag_label,
                    "total_customers": len(subset),
                    "customers_with_zero_fully_missing_windows": int(
                        np.count_nonzero(counts == 0)
                    ),
                    "customers_with_at_least_one_fully_missing_window": int(
                        np.count_nonzero(counts > 0)
                    ),
                    "customers_all_windows_fully_missing": int(
                        np.count_nonzero(counts == WINDOWS_PER_CUSTOMER)
                    ),
                    **{
                        f"fully_missing_count_{key}": value
                        for key, value in count_stats.items()
                    },
                }
            )
            run_summary_records.append(
                {
                    "scope": scope_name,
                    "FLAG": flag_label,
                    "total_customers": len(subset),
                    "customers_with_at_least_one_fully_missing_run": int(
                        np.count_nonzero(runs > 0)
                    ),
                    **{
                        f"maximum_run_{key}": value
                        for key, value in run_stats.items()
                    },
                }
            )
            count_values, count_frequencies = np.unique(counts, return_counts=True)
            for value, frequency in zip(count_values, count_frequencies):
                count_distribution_records.append(
                    {
                        "scope": scope_name,
                        "FLAG": flag_label,
                        "fully_missing_windows_per_customer": int(value),
                        "customers": int(frequency),
                        "percentage": 100.0 * int(frequency) / len(subset),
                    }
                )
            run_values, run_frequencies = np.unique(runs, return_counts=True)
            for value, frequency in zip(run_values, run_frequencies):
                run_distribution_records.append(
                    {
                        "scope": scope_name,
                        "FLAG": flag_label,
                        "maximum_consecutive_run": int(value),
                        "customers": int(frequency),
                        "percentage": 100.0 * int(frequency) / len(subset),
                    }
                )
    return (
        pd.DataFrame(summary_records),
        pd.DataFrame(count_distribution_records),
        pd.DataFrame(run_summary_records),
        pd.DataFrame(run_distribution_records),
    )


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-safe records."""
    return json.loads(frame.to_json(orient="records"))


def threshold_table_markdown(
    threshold_comparison: pd.DataFrame,
    customer_retention: pd.DataFrame,
) -> str:
    """Build a concise combined-threshold table for the Markdown report."""
    combined = threshold_comparison.loc[
        threshold_comparison["scope"].eq("overall")
    ].merge(
        customer_retention.loc[customer_retention["scope"].eq("overall")][
            [
                "threshold_label",
                "customers_losing_all_windows",
                "theft_customers_losing_all",
            ]
        ],
        on="threshold_label",
        how="left",
    )
    lines = [
        "| Rule | Retained windows | Retained % | Theft % before | Theft % after | Customers losing all | Theft customers losing all |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.itertuples():
        lines.append(
            f"| {row.threshold_label} | {row.retained_windows:,} | "
            f"{row.retained_percentage:.2f}% | "
            f"{row.theft_percentage_before:.3f}% | "
            f"{row.theft_percentage_after:.3f}% | "
            f"{row.customers_losing_all_windows:,} | "
            f"{row.theft_customers_losing_all:,} |"
        )
    return "\n".join(lines)


def build_markdown_report(
    threshold_comparison: pd.DataFrame,
    customer_retention: pd.DataFrame,
    class_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    fully_window_summary: pd.DataFrame,
    fully_customer_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    top_run_examples: pd.DataFrame,
) -> str:
    """Create the human-readable evidence review and proposed recommendation."""
    minimum_label = "<1.00 (remove fully missing only)"
    overall_windows = threshold_comparison.loc[
        threshold_comparison["scope"].eq("overall")
        & threshold_comparison["threshold_label"].eq(minimum_label)
    ].iloc[0]
    overall_customers = customer_retention.loc[
        customer_retention["scope"].eq("overall")
        & customer_retention["threshold_label"].eq(minimum_label)
    ].iloc[0]
    full_class = fully_window_summary.loc[
        fully_window_summary["scope"].eq("overall")
    ].set_index("FLAG")
    normal_full = full_class.loc[0]
    theft_full = full_class.loc[1]
    all_customer_full = fully_customer_summary.loc[
        fully_customer_summary["scope"].eq("overall")
        & fully_customer_summary["FLAG"].eq("all")
    ].iloc[0]
    overall_run = run_summary.loc[
        run_summary["scope"].eq("overall")
        & run_summary["FLAG"].eq("all")
    ].iloc[0]
    overall_positions = position_summary.loc[
        position_summary["scope"].eq("overall")
    ].sort_values("average_missing_fraction", ascending=False)
    class_overall = class_summary.loc[class_summary["scope"].eq("overall")]

    top_lines = [
        "| CONS_NO | FLAG | Split | Fully missing windows | Maximum consecutive run |",
        "|---|---:|---|---:|---:|",
    ]
    for row in top_run_examples.itertuples():
        top_lines.append(
            f"| {row.CONS_NO} | {row.FLAG} | {row.split} | "
            f"{row.fully_missing_windows} | "
            f"{row.maximum_consecutive_fully_missing_run} |"
        )

    class_lines = [
        "| FLAG | Mean | Median | P90 | Fully missing | Zero missing |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in class_overall.itertuples():
        class_lines.append(
            f"| {row.FLAG} | {row.mean:.4f} | {row.median:.4f} | "
            f"{row.p90:.4f} | {100*row.fraction_fully_missing:.2f}% | "
            f"{100*row.fraction_zero_missing:.2f}% |"
        )

    position_lines = [
        "| Position | Dates | Mean missing | Fully missing | Zero missing |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in overall_positions.head(5).itertuples():
        position_lines.append(
            f"| {row.window_number_within_customer} | "
            f"{row.start_date} to {row.end_date} | "
            f"{row.average_missing_fraction:.4f} | "
            f"{row.percentage_fully_missing:.2f}% | "
            f"{row.percentage_zero_missing:.2f}% |"
        )

    return f"""# Stage 3B — Missingness Quality Analysis

## Scope

This analysis evaluates the canonical {WINDOW_DIR.name} manifests without filtering or
rewriting them. A threshold of 0.25 means retain windows with
missing_fraction <= 0.25. Comparisons use integer counts out of 30 to avoid
floating-point boundary ambiguity. The special <1.00 rule retains every window
except those with 30/30 originally missing positions.

## Threshold comparison

{threshold_table_markdown(threshold_comparison, customer_retention)}

## Fully missing windows

- Fully missing normal-customer windows: {int(normal_full.fully_missing_windows):,}
  ({normal_full.percentage_fully_missing:.2f}% of normal windows).
- Fully missing theft-customer windows: {int(theft_full.fully_missing_windows):,}
  ({theft_full.percentage_fully_missing:.2f}% of theft-customer windows).
- Customers with at least one fully missing window:
  {int(all_customer_full.customers_with_at_least_one_fully_missing_window):,}.
- Customers whose all {WINDOWS_PER_CUSTOMER} windows are fully missing:
  {int(all_customer_full.customers_all_windows_fully_missing):,}.
- Longest consecutive fully missing run: {int(overall_run.maximum_run_maximum)}
  windows.

## Missingness by customer class

{chr(10).join(class_lines)}

FLAG is customer-level weak supervision. The class comparison therefore does
not establish whether any particular positive window contains theft.

## Temporal concentration

The five window positions with the highest average missingness are:

{chr(10).join(position_lines)}

## Longest fully missing runs

{chr(10).join(top_lines)}

## Evidence-based recommendation

### A. Minimum obvious rule

**Proposed for review:** remove only fully originally missing windows
(missing_fraction < 1.0). Such windows contain no directly observed consumption
information even though Stage 1 imputation supplied numerical values. This
rule would retain {int(overall_windows.retained_windows):,} windows
({overall_windows.retained_percentage:.2f}%) and remove
{int(overall_windows.removed_windows):,}. It would cause
{int(overall_customers.customers_losing_all_windows):,} customers—including
{int(overall_customers.theft_customers_losing_all):,} theft-labelled
customers—to lose all windows. That customer-level effect must be reviewed
before the rule becomes canonical.

### B. More aggressive rules

Thresholds of <=0.75, <=0.50, <=0.25, <=0.10, and 0% increasingly reduce
dependence on imputation, but they discard more windows and may remove every
window for additional customers. Their actual retention, class composition,
and customer loss are shown above and in the CSV outputs.

The descriptive results do **not** establish an objectively optimal aggressive
threshold. No canonical filtering decision is made here. The minimum rule is a
methodologically clear proposal because 30/30-missing windows contain zero
observed readings; more aggressive filtering should be selected only after
reviewing customer coverage, class distortion, and later sensitivity analysis.
"""


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write an analysis table."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text(text: str, path: Path) -> None:
    """Atomically write a UTF-8 text artifact."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Select one supported manifest stride without changing analysis policy."""
    parser = argparse.ArgumentParser(
        description="Analyze temporal-window missingness without filtering manifests."
    )
    parser.add_argument(
        "--stride", type=int, choices=SUPPORTED_STRIDES, default=DEFAULT_STRIDE
    )
    return parser.parse_args(arguments)


def main() -> None:
    """Run Stage 3B analysis and write reports without changing manifests."""
    args = parse_args()
    configure_stride(args.stride)
    hashes_before = canonical_hashes()
    manifests = load_manifests()
    frames = scopes(manifests)

    threshold_comparison, customer_retention = evaluate_thresholds(frames)
    class_summary = summarize_by_class(frames)
    position_summary = analyze_window_positions(frames)
    customer_details = customer_full_missing_details(manifests)
    fully_window_summary = summarize_fully_missing_windows(frames)
    (
        fully_customer_summary,
        fully_count_distribution,
        run_summary,
        run_distribution,
    ) = summarize_fully_missing_customers(customer_details)

    top_run_examples = customer_details.sort_values(
        [
            "maximum_consecutive_fully_missing_run",
            "fully_missing_windows",
            ID_COLUMN,
        ],
        ascending=[False, False, True],
    ).head(10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "threshold_comparison.csv": threshold_comparison,
        "customer_retention_by_threshold.csv": customer_retention,
        "class_missingness_summary.csv": class_summary,
        "window_position_missingness.csv": position_summary,
        "fully_missing_window_summary.csv": fully_window_summary,
        "fully_missing_customer_summary.csv": fully_customer_summary,
        "fully_missing_count_distribution.csv": fully_count_distribution,
        "fully_missing_run_summary.csv": run_summary,
        "fully_missing_run_distribution.csv": run_distribution,
        "top_fully_missing_runs.csv": top_run_examples,
    }
    for filename, frame in outputs.items():
        write_csv(frame, OUTPUT_DIR / filename)

    report = build_markdown_report(
        threshold_comparison,
        customer_retention,
        class_summary,
        position_summary,
        fully_window_summary,
        fully_customer_summary,
        run_summary,
        top_run_examples,
    )
    if args.stride == 15:
        marker = "## Evidence-based recommendation"
        report = report[: report.index(marker)] + """## Fixed baseline quality rule

The adopted rule remains `missing_count < 30`: remove only fully originally
missing windows. This diagnostic does not select or reconsider a threshold.
Retention, class composition, and customer coverage are reported above. More
aggressive thresholds remain diagnostic only and are not candidates for this
controlled ablation.
"""

    write_text(report, OUTPUT_DIR / "report.md")

    analysis_json = {
        "configuration": {
            "window_length": WINDOW_LENGTH,
            "windows_per_customer": WINDOWS_PER_CUSTOMER,
            "threshold_semantics": (
                "Retain missing_count <= maximum_allowed_missing_count; "
                "<1.00 retains counts 0 through 29."
            ),
            "stride": args.stride,
            "canonical_manifests_filtered_or_modified": False,
            "plots_created": False,
            "plots_note": (
                "Matplotlib was not an existing dependency; auditable tables "
                "and Markdown were prioritized."
            ),
        },
        "threshold_comparison_overall": dataframe_records(
            threshold_comparison.loc[threshold_comparison["scope"].eq("overall")]
        ),
        "customer_retention_overall": dataframe_records(
            customer_retention.loc[customer_retention["scope"].eq("overall")]
        ),
        "class_missingness_overall": dataframe_records(
            class_summary.loc[class_summary["scope"].eq("overall")]
        ),
        "fully_missing_windows": dataframe_records(fully_window_summary),
        "fully_missing_customers": dataframe_records(fully_customer_summary),
        "fully_missing_run_summary": dataframe_records(run_summary),
        "top_fully_missing_run_examples": dataframe_records(top_run_examples),
        "canonical_sha256": hashes_before,
        "recommendation_status": (
            "Fixed baseline rule: remove only fully missing windows; diagnostic only."
            if args.stride == 15
            else "Proposed minimum rule pending review: remove only fully missing "
            "windows. No canonical filtering decision has been made."
        ),
    }
    write_text(
        json.dumps(analysis_json, indent=2) + "\n",
        OUTPUT_DIR / "missingness_analysis.json",
    )

    hashes_after = canonical_hashes()
    if hashes_before != hashes_after:
        changed = [
            path
            for path in hashes_before
            if hashes_before[path] != hashes_after[path]
        ]
        raise RuntimeError(f"Canonical artifacts changed during analysis: {changed}")

    combined_thresholds = threshold_comparison.loc[
        threshold_comparison["scope"].eq("overall")
    ]
    combined_customers = customer_retention.loc[
        customer_retention["scope"].eq("overall")
    ]
    full_customer_overall = fully_customer_summary.loc[
        fully_customer_summary["scope"].eq("overall")
        & fully_customer_summary["FLAG"].eq("all")
    ].iloc[0]
    run_overall = run_summary.loc[
        run_summary["scope"].eq("overall")
        & run_summary["FLAG"].eq("all")
    ].iloc[0]

    print("\nSGCC Stage 3B missingness-quality report")
    for row in combined_thresholds.itertuples():
        customer_row = combined_customers.loc[
            combined_customers["threshold_label"].eq(row.threshold_label)
        ].iloc[0]
        print(
            f"{row.threshold_label}: retained={row.retained_windows:,} "
            f"({row.retained_percentage:.2f}%), "
            f"removed={row.removed_windows:,}, "
            f"theft%={row.theft_percentage_before:.3f}->"
            f"{row.theft_percentage_after:.3f}, "
            f"customers losing all="
            f"{int(customer_row.customers_losing_all_windows):,}"
        )
    print(
        "Fully missing customer impact: "
        f"at least one={int(full_customer_overall.customers_with_at_least_one_fully_missing_window):,}, "
        f"all {WINDOWS_PER_CUSTOMER}="
        f"{int(full_customer_overall.customers_all_windows_fully_missing):,}"
    )
    print(
        "Consecutive fully missing runs: "
        f"maximum={int(run_overall.maximum_run_maximum)} windows"
    )
    if args.stride == 15:
        print("Policy: fixed missing_count < 30 rule; diagnostic analysis only.")
    else:
        print(
            "Recommendation: proposed minimum rule is to remove only fully missing "
            "windows; no canonical filtering decision was applied."
        )
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(
                f"output: {path} "
                f"({path.stat().st_size / 1024**2:.4f} MiB)"
            )
    print("Canonical Stage 1/2/3 hashes unchanged: True")


if __name__ == "__main__":
    main()
