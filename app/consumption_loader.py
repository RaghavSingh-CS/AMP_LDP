"""Read one original CSV history through a reviewed, ID-keyed byte index.

No model imports, inference, transformations, interpolation or artifact writes.
Large immutable-file hashes are cached by filesystem identity. Selected record
bytes and their own SHA-256/CONS_NO/FLAG are checked on every history request.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.config import REPOSITORY_ROOT
from app.data_loader import ArtifactError, _csv, _json, _path, _verified, load_evidence, load_predictions

CONSUMPTION_CATALOG = "app/consumption_catalog.json"


@dataclass(frozen=True)
class CustomerHistory:
    customer_id: str
    readings: pd.DataFrame
    windows: pd.DataFrame
    source: str
    provenance: str
    source_date_columns: int


def require(condition, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def file_identity(path: Path) -> tuple:
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError as exc:
        raise ArtifactError(f"History artifact unavailable: {path.name} ({exc.strerror})") from exc


@lru_cache(maxsize=16)
def _hash_checked(path: str, identity: tuple, expected: str) -> None:
    with Path(path).open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    require(actual == expected, f"Consumption source SHA-256 mismatch: {Path(path).name}")
    require(file_identity(Path(path)) == identity, "History artifact changed during hash verification.")


def verify_file(root: Path, record: dict) -> Path:
    path = _path(root, record["path"])
    identity = file_identity(path)
    _hash_checked(str(path), identity, record["sha256"])
    return path


@lru_cache(maxsize=8)
def _source_calendar(headers: tuple[str, ...]):
    columns = [name for name in headers if name not in {"CONS_NO", "FLAG"}]
    dates = pd.to_datetime(columns, format="%m/%d/%Y", errors="raise")
    require(dates.is_unique and len(dates) > 0, "Invalid or duplicate source dates.")
    return columns, dates, pd.date_range(dates.min(), dates.max(), freq="D")


def parse_record(content: bytes, headers: list[str], customer_id: str, expected_label: int) -> pd.DataFrame:
    """Decode authentic source values on a complete, explicitly marked calendar."""
    try:
        fields = next(csv.reader([content.decode("utf-8")], strict=True))
        require(len(fields) == len(headers), "Raw history record schema mismatch.")
        mapping = dict(zip(headers, fields))
        require(mapping["CONS_NO"] == customer_id, "Raw record belongs to a different customer.")
        require(mapping["FLAG"] == str(expected_label), "Raw evaluation label disagrees with frozen score label.")
        columns, dates, calendar = _source_calendar(tuple(headers))
        values = []
        for name in columns:
            token = mapping[name].strip()
            values.append(np.nan if token.lower() in {"", "nan", "na", "n/a", "null", "#n/a"} else float(token))
        observed = np.asarray(values, dtype=float)
        require(not np.isinf(observed).any() and not (observed < 0).any(), "Invalid original consumption values.")
        series = pd.Series(observed, index=dates).sort_index()
        result = pd.DataFrame({"date": calendar, "consumption": series.reindex(calendar).to_numpy()})
        result["source_column_present"] = result["date"].isin(dates)
        result["missing"] = result["consumption"].isna()
        result["status"] = np.where(~result["source_column_present"], "No source column",
                                    np.where(result["missing"], "Missing reading", "Observed"))
        # This calendar exclusion is part of the frozen cleaning contract.
        result["model_calendar"] = result["date"].ne(pd.Timestamp("2014-10-03"))
        result["model_calendar_note"] = np.where(result["model_calendar"], "Included in processed calendar",
                                                "Excluded by frozen preprocessing")
        # Explicit segments prevent the chart from connecting across missing days.
        result["segment"] = result["missing"].cumsum()
        return result
    except ArtifactError:
        raise
    except (KeyError, ValueError, TypeError, UnicodeError, csv.Error, StopIteration) as exc:
        raise ArtifactError(f"Malformed raw history: {exc}") from exc


def summarize(readings: pd.DataFrame) -> dict:
    values = readings.loc[~readings["missing"], "consumption"]
    count = len(readings)
    return {"calendar_days": count, "observed": len(values), "missing": count - len(values),
            "missing_percent": 100.0 * (count - len(values)) / count if count else 0.0,
            "zero_readings": int(values.eq(0).sum()),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "population_sd": float(values.std(ddof=0)) if len(values) else None}


@lru_cache(maxsize=16)
def _customer_windows(content: bytes, customer_id: str) -> pd.DataFrame:
    return pq.read_table(BytesIO(content), filters=[("CONS_NO", "=", customer_id)]).to_pandas()


def validate_windows(windows: pd.DataFrame, readings: pd.DataFrame, customer_id: str, label: int) -> pd.DataFrame:
    required = {"window_id", "CONS_NO", "FLAG", "split", "window_number_within_customer",
                "start_feature_index", "end_feature_index", "start_date", "end_date", "missing_count"}
    require(required.issubset(windows.columns) and not windows.empty, "Evaluated-window metadata unavailable or malformed.")
    require(windows["CONS_NO"].eq(customer_id).all() and windows["FLAG"].eq(label).all()
            and windows["split"].eq("test").all(), "Evaluated windows belong to another customer or cohort.")
    require(not windows["window_id"].duplicated().any(), "Duplicate evaluated windows.")
    calendar = readings.loc[readings["model_calendar"]].reset_index(drop=True)
    result = windows.sort_values("window_number_within_customer").copy()
    starts = result["start_feature_index"].to_numpy(dtype=int)
    ends = result["end_feature_index"].to_numpy(dtype=int)
    numbers = result["window_number_within_customer"].to_numpy(dtype=int)
    require(bool(((starts == numbers * 30) & (ends == starts + 29) & (starts >= 0) & (ends < len(calendar))).all()),
            "Frozen 30-position window bounds mismatch.")
    require(np.array_equal(pd.to_datetime(result["start_date"], format="%m/%d/%Y").to_numpy(), calendar.date.iloc[starts].to_numpy())
            and np.array_equal(pd.to_datetime(result["end_date"], format="%m/%d/%Y").to_numpy(), calendar.date.iloc[ends].to_numpy()),
            "Evaluated window dates do not match the processed calendar.")
    cumulative = np.r_[0, calendar["missing"].to_numpy().cumsum()]
    missing = cumulative[ends + 1] - cumulative[starts]
    require(bool((missing < 30).all()) and np.array_equal(missing, result["missing_count"].to_numpy()),
            "Window missingness or eligibility mismatch.")
    return result.reset_index(drop=True)


def load_history(customer_id: str, root: Path = REPOSITORY_ROOT) -> CustomerHistory:
    """Resolve identity using verified scores, index key, record bytes and windows."""
    try:
        require(isinstance(customer_id, str) and bool(customer_id), "A scored customer identifier is required.")
        scores = load_predictions(root)
        require(scores.frame is not None, f"Frozen scores unavailable; history alignment cannot be established. {scores.message}")
        selected = scores.frame.loc[scores.frame["CONS_NO"].eq(customer_id)]
        require(len(selected) == 1, "Customer is not in the verified final evaluation cohort.")
        label = int(selected.iloc[0]["FLAG"])
        catalog = _json((root / CONSUMPTION_CATALOG).read_bytes())
        require(catalog["schema_version"] == 1, "Unsupported consumption catalog.")
        provenance = _json(_verified(root, catalog["provenance"]))
        evidence = load_evidence(root)
        require(provenance["status"] == "VERIFIED" and provenance["representation"] == "original_source_readings"
                and provenance["customers"] == evidence.customers
                and provenance["score_artifact_sha256"] == evidence.catalog["predictions"]["sha256"]
                and provenance["customer_membership_sha256"] == evidence.integrity["customer_membership_sha256"],
                "Consumption lineage does not match Stage 1B scores.")
        require(provenance["source"]["sha256"] == catalog["source"]["sha256"]
                and provenance["source"]["path"] == catalog["source"]["path"]
                and provenance["index"]["path"] == catalog["index"]["path"]
                and provenance["index"]["sha256"] == catalog["index"]["sha256"], "Consumption source/index binding mismatch.")
        raw_path = verify_file(root, catalog["source"])
        verify_file(root, catalog["test_mask"])
        require(catalog["windows"] == provenance["windows"] and catalog["test_mask"] == provenance["test_mask"],
                "Consumption mask/window lineage mismatch.")
        index = _csv(_verified(root, catalog["index"]))
        require(set(index.columns) == {"CONS_NO", "FLAG", "byte_offset", "byte_length", "record_sha256",
                                       "observed", "missing", "zero_readings", "usable_windows"}
                and len(index) == evidence.customers and not index["CONS_NO"].duplicated().any()
                and set(index["CONS_NO"]) == set(scores.frame["CONS_NO"]), "History index schema or customer membership mismatch.")
        row = index.loc[index["CONS_NO"].eq(customer_id)].iloc[0]
        require(row["FLAG"] == str(label), "History index evaluation label mismatch.")
        offset, length = int(row["byte_offset"]), int(row["byte_length"])
        require(offset >= provenance["header_bytes"] and 0 < length <= 1024 * 1024
                and offset + length <= raw_path.stat().st_size, "History index byte bounds invalid.")
        with raw_path.open("rb") as handle:
            handle.seek(offset)
            content = handle.read(length)
        require(hashlib.sha256(content).hexdigest() == row["record_sha256"], "Selected history record SHA-256 mismatch.")
        readings = parse_record(content, provenance["source_headers"], customer_id, label)
        require(len(readings) == provenance["calendar_days"]
                and readings.date.min().strftime("%Y-%m-%d") == provenance["period_start"]
                and readings.date.max().strftime("%Y-%m-%d") == provenance["period_end"], "Source calendar mismatch.")
        stats = summarize(readings)
        require(all(stats[name] == int(row[name]) for name in ("observed", "missing", "zero_readings")),
                "History coverage disagrees with the validated index.")
        # Parquet must be hashed as exact binary bytes, never text-normalized.
        window_path = verify_file(root, catalog["windows"])
        window_bytes = window_path.read_bytes()
        require(hashlib.sha256(window_bytes).hexdigest() == catalog["windows"]["sha256"],
                "Evaluated-window artifact changed while reading.")
        windows = validate_windows(_customer_windows(window_bytes, customer_id), readings, customer_id, label)
        require(len(windows) == int(row["usable_windows"]), "Evaluated-window count mismatch.")
        return CustomerHistory(customer_id, readings, windows, catalog["source"]["path"],
                               catalog["provenance"]["path"], len(provenance["source_headers"]) - 2)
    except ArtifactError:
        raise
    except (OSError, KeyError, ValueError, TypeError, UnicodeError, IndexError) as exc:
        raise ArtifactError(f"Customer history unavailable: {exc}") from exc
