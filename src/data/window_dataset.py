"""Memory-conscious access to Stage 3C temporal windows.

One processed customer split is held as a contiguous float32 NumPy matrix.
Manifest customer IDs are mapped to integer row references once, and each
``__getitem__`` call performs only an indexed 30-value NumPy slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
import torch
from torch import Tensor
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WINDOWS_DIR = PROJECT_ROOT / "data" / "windows"

ID_COLUMN = "CONS_NO"
LABEL_COLUMN = "FLAG"
WINDOW_LENGTH = 30
SUPPORTED_STRIDES = (30,)
SPLIT_NAMES = ("train", "val", "test")
METADATA_COLUMNS = (
    "window_id",
    ID_COLUMN,
    LABEL_COLUMN,
    "start_feature_index",
    "end_feature_index",
    "missing_fraction",
)


@dataclass(slots=True)
class ProcessedSplitData:
    """Compact scaled-consumption matrix and customer lookup for one split."""

    split: str
    customer_ids: pd.Index
    customer_labels: np.ndarray
    values: np.ndarray
    feature_columns: tuple[str, ...]

    @classmethod
    def load(
        cls, split: str, project_root: Path = PROJECT_ROOT
    ) -> "ProcessedSplitData":
        """Load and validate exactly one processed split."""
        validate_split_name(split)
        path = project_root / "data" / "processed" / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Processed split not found: {path}")

        frame = pd.read_parquet(path, engine="pyarrow")
        if ID_COLUMN not in frame or LABEL_COLUMN not in frame:
            raise ValueError(f"{path} is missing {ID_COLUMN} or {LABEL_COLUMN}")
        feature_columns = tuple(
            column for column in frame.columns if column not in {ID_COLUMN, LABEL_COLUMN}
        )
        dates = pd.to_datetime(feature_columns, format="%m/%d/%Y", errors="coerce")
        if len(feature_columns) < WINDOW_LENGTH or dates.isna().any():
            raise ValueError(f"{path} does not have valid date-named consumption columns")
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError(f"{path} consumption columns are not uniquely chronological")

        ids = pd.Index(frame[ID_COLUMN].astype(str), name=ID_COLUMN)
        if ids.hasnans or not ids.is_unique:
            raise ValueError(f"{path} customer IDs must be non-missing and unique")
        labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8, copy=True)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{path} FLAG must be binary")
        values = np.ascontiguousarray(
            frame.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=False)
        )
        if values.shape != (len(ids), len(feature_columns)):
            raise ValueError(f"{path} consumption matrix has an unexpected shape")
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite scaled consumption")
        return cls(split, ids, labels, values, feature_columns)

    @property
    def memory_bytes(self) -> int:
        """Return the primary NumPy storage footprint."""
        return int(self.values.nbytes + self.customer_labels.nbytes)


class WindowDataset(Dataset[Tensor]):
    """Lazy fixed-length windows backed by one compact processed split."""

    def __init__(
        self,
        split: str,
        flag: int,
        *,
        stride: int = 30,
        store: ProcessedSplitData | None = None,
        include_metadata: bool = False,
        max_windows: int | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        validate_split_name(split)
        if stride not in SUPPORTED_STRIDES:
            raise ValueError(f"Supported Stage 4 strides: {SUPPORTED_STRIDES}; got {stride}")
        if flag not in (0, 1):
            raise ValueError(f"FLAG filter must be 0 or 1; got {flag}")
        if max_windows is not None and max_windows <= 0:
            raise ValueError("max_windows must be positive when supplied")

        self.split = split
        self.flag = flag
        self.stride = stride
        self.store = store or ProcessedSplitData.load(split, project_root)
        if self.store.split != split:
            raise ValueError(
                f"Processed store is for {self.store.split}, not requested split {split}"
            )

        manifest_path = (
            project_root
            / "data"
            / "windows"
            / f"stride_{stride}"
            / "usable"
            / f"{split}_windows.parquet"
        )
        manifest = load_filtered_manifest(manifest_path, flag, max_windows)
        if manifest.empty:
            raise ValueError(f"No FLAG={flag} windows found in {manifest_path}")

        manifest_ids = pd.Index(manifest[ID_COLUMN].astype(str))
        customer_rows = self.store.customer_ids.get_indexer(manifest_ids)
        if np.any(customer_rows < 0):
            unknown = manifest_ids[customer_rows < 0].unique().tolist()[:5]
            raise ValueError(f"Manifest references unknown {split} customer IDs: {unknown}")

        manifest_labels = manifest[LABEL_COLUMN].to_numpy(dtype=np.int8, copy=False)
        if not np.all(manifest_labels == flag):
            raise ValueError(f"{split} Dataset contains FLAG values other than {flag}")
        if not np.array_equal(self.store.customer_labels[customer_rows], manifest_labels):
            raise ValueError(f"{split} manifest labels disagree with processed customer labels")

        starts = manifest["start_feature_index"].to_numpy(dtype=np.int32, copy=True)
        ends = manifest["end_feature_index"].to_numpy(dtype=np.int32, copy=True)
        if np.any(starts < 0) or np.any(ends >= self.store.values.shape[1]):
            raise IndexError(f"{split} manifest contains out-of-range window indices")
        if not np.all(ends - starts + 1 == WINDOW_LENGTH):
            raise ValueError(f"{split} manifest contains non-{WINDOW_LENGTH}-value windows")
        missing_fractions = manifest["missing_fraction"].to_numpy(
            dtype=np.float32, copy=False
        )
        if not np.isfinite(missing_fractions).all() or np.any(missing_fractions >= 1.0):
            raise ValueError(f"{split} manifest is not a valid Stage 3C usable subset")

        self._customer_rows = customer_rows.astype(np.int32, copy=False)
        self._starts = starts
        self._metadata = (
            manifest.loc[
                :, ["window_id", ID_COLUMN, LABEL_COLUMN, "missing_fraction"]
            ].reset_index(drop=True)
            if include_metadata
            else None
        )

    def __len__(self) -> int:
        return int(self._starts.size)

    def __getitem__(self, index: int) -> Tensor:
        row = int(self._customer_rows[index])
        start = int(self._starts[index])
        values = self.store.values[row, start : start + WINDOW_LENGTH]
        if values.shape != (WINDOW_LENGTH,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid window encountered at Dataset index {index}")
        return torch.from_numpy(values)

    @property
    def metadata(self) -> pd.DataFrame:
        """Return reconstruction-error metadata without exposing it to the model."""
        if self._metadata is None:
            raise RuntimeError("Dataset was created with include_metadata=False")
        return self._metadata.copy()

    @property
    def reference_memory_bytes(self) -> int:
        """Return compact integer-reference storage, excluding optional metadata."""
        return int(self._customer_rows.nbytes + self._starts.nbytes)


def load_filtered_manifest(
    path: Path, flag: int, max_windows: int | None = None
) -> pd.DataFrame:
    """Read only needed manifest fields and one requested FLAG population."""
    if not path.is_file():
        raise FileNotFoundError(f"Usable window manifest not found: {path}")
    dataset = arrow_dataset.dataset(path, format="parquet")
    scanner = dataset.scanner(
        columns=list(METADATA_COLUMNS),
        filter=arrow_dataset.field(LABEL_COLUMN) == flag,
    )
    table = scanner.head(max_windows) if max_windows is not None else scanner.to_table()
    return table.to_pandas()


def validate_split_name(split: str) -> None:
    """Reject paths outside the three authoritative Stage 2 splits."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"Split must be one of {SPLIT_NAMES}; got {split!r}")
