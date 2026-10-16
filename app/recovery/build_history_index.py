"""Index original held-out histories without copying readings or changing data."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pyarrow.parquet as pq

from app.config import REPOSITORY_ROOT
from app.consumption_loader import parse_record, require, summarize, validate_windows
from app.data_loader import load_evidence, load_predictions


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def record(relative: str) -> dict:
    return {"path": relative, "sha256": digest(REPOSITORY_ROOT / relative)}


def main() -> None:
    root = REPOSITORY_ROOT
    output = root / "app/derived/stage2"
    require(not (output / "history_index.csv").exists() and not (output / "history_provenance.json").exists(),
            "History index builder does not overwrite derived artifacts.")
    baseline = json.loads((output / "research_snapshot.json").read_text())
    require(all(digest(root / path) == value for path, value in baseline.items()), "Protected baseline differs before indexing.")
    scores = load_predictions()
    require(scores.frame is not None, scores.message)
    expected = scores.frame.set_index("CONS_NO")
    evidence = load_evidence()
    historic = json.loads((root / "results/cnn_autoencoder/stride_30/test_evaluation/test_metrics.json").read_text())["protected_pretest_sha256"]
    paths = {"source": "data/raw/data set.csv", "test_mask": "data/processed/test_mask.parquet",
             "windows": "data/windows/stride_30/usable/test_windows.parquet", "processed": "data/processed/test.parquet",
             "split": "data/processed/split_manifest.parquet"}
    for path in paths.values():
        require(digest(root / path) == historic[path], f"Historical input hash mismatch: {path}")
    mask = pd.read_parquet(root / paths["test_mask"]).set_index("CONS_NO")
    split = pd.read_parquet(root / paths["split"])
    final = split.loc[split["split"].eq("test")].set_index("CONS_NO")
    require(set(mask.index) == set(final.index) == set(expected.index), "History/test/mask/scored membership mismatch.")
    require(final.loc[expected.index, "FLAG"].tolist() == expected["FLAG"].tolist(), "Historical test labels disagree.")
    model_features = list(mask.columns)
    require(model_features == [name for name in pq.ParquetFile(root / paths["processed"]).schema_arrow.names if name not in {"CONS_NO", "FLAG"}],
            "Processed calendar and frozen mask differ.")
    windows = pd.read_parquet(root / paths["windows"])
    require(set(windows["CONS_NO"]) == set(expected.index) and len(windows) == evidence.integrity["eligible_windows"],
            "Evaluated-window membership mismatch.")
    grouped = {customer: group for customer, group in windows.groupby("CONS_NO", sort=False)}
    rows = []
    seen = set()
    with (root / paths["source"]).open("rb") as handle:
        header = handle.readline()
        headers = next(csv.reader([header.decode("utf-8-sig")], strict=True))
        id_index, label_index = headers.index("CONS_NO"), headers.index("FLAG")
        while True:
            offset = handle.tell()
            content = handle.readline()
            if not content:
                break
            fields = next(csv.reader([content.decode("utf-8")], strict=True))
            require(len(fields) == len(headers), "Source requires a single physical CSV line per customer.")
            customer = fields[id_index]
            require(customer not in seen, "Duplicate raw customer identifiers.")
            seen.add(customer)
            if customer not in expected.index:
                continue
            label = int(expected.loc[customer, "FLAG"])
            require(fields[label_index] == str(label), "Raw/scored label mismatch.")
            readings = parse_record(content, headers, customer, label)
            require(len(readings) == 1035 and readings.loc[~readings.source_column_present, "date"].tolist()
                    == [pd.Timestamp("2016-09-18")], "Raw source calendar is not the frozen calendar.")
            processed_days = readings.loc[readings.model_calendar]
            require(pd.to_datetime(model_features, format="%m/%d/%Y").tolist() == processed_days.date.tolist(),
                    "Original/model calendar alignment failed.")
            require(processed_days.missing.astype("uint8").tolist() == mask.loc[customer].tolist(),
                    "Original missing readings disagree with the frozen mask.")
            evaluated = validate_windows(grouped[customer], readings, customer, label)
            stats = summarize(readings)
            rows.append({"CONS_NO": customer, "FLAG": label, "byte_offset": offset, "byte_length": len(content),
                         "record_sha256": hashlib.sha256(content).hexdigest(),
                         **{key: stats[key] for key in ("observed", "missing", "zero_readings")},
                         "usable_windows": len(evaluated)})
    require(len(rows) == evidence.customers, "Not all scored histories were indexed.")
    require(all(digest(root / path) == value for path, value in baseline.items()), "Protected artifacts changed during indexing.")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("CONS_NO").to_csv(output / "history_index.csv", index=False, lineterminator="\n")
    relative = "app/derived/stage2"
    provenance = {"schema_version": 1, "status": "VERIFIED", "representation": "original_source_readings",
                  "source": record(paths["source"]), "index": record(f"{relative}/history_index.csv"),
                  "test_mask": record(paths["test_mask"]), "windows": record(paths["windows"]),
                  "processed": record(paths["processed"]), "split": record(paths["split"]),
                  "governance": evidence.catalog["artifacts"]["governance"],
                  "score_artifact_sha256": evidence.catalog["predictions"]["sha256"],
                  "customer_membership_sha256": evidence.integrity["customer_membership_sha256"],
                  "customers": len(rows), "raw_customers_checked": len(seen), "calendar_days": 1035,
                  "period_start": "2014-01-01", "period_end": "2016-10-31", "header_bytes": len(header),
                  "source_headers": headers, "measurement_unit": None,
                  "missingness": "Original blank/NA reading; absent 2016-09-18 source column explicitly missing. Zero is observed.",
                  "model_calendar": "Exclude 2014-10-03; include inserted 2016-09-18 as missing; 34 candidate windows of 30 positions, 14 trailing positions unused; eligibility missing_count<30.",
                  "all_customer_masks_and_windows_validated": True, "row_mapping": "Exact CONS_NO join plus byte offset, length and per-record SHA-256; FLAG checked against final scores and historical test split.",
                  "protected_files_unchanged": len(baseline), "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "builder": record("app/recovery/build_history_index.py"),
                  "command": ".\\.venv\\Scripts\\python.exe -m app.recovery.build_history_index",
                  "output_policy": "Metadata only; original consumption values are not duplicated."}
    (output / "history_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    catalog = {"schema_version": 1, **{key: provenance[key] for key in ("source", "index", "test_mask", "windows")},
               "provenance": record(f"{relative}/history_provenance.json")}
    (root / "app/consumption_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"histories": len(rows), "raw_customers_checked": len(seen), "index": provenance["index"],
                      "protected_files_unchanged": len(baseline)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
