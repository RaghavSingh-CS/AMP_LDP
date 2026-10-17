"""Read and validate frozen evidence. Never fit, score, recompute metrics or write.

Parsed immutable bytes are cached, while file existence and SHA-256 are checked
on every load. Cache hits cannot mask a missing or modified artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import (
    CATALOG_PATH, COHORT_NAME, MAX_ARTIFACT_BYTES, MODEL_ID, REPOSITORY_ROOT, SEEDS,
)


class ArtifactError(ValueError):
    """Missing, malformed or unverified interface evidence."""


@dataclass(frozen=True)
class FrozenEvidence:
    customers: int
    theft_customers: int
    roc_auc: float
    average_precision: float
    model: dict[str, Any]
    integrity: dict[str, Any]
    disclosure: str
    catalog: dict[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    frame: pd.DataFrame | None
    source: str | None
    message: str
    provenance_source: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()),
             "Artifact path must stay inside the repository.")
    return path


def _read(root: Path, relative: str) -> bytes:
    path = _path(root, relative)
    try:
        _require(path.stat().st_size <= MAX_ARTIFACT_BYTES, f"Artifact is too large: {relative}")
        return path.read_bytes().replace(b"\r\n", b"\n")
    except OSError as exc:
        raise ArtifactError(f"Cannot read artifact: {relative} ({exc.strerror})") from exc


def _verified(root: Path, record: dict[str, Any]) -> bytes:
    content = _read(root, record["path"])
    _require(hashlib.sha256(content).hexdigest() == record["sha256"],
             f"SHA-256 mismatch: {record['path']}. Frozen evidence is unavailable.")
    return content


@lru_cache(maxsize=16)
def _json(content: bytes) -> dict[str, Any]:
    value = json.loads(content.decode("utf-8"))
    _require(isinstance(value, dict), "Expected a JSON object.")
    return value


@lru_cache(maxsize=8)
def _csv(content: bytes) -> pd.DataFrame:
    # String-first parsing preserves identifiers, including leading zeros and NA.
    return pd.read_csv(StringIO(content.decode("utf-8")), dtype=str, keep_default_na=False)


def _catalog(root: Path) -> dict[str, Any]:
    catalog = _json(_read(root, CATALOG_PATH.as_posix()))
    _require(catalog["schema_version"] == 1, "Unsupported artifact catalog version.")
    return catalog


def load_evidence(root: Path = REPOSITORY_ROOT) -> FrozenEvidence:
    """Only validated aggregate artifacts supply dashboard metrics and counts."""
    try:
        catalog = _catalog(root)
        records = catalog["artifacts"]
        model, protocol, integrity, access = (
            _json(_verified(root, records[name]))
            for name in ("model", "protocol", "integrity", "access")
        )
        metrics = _csv(_verified(root, records["metrics"]))
        _require(set(metrics.columns) == {"model", "customers", "ROC_AUC", "average_precision"},
                 "Invalid frozen metric columns.")
        _require(len(metrics) == 3 and set(metrics["model"]) == {"LATENT", "FULL_48", "DIRECT"},
                 "Missing or duplicate final model metric rows.")
        customers = integrity["eligible_customers"]
        theft = integrity["theft_customers"]
        _require(type(customers) is int and type(theft) is int and 0 < theft < customers,
                 "Invalid final cohort counts.")
        cohort = protocol["cohort"]
        _require(customers == integrity["governed_customers"] == cohort["expected_customers_from_committed_aggregate_governance"]
                 and theft == cohort["expected_theft_customers"]
                 and integrity["normal_customers"] + theft == customers
                 and integrity["exclusions"] == 0, "Frozen cohort evidence disagrees.")
        _require(cohort["governance_role"] == "FINAL_TEST" and cohort["historical_split"] == "test"
                 and integrity["cohort_terminology"] == access["cohort_terminology"] == COHORT_NAME,
                 "Incorrect final evaluation cohort provenance.")
        _require(access["authorized_by_preregistration_sha256"] == records["protocol"]["sha256"]
                 and access["stage27b_access_count"] == 1, "Invalid terminal access evidence.")
        _require(protocol["primary_model"]["id"] == MODEL_ID
                 and model["seeds"] == SEEDS
                 and model["training_population"] == "DEV_TRAIN only"
                 and model["method"]["latent_representation_dimension"] == 96
                 and model["model_development_closed"] is True
                 and model["refitting_permitted"] is False,
                 "Frozen AMP-LDP model metadata does not match the interface contract.")
        values = metrics[["ROC_AUC", "average_precision"]].astype(float).to_numpy()
        _require(bool(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()),
                 "Invalid frozen ranking metrics.")
        _require(metrics["customers"].eq(str(customers)).all(), "Metric cohort count mismatch.")
        row = metrics.loc[metrics["model"].eq("LATENT")].iloc[0]
        return FrozenEvidence(customers, theft, float(row["ROC_AUC"]),
                              float(row["average_precision"]), model, integrity,
                              access["disclosure"], catalog)
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, IndexError) as exc:
        raise ArtifactError(f"Malformed frozen evidence: {exc}") from exc


def stable_hash(values: list[str]) -> str:
    """Same ordered UTF-8 newline join used by the frozen Stage 27B evaluator."""
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def validate_predictions(
    frame: pd.DataFrame, governance: pd.DataFrame, *, customers: int,
    theft_customers: int, membership_sha256: str, labels_sha256: str,
) -> pd.DataFrame:
    """Validate a future canonical export before sorting. No metric calculation.

    Input order must match Stage 27B's persisted membership and label hashes.
    Labels are evaluation annotations only. Ranking is deterministic, ordinal,
    with original ID as tie-breaker; filtering never changes the global rank.
    """
    _require(set(frame.columns) == {"CONS_NO", "FLAG", "risk_score"},
             "Prediction columns must be CONS_NO, FLAG, risk_score.")
    result = frame.copy()
    ids = result["CONS_NO"]
    _require(ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
             "Customer identifiers must be nonempty original strings.")
    _require(not ids.duplicated().any(), "Duplicate customer identifiers.")
    _require(len(result) == customers, "Prediction cohort size mismatch.")
    _require({"CONS_NO", "research_role", "original_split"}.issubset(governance.columns),
             "Missing governance columns.")
    _require(not governance["CONS_NO"].duplicated().any(), "Duplicate governance identifiers.")
    final = governance.loc[governance["research_role"].eq("FINAL_TEST")]
    _require(final["original_split"].eq("test").all()
             and set(ids) == set(final["CONS_NO"]), "Incorrect FINAL_TEST split membership.")
    _require(stable_hash(ids.tolist()) == membership_sha256, "Frozen membership order/hash mismatch.")
    labels = result["FLAG"].astype(str)
    _require(labels.isin(["0", "1"]).all(), "Invalid evaluation labels.")
    _require(int(labels.eq("1").sum()) == theft_customers
             and stable_hash(labels.tolist()) == labels_sha256, "Frozen evaluation label hash/count mismatch.")
    try:
        # Python's correctly rounded float parser preserves the Stage 1B
        # 17-digit CSV roundtrip; pandas.to_numeric can drop final digits.
        scores = result["risk_score"].map(float).astype(float)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("Non-numeric risk scores.") from exc
    _require(bool(np.isfinite(scores).all() and scores.between(0, 1).all()),
             "Risk scores must be finite and within [0, 1].")
    result["risk_score"] = scores
    result["FLAG"] = labels.astype(int)
    result = result.sort_values(["risk_score", "CONS_NO"], ascending=[False, True], kind="stable")
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def load_predictions(root: Path = REPOSITORY_ROOT) -> PredictionResult:
    """No filename guessing or fallback to predictions from another model/split."""
    try:
        evidence = load_evidence(root)
        record = evidence.catalog["predictions"]
        if record is None:
            return PredictionResult(None, None, evidence.catalog["prediction_unavailable_reason"])
        _require(record["model_id"] == MODEL_ID and record["source_stage"] == "27B"
                 and record["research_role"] == "FINAL_TEST" and record["seeds"] == SEEDS
                 and record["ensemble"] == "arithmetic_mean"
                 and record["model_freeze_sha256"] == evidence.catalog["artifacts"]["model"]["sha256"],
                 "Prediction provenance does not match frozen AMP-LDP.")
        description = "Verified frozen evaluation scores."
        if record.get("origin") == "reconstructed_inference_only":
            provenance = _json(_verified(root, record["provenance"]))
            _validate_reconstruction(provenance, record, evidence)
            description = ("Derived interface scores reconstructed by frozen inference replay in Interface Stage 1B. "
                           "These are not original persisted Stage 27B predictions. "
                           "Independent ROC-AUC and AP checks match the frozen evaluation to six decimals.")
        else:
            _require(record.get("origin") in (None, "recovered_original"), "Unsupported score artifact origin.")
        frame = _csv(_verified(root, record))
        governance = _csv(_verified(root, evidence.catalog["artifacts"]["governance"]))
        validated = validate_predictions(
            frame, governance, customers=evidence.customers, theft_customers=evidence.theft_customers,
            membership_sha256=evidence.integrity["customer_membership_sha256"],
            labels_sha256=evidence.integrity["labels_sha256"],
        )
        return PredictionResult(validated, record["path"], description,
                                record.get("provenance", {}).get("path"))
    except (ArtifactError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        return PredictionResult(None, None, f"Customer scores unavailable: {exc}")


def _validate_reconstruction(provenance: dict, record: dict, evidence: FrozenEvidence) -> None:
    """Validate the reviewed replay manifest without loading a model at runtime."""
    _require(provenance["schema_version"] == 1 and provenance["status"] == "VERIFIED"
             and provenance["origin"] == "reconstructed_inference_only"
             and provenance["original_stage27b_persisted_predictions"] is False
             and provenance["model_id"] == MODEL_ID
             and provenance["source_protocol_stage"] == "27B"
             and provenance["source_evidence_commit"] == evidence.catalog["evidence_commit"]
             and provenance["model_freeze_sha256"] == record["model_freeze_sha256"]
             and provenance["protocol_sha256"] == evidence.catalog["artifacts"]["protocol"]["sha256"],
             "Reconstruction provenance identity mismatch.")
    output = provenance["output"]
    _require(output["path"] == record["path"] and output["sha256"] == record["sha256"]
             and output["customers"] == evidence.customers and output["theft_customers"] == evidence.theft_customers,
             "Reconstruction output binding mismatch.")
    scoring = provenance["scoring"]
    _require(scoring["seeds"] == SEEDS and scoring["seed_order"] == SEEDS
             and scoring["ensemble"] == "arithmetic_mean" and scoring["class_column"] == 1
             and scoring["fit_calls"] == scoring["backpropagation_calls"] == 0
             and scoring["model_mode"] == "eval" and provenance["no_threshold_changes"] is True
             and provenance["no_training_or_validation_features_loaded"] is True
             and provenance["research_integrity"]["unchanged"] is True,
             "Reconstruction violates frozen inference boundaries.")
    for key in ("customer_membership_sha256", "window_membership_sha256", "labels_sha256"):
        _require(provenance["cohort_integrity"][key] == evidence.integrity[key], "Reconstruction cohort hash mismatch.")
    verification = provenance["verification"]
    _require(verification["agreement"] is True and verification["required_decimal_places"] == 6,
             "Reconstruction metric verification missing.")
    for name, expected in (("ROC_AUC", evidence.roc_auc), ("average_precision", evidence.average_precision)):
        observed = float(verification["observed"][name])
        _require(np.isfinite(observed) and f"{observed:.6f}" == f"{expected:.6f}"
                 and verification["expected"][name] == expected,
                 f"Reconstruction metric mismatch: {name}.")


def explore(frame: pd.DataFrame, query: str, order: str) -> pd.DataFrame:
    """Literal, case-insensitive ID search; original identifiers stay unchanged."""
    result = frame.loc[frame["CONS_NO"].str.contains(query.strip(), case=False, regex=False)]
    if order == "Lowest risk first":
        return result.sort_values(["risk_score", "CONS_NO"], ascending=[True, True])
    if order == "Customer identifier":
        return result.sort_values("CONS_NO")
    return result.sort_values("rank")
