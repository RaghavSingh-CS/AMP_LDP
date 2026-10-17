"""Reproducible presenter selections and aggregate-only local export."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from app.config import COHORT_NAME, MODEL_ID, REPOSITORY_ROOT
from app.data_loader import ArtifactError, _json, _read, _require, load_evidence, load_predictions
from app.performance_loader import AP_DEFINITION, CurveResult, PerformanceEvidence

SELECTIONS = {
    "high_theft": (1, True, "High-ranked theft-labelled", "Smallest global rank among FLAG=1 customers"),
    "low_normal": (0, False, "Low-ranked normal-labelled", "Largest global rank among FLAG=0 customers"),
    "high_normal": (0, True, "Difficult case: high-ranked normal-labelled", "Smallest global rank among FLAG=0 customers"),
}


@dataclass(frozen=True)
class DemoEvidence:
    examples: pd.DataFrame
    catalog: dict


def select_examples(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (label, ascending, title, criterion) in SELECTIONS.items():
        selected = frame.loc[frame.FLAG.eq(label)].sort_values("rank", ascending=ascending)
        _require(not selected.empty, f"No verified customer for presenter criterion: {criterion}")
        rows.append({"key": key, "title": title, "criterion": criterion, **selected.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def load_demo(root: Path = REPOSITORY_ROOT) -> DemoEvidence:
    try:
        catalog = _json(_read(root, "app/demo_catalog.json"))
        evidence = load_evidence(root)
        source = evidence.catalog["predictions"]
        _require(catalog["schema_version"] == 1 and catalog["model_id"] == MODEL_ID
                 and catalog["source_scores"] == {"path": source["path"], "sha256": source["sha256"]}
                 and catalog["source_provenance"] == source["provenance"]
                 and catalog["selection_uses_evaluation_labels"] is True
                 and catalog["inference_inputs_changed"] is False,
                 "Presenter-selection provenance mismatch.")
        result = load_predictions(root)
        _require(result.frame is not None, result.message)
        selected = select_examples(result.frame)
        _require(catalog["examples"] == selected[["key", "CONS_NO", "criterion"]].to_dict("records"),
                 "Presenter identifiers or selection rules disagree with the frozen rankings.")
        return DemoEvidence(selected, catalog)
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, IndexError) as exc:
        raise ArtifactError(f"Presenter examples unavailable: {exc}") from exc


def aggregate_summary(evidence: PerformanceEvidence, curves: CurveResult) -> bytes:
    """Allow-list aggregate fields; never serialize a customer frame or identifier."""
    frozen = evidence.frozen
    source = frozen.catalog["predictions"]
    payload = {
        "schema_version": 1,
        "title": "AMP frozen customer-level evaluation summary",
        "evidence_commit": frozen.catalog["evidence_commit"],
        "cohort": {"name": COHORT_NAME, "dataset": "SGCC", "role": "FINAL_TEST",
                   "customers": frozen.customers, "theft_labelled": frozen.theft_customers,
                   "prevalence": evidence.prevalence, "positive_label": "FLAG=1",
                   "historical_access_disclosure": frozen.disclosure},
        "metric_definitions": {"ROC_AUC": "Customer ROC-AUC; sklearn.metrics.roc_auc_score",
                               "average_precision": AP_DEFINITION},
        "metrics": evidence.comparison.to_dict("records"),
        "training_population": "DEV_TRAIN only",
        "score_context": {"origin": source["origin"], "original_persisted_stage27b_predictions": False,
                          "description": "Explorer displays validated scores reconstructed once by frozen inference replay in Stage 1B; selecting a customer runs no inference."},
        "curves": {"AMP-LDP": "verified derived coordinates" if curves.provenance else "unavailable",
                   "FULL_48": "unavailable; aggregate metrics only", "DIRECT": "unavailable; aggregate metrics only",
                   "AP_is_trapezoidal_PR_area": False},
        "threshold": "No applicable frozen AMP-LDP binary operating rule; ranking only",
        "limits": ["Scores are not established calibrated theft probabilities or proof of theft.",
                   "No exact theft-day localization or causal attribution is established.",
                   "Independent external utility and operational generalization are not established.",
                   "Illustrative presenter customers and all customer identifiers/histories are excluded from this export."],
        "sources": {**{key: frozen.catalog["artifacts"][key] for key in ("metrics", "integrity", "model", "protocol", "access")},
                    "comparators": evidence.catalog["comparators"], "score_provenance": source["provenance"]},
        "hash_policy": "SHA-256 over UTF-8 text with CRLF-to-LF normalization; derived evidence stored as LF bytes",
        "sharing": "Local aggregate presentation export only; no customer-level sharing permission is asserted.",
    }
    if curves.provenance:
        payload["sources"]["curve_provenance"] = evidence.catalog["curves"]
        payload["sources"]["curve_coordinates"] = curves.provenance["coordinates"]
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
