"""Validated presentation evidence; no inference, fitting or sklearn at runtime."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import COHORT_NAME, REPOSITORY_ROOT
from app.data_loader import ArtifactError, FrozenEvidence, _csv, _json, _read, _require, _verified, load_evidence, load_predictions

MODEL_ORDER = ["AMP-LDP", "FULL_48", "DIRECT"]
METRICS = {"ROC_AUC": "ROC-AUC", "average_precision": "Average Precision"}
AP_DEFINITION = "sklearn.metrics.average_precision_score; non-interpolated recall-increment weighted precision"


@dataclass(frozen=True)
class PerformanceEvidence:
    frozen: FrozenEvidence
    comparison: pd.DataFrame
    prevalence: float
    catalog: dict


@dataclass(frozen=True)
class CurveResult:
    roc: pd.DataFrame | None
    pr: pd.DataFrame | None
    provenance: dict | None
    message: str


def load_performance(root: Path = REPOSITORY_ROOT) -> PerformanceEvidence:
    try:
        frozen = load_evidence(root)
        catalog = _json(_read(root, "app/performance_catalog.json"))
        _require(catalog["schema_version"] == 1 and catalog["evidence_commit"] == frozen.catalog["evidence_commit"],
                 "Performance catalog identity mismatch.")
        _require(catalog["comparator_curves"] == {"FULL_48": None, "DIRECT": None},
                 "Unreviewed comparator curve registration.")
        records = frozen.catalog["artifacts"]
        protocol = _json(_verified(root, records["protocol"]))
        comparators = _json(_verified(root, catalog["comparators"]))
        # Bind metric definitions and the 30-day method description to reviewed source.
        _verified(root, catalog["evaluator"])
        _verified(root, catalog["method_document"])
        _verified(root, catalog["historical_architectures"])
        fairness = comparators["fairness_assessment"]
        _require(fairness["symmetric_training_population"] is True
                 and protocol["refit_decision"]["training_population_for_all_primary_models"] == "DEV_TRAIN only"
                 and protocol["metrics"]["primary"] == ["customer ROC-AUC", "customer average precision"]
                 and protocol["metrics"]["threshold_optimization"] is False
                 and protocol["cohort"]["required_name"] == COHORT_NAME
                 and protocol["cohort"]["new_exclusions"] is False,
                 "Comparators do not share the frozen customer evaluation protocol.")
        definitions = comparators["comparators"]
        planned = protocol["comparators"]
        _require(len(planned) == 2 and {c["id"] for c in planned} == set(definitions),
                 "Comparator identities mismatch.")
        for item in planned:
            definition = definitions[item["id"]]
            _require(item["model_sha256"] == definition["model_sha256"]
                     and item["training_population"] == definition["training_population"] == "DEV_TRAIN only"
                     and item["definition_source"] == catalog["comparators"]["path"],
                     "Comparator checkpoint or training population mismatch.")
        _require(definitions["FULL_48_HGB"]["feature_dimension"] == 48
                 and definitions["DIRECT_SUMMARY_DISTRIBUTION_HGB"]["feature_dimension"] == 102
                 and frozen.model["method"]["encoder"] == ["Linear(17,32)", "ReLU", "Dropout(0.1)", "Linear(32,16)", "ReLU"]
                 and len(frozen.model["method"]["distribution_statistics_in_order"]) == 6,
                 "Frozen representation contract mismatch.")
        prevalence = frozen.theft_customers / frozen.customers
        _require(prevalence == frozen.integrity["prevalence"], "Prevalence metadata mismatch.")
        frame = _csv(_verified(root, records["metrics"])).copy()
        frame["model"] = frame.model.replace({"LATENT": "AMP-LDP"})
        frame = frame.set_index("model").loc[MODEL_ORDER].reset_index()
        frame["customers"] = frame.customers.astype(int)
        for metric in METRICS:
            frame[metric] = frame[metric].map(float)
        return PerformanceEvidence(frozen, frame, prevalence, catalog)
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, IndexError, OSError) as exc:
        raise ArtifactError(f"Malformed performance evidence: {exc}") from exc


def absolute_differences(evidence: PerformanceEvidence) -> pd.DataFrame:
    frame = evidence.comparison.set_index("model")
    return pd.DataFrame([{"comparison": f"AMP-LDP − {baseline}",
                          **{key: float(frame.loc["AMP-LDP", key] - frame.loc[baseline, key]) for key in METRICS}}
                         for baseline in ("FULL_48", "DIRECT")])


def load_curves(evidence: PerformanceEvidence, root: Path = REPOSITORY_ROOT) -> CurveResult:
    """Failure of optional curves never substitutes another cohort/model's curve."""
    try:
        record = evidence.catalog["curves"]
        _require(record is not None, "No verified AMP-LDP curve artifact is registered.")
        provenance = _json(_verified(root, record))
        predictions = load_predictions(root)
        _require(predictions.frame is not None, predictions.message)
        source = evidence.frozen.catalog["predictions"]
        _require(provenance["schema_version"] == 1 and provenance["status"] == "VERIFIED"
                 and provenance["source_scores"] == {"path": source["path"], "sha256": source["sha256"]}
                 and provenance["source_provenance"] == source["provenance"]
                 and provenance["score_origin"] == "reconstructed_inference_only"
                 and provenance["model_id"] == source["model_id"]
                 and provenance["protocol_sha256"] == evidence.frozen.catalog["artifacts"]["protocol"]["sha256"]
                 and provenance["metric_source"] == evidence.frozen.catalog["artifacts"]["metrics"]
                 and provenance["customers"] == evidence.frozen.customers
                 and provenance["theft_customers"] == evidence.frozen.theft_customers,
                 "Curve source/model identity mismatch.")
        for key in ("customer_membership_sha256", "labels_sha256"):
            _require(provenance[key] == evidence.frozen.integrity[key], "Curve label/membership alignment mismatch.")
        _require(provenance["AP_definition"] == AP_DEFINITION
                 and provenance["ROC_definition"] == "sklearn.metrics.roc_auc_score"
                 and provenance["curve_functions"] == ["sklearn.metrics.roc_curve", "sklearn.metrics.precision_recall_curve"]
                 and provenance["no_inference_or_fitting"] is True
                 and provenance["no_operating_threshold_selected"] is True,
                 "Curve calculation policy mismatch.")
        for key, expected in (("ROC_AUC", evidence.frozen.roc_auc), ("average_precision", evidence.frozen.average_precision)):
            _require(provenance["verified_metrics"][key] == expected, "Curve frozen metric mismatch.")
        coordinates = _json(_verified(root, provenance["coordinates"]))
        _require(coordinates["schema_version"] == 1 and set(coordinates) == {"schema_version", "roc", "pr"},
                 "Invalid curve coordinate schema.")
        roc = pd.DataFrame(coordinates["roc"], columns=["false_positive_rate", "true_positive_rate"])
        pr = pd.DataFrame(coordinates["pr"], columns=["recall", "precision"])
        for frame in (roc, pr):
            values = frame.to_numpy(dtype=float)
            _require(len(frame) >= 2 and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                     "Invalid curve coordinates.")
            _require(frame.iloc[:, 0].is_monotonic_increasing, "Curve coordinates out of order.")
        _require(roc.true_positive_rate.is_monotonic_increasing
                 and roc.iloc[0].tolist() == [0, 0] and roc.iloc[-1].tolist() == [1, 1]
                 and pr.iloc[0].tolist() == [0, 1] and pr.iloc[-1].tolist() == [1, evidence.prevalence],
                 "Curve endpoints mismatch.")
        _require(len(roc) == provenance["points"]["roc"] and len(pr) == provenance["points"]["pr"],
                 "Curve point count mismatch.")
        # Geometric validation only; AP uses right endpoint step heights, never trapezoids.
        _require(abs(float(np.trapezoid(roc.true_positive_rate, roc.false_positive_rate)) - evidence.frozen.roc_auc) < 1e-12
                 and abs(float(np.sum(np.diff(pr.recall) * pr.precision.to_numpy()[1:])) - evidence.frozen.average_precision) < 1e-12,
                 "Curve geometry disagrees with verified metrics.")
        return CurveResult(roc, pr, provenance,
                           "AMP-LDP only · derived from verified Stage 1B reconstructed scores and aligned evaluation labels.")
    except (ArtifactError, KeyError, TypeError, ValueError, UnicodeError, IndexError, OSError) as exc:
        return CurveResult(None, None, None, f"Evaluation curves unavailable: {exc}")
