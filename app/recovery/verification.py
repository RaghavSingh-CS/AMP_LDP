"""Independent integrity metrics using Stage 27B's definitions and FLAG=1."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from app.data_loader import ArtifactError


def verify_metrics(labels, scores, *, roc_auc: float, average_precision: float) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ArtifactError("Metric integrity requires aligned binary customer labels.")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ArtifactError("Metric integrity requires finite [0,1] risk scores.")
    observed = {"ROC_AUC": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores))}
    expected = {"ROC_AUC": roc_auc, "average_precision": average_precision}
    differences = {name: observed[name] - expected[name] for name in expected}
    matches = all(f"{observed[name]:.6f}" == f"{expected[name]:.6f}" for name in expected)
    if not matches:
        raise ArtifactError(f"Frozen metric disagreement; observed={observed}; expected={expected}; differences={differences}")
    return {"definition": "sklearn roc_auc_score and average_precision_score, positive label FLAG=1",
            "expected": expected, "observed": observed, "differences": differences,
            "required_decimal_places": 6, "agreement": True}
