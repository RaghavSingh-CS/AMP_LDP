"""Create-only, offline visualization of existing scores. No model imports/access."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone

import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from app.config import REPOSITORY_ROOT
from app.data_loader import _csv, _require, _verified, stable_hash, load_predictions
from app.performance_loader import AP_DEFINITION, load_performance


def main():
    root = REPOSITORY_ROOT
    output = root / "app/derived/stage3"
    paths = [output / "curves.json", output / "curve_provenance.json"]
    _require(all(not p.exists() for p in paths), "Curve builder never overwrites artifacts.")
    snapshot = json.loads((output / "research_snapshot.json").read_text())

    def digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def unchanged():
        _require(all(digest(root / path) == value for path, value in snapshot.items()), "Protected artifact changed.")

    unchanged()
    evidence = load_performance()
    result = load_predictions()
    _require(result.frame is not None, result.message)
    source = evidence.frozen.catalog["predictions"]
    original = _csv(_verified(root, source))
    ids, labels = original.CONS_NO.tolist(), original.FLAG.tolist()
    for values, key in ((ids, "customer_membership_sha256"), (labels, "labels_sha256")):
        _require(stable_hash(values) == evidence.frozen.integrity[key], "Original score alignment mismatch.")
    y = [int(v) for v in labels]
    scores = [float(v) for v in original.risk_score]
    metrics = {"ROC_AUC": float(roc_auc_score(y, scores)), "average_precision": float(average_precision_score(y, scores))}
    _require(metrics == {"ROC_AUC": evidence.frozen.roc_auc, "average_precision": evidence.frozen.average_precision},
             "Curve source does not reproduce frozen metrics exactly.")
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1, drop_intermediate=True)
    precision, recall, _ = precision_recall_curve(y, scores, pos_label=1, drop_intermediate=False)
    coordinates = {"schema_version": 1, "roc": list(map(list, zip(fpr.tolist(), tpr.tolist()))),
                   "pr": list(map(list, zip(recall[::-1].tolist(), precision[::-1].tolist())))}
    unchanged()
    paths[0].write_text(json.dumps(coordinates, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    relative = "app/derived/stage3"
    provenance = {"schema_version": 1, "status": "VERIFIED", "created_at": datetime.now(timezone.utc).isoformat(),
                  "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "builder": {"path": "app/recovery/build_performance_curves.py", "sha256": digest(root / "app/recovery/build_performance_curves.py")},
                  "source_scores": {"path": source["path"], "sha256": source["sha256"]},
                  "source_provenance": source["provenance"], "score_origin": source["origin"], "model_id": source["model_id"],
                  "protocol_sha256": evidence.frozen.catalog["artifacts"]["protocol"]["sha256"],
                  "metric_source": evidence.frozen.catalog["artifacts"]["metrics"],
                  "customers": len(y), "theft_customers": sum(y),
                  **{key: evidence.frozen.integrity[key] for key in ("customer_membership_sha256", "labels_sha256")},
                  "ROC_definition": "sklearn.metrics.roc_auc_score", "AP_definition": AP_DEFINITION,
                  "curve_functions": ["sklearn.metrics.roc_curve", "sklearn.metrics.precision_recall_curve"],
                  "parameters": {"pos_label": 1, "roc_drop_intermediate": True, "pr_drop_intermediate": False, "pr_order": "increasing recall; step-before"},
                  "verified_metrics": metrics, "no_inference_or_fitting": True, "no_operating_threshold_selected": True,
                  "points": {key: len(coordinates[key]) for key in ("roc", "pr")},
                  "coordinates": {"path": f"{relative}/curves.json", "sha256": digest(paths[0])},
                  "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
                  "protected_files_unchanged": len(snapshot)}
    paths[1].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    catalog_path = root / "app/performance_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["curves"] = {"path": f"{relative}/curve_provenance.json", "sha256": digest(paths[1])}
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8", newline="\n")
    unchanged()
    print(json.dumps({"metrics": metrics, "points": provenance["points"], "protected_files_unchanged": len(snapshot)}))


if __name__ == "__main__":
    main()
