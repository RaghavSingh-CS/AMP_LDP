"""Independent original-definition checks; never replay model inference."""

import hashlib
import json
import subprocess
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from app.config import REPOSITORY_ROOT as ROOT


class CurveVerificationTests(unittest.TestCase):
    def test_existing_scores_match_frozen_metrics_and_all_coordinates(self):
        scores = pd.read_csv(ROOT / "app/derived/stage1b/customer_scores.csv", dtype={"CONS_NO": str}, float_precision="round_trip")
        frozen = pd.read_csv(ROOT / "results/supervised/stage_27b/final_primary_metrics.csv", float_precision="round_trip").set_index("model").loc["LATENT"]
        self.assertEqual(roc_auc_score(scores.FLAG, scores.risk_score), frozen.ROC_AUC)
        self.assertEqual(average_precision_score(scores.FLAG, scores.risk_score), frozen.average_precision)
        coords = json.loads((ROOT / "app/derived/stage3/curves.json").read_text())
        fpr, tpr, _ = roc_curve(scores.FLAG, scores.risk_score, pos_label=1)
        p, r, _ = precision_recall_curve(scores.FLAG, scores.risk_score, pos_label=1)
        np.testing.assert_array_equal(coords["roc"], np.column_stack([fpr, tpr]))
        np.testing.assert_array_equal(coords["pr"], np.column_stack([r[::-1], p[::-1]]))
        manifest = json.loads((ROOT / "app/derived/stage3/curve_provenance.json").read_text())
        self.assertEqual(manifest["builder"]["sha256"], hashlib.sha256((ROOT / manifest["builder"]["path"]).read_bytes()).hexdigest())

    def test_create_only_builder_rejects_overwrite(self):
        path = ROOT / "app/derived/stage3/curves.json"
        before = path.read_bytes()
        proc = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "app.recovery.build_performance_curves"], cwd=ROOT, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("never overwrites", proc.stderr)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
