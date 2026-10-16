"""Offline integrity metrics; kept separate from CPU-only dashboard tests."""

import unittest

import json
import pandas as pd

from app.config import REPOSITORY_ROOT
from app.data_loader import ArtifactError, load_evidence
from app.recovery.replay_stage1b import forbid_fitting, run
from app.recovery.verification import verify_metrics


class MetricVerificationTests(unittest.TestCase):
    def test_derived_scores_match_full_precision_frozen_metrics(self):
        report = json.loads((REPOSITORY_ROOT / "app/derived/stage1b/provenance.json").read_text())
        evidence = load_evidence()
        frame = pd.read_csv(REPOSITORY_ROOT / report["output"]["path"], dtype={"CONS_NO": str}, float_precision="round_trip")
        result = verify_metrics(frame["FLAG"], frame["risk_score"],
                                roc_auc=evidence.roc_auc, average_precision=evidence.average_precision)
        self.assertEqual(result["differences"], {"ROC_AUC": 0.0, "average_precision": 0.0})

    def test_fit_and_backward_tripwires(self):
        import torch
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        with forbid_fitting():
            with self.assertRaisesRegex(RuntimeError, "prohibited"):
                StandardScaler().fit([[0], [1]])
            with self.assertRaisesRegex(RuntimeError, "prohibited"):
                HistGradientBoostingClassifier().fit([[0], [1]], [0, 1])
            with self.assertRaisesRegex(RuntimeError, "prohibited"):
                torch.tensor(1.0, requires_grad=True).backward()

    def test_replay_cannot_overwrite_or_target_research(self):
        for target in ["app/derived/stage1b", "results/supervised/replay", "data/replay", "../outside"]:
            with self.subTest(target=target), self.assertRaises(ArtifactError):
                run(target)

    def test_six_decimal_agreement(self):
        result = verify_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8],
                                roc_auc=1.0 - 1e-8, average_precision=1.0)
        self.assertTrue(result["agreement"])
        self.assertEqual(result["observed"], {"ROC_AUC": 1.0, "average_precision": 1.0})

    def test_metric_mismatch_reports_differences(self):
        with self.assertRaisesRegex(ArtifactError, "differences="):
            verify_metrics([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2],
                           roc_auc=1.0, average_precision=1.0)

    def test_alignment_label_orientation_and_scores(self):
        for labels, scores in [([0, 1], [0.1]), ([0, 2], [0.1, 0.9]),
                               ([0, 1], [0.1, float("nan")]), ([0, 1], [0.1, 1.1])]:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ArtifactError):
                verify_metrics(labels, scores, roc_auc=1.0, average_precision=1.0)


if __name__ == "__main__":
    unittest.main()
