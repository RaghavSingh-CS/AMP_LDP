"""Real derived-artifact integration and unavailable-state regression tests."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from app.config import REPOSITORY_ROOT
from app.data_loader import explore, load_predictions


class DerivedScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".fixture-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = json.loads((REPOSITORY_ROOT / "app/artifact_catalog.json").read_text())
        self.record = self.catalog["predictions"]
        paths = ["app/artifact_catalog.json", self.record["path"], self.record["provenance"]["path"]]
        paths.extend(r["path"] for r in self.catalog["artifacts"].values())
        for relative in paths:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY_ROOT / relative, target)

    def repin_manifest(self, edit):
        path = self.root / self.record["provenance"]["path"]
        manifest = json.loads(path.read_text())
        edit(manifest)
        content = (json.dumps(manifest) + "\n").encode()
        path.write_bytes(content)
        self.record["provenance"]["sha256"] = hashlib.sha256(content).hexdigest()
        (self.root / "app/artifact_catalog.json").write_text(json.dumps(self.catalog))

    def test_real_score_schema_and_alignment(self):
        result = load_predictions(self.root)
        self.assertIsNotNone(result.frame, result.message)
        frame = result.frame
        self.assertEqual(len(frame), 6356)
        self.assertEqual(frame["FLAG"].sum(), 542)
        self.assertEqual(list(frame.columns), ["rank", "CONS_NO", "FLAG", "risk_score"])
        self.assertFalse(frame["CONS_NO"].duplicated().any())
        self.assertTrue(frame["risk_score"].between(0, 1).all())
        self.assertEqual(frame["rank"].tolist(), list(range(1, 6357)))
        self.assertTrue(frame["risk_score"].is_monotonic_decreasing)
        self.assertIn("not original persisted", result.message)
        self.assertEqual(result.provenance_source, self.record["provenance"]["path"])
        customer = frame.iloc[0]["CONS_NO"]
        self.assertEqual(explore(frame, customer.lower(), "Highest risk first")["rank"].tolist(), [1])
        self.assertTrue(explore(frame, "", "Lowest risk first")["risk_score"].is_monotonic_increasing)

    def test_missing_scores_after_cached_load(self):
        self.assertIsNotNone(load_predictions(self.root).frame)
        (self.root / self.record["path"]).unlink()
        result = load_predictions(self.root)
        self.assertIsNone(result.frame)
        self.assertIn("Cannot read", result.message)

    def test_corrupt_scores_after_cached_load(self):
        self.assertIsNotNone(load_predictions(self.root).frame)
        path = self.root / self.record["path"]
        path.write_bytes(path.read_bytes() + b"corrupt\n")
        result = load_predictions(self.root)
        self.assertIsNone(result.frame)
        self.assertIn("SHA-256 mismatch", result.message)

    def test_missing_provenance(self):
        (self.root / self.record["provenance"]["path"]).unlink()
        self.assertIsNone(load_predictions(self.root).frame)

    def test_corrupt_provenance(self):
        (self.root / self.record["provenance"]["path"]).write_bytes(b"{}")
        result = load_predictions(self.root)
        self.assertIsNone(result.frame)
        self.assertIn("SHA-256 mismatch", result.message)

    def test_wrong_output_binding(self):
        self.repin_manifest(lambda manifest: manifest["output"].update(sha256="0" * 64))
        result = load_predictions(self.root)
        self.assertIsNone(result.frame)
        self.assertIn("output binding", result.message)

    def test_wrong_model_identity(self):
        self.repin_manifest(lambda manifest: manifest.update(model_id="FULL_48"))
        self.assertIn("identity mismatch", load_predictions(self.root).message)

    def test_metric_disagreement(self):
        self.repin_manifest(lambda manifest: manifest["verification"]["observed"].update(ROC_AUC=0.5))
        self.assertIn("metric mismatch", load_predictions(self.root).message)

    def test_bad_label_hash(self):
        self.repin_manifest(lambda manifest: manifest["cohort_integrity"].update(labels_sha256="0" * 64))
        self.assertIn("cohort hash", load_predictions(self.root).message)

    def test_fitting_invalidates_provenance(self):
        self.repin_manifest(lambda manifest: manifest["scoring"].update(fit_calls=1))
        self.assertIn("boundaries", load_predictions(self.root).message)


if __name__ == "__main__":
    unittest.main()
