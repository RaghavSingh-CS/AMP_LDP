"""Temporary fixtures exercise error paths without changing research artifacts."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from app.config import MODEL_ID, REPOSITORY_ROOT, SEEDS
from app.data_loader import (
    ArtifactError, explore, load_evidence, load_predictions, stable_hash, validate_predictions,
)


def fixture():
    # Deliberately synthetic unit-test records; never exposed by the application.
    frame = pd.DataFrame({"CONS_NO": ["001", "NA", "a[b"], "FLAG": ["0", "1", "0"],
                          "risk_score": ["0.1", "0.9", "0.9"]})
    governance = pd.DataFrame({"CONS_NO": ["001", "NA", "a[b", "training"],
                               "research_role": ["FINAL_TEST"] * 3 + ["DEV_TRAIN"],
                               "original_split": ["test"] * 3 + ["train"]})
    kwargs = {"customers": 3, "theft_customers": 1,
              "membership_sha256": stable_hash(frame["CONS_NO"].tolist()),
              "labels_sha256": stable_hash(frame["FLAG"].tolist())}
    return frame, governance, kwargs


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".fixture-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = json.loads((REPOSITORY_ROOT / "app/artifact_catalog.json").read_text())
        for relative in ["app/artifact_catalog.json"] + [r["path"] for r in self.catalog["artifacts"].values()]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY_ROOT / relative, target)

    def edit(self, name, content, repin=False):
        record = self.catalog["artifacts"][name]
        (self.root / record["path"]).write_bytes(content)
        if repin:
            record["sha256"] = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
            (self.root / "app/artifact_catalog.json").write_text(json.dumps(self.catalog))

    def test_genuine_frozen_metrics(self):
        evidence = load_evidence(self.root)
        self.assertEqual((evidence.customers, evidence.theft_customers), (6356, 542))
        self.assertEqual(evidence.roc_auc, 0.8117294176037735)
        self.assertEqual(evidence.average_precision, 0.42186322254778824)
        self.assertIn("Stage 13", evidence.disclosure)

    def test_missing_metric_after_cached_load(self):
        load_evidence(self.root)
        (self.root / self.catalog["artifacts"]["metrics"]["path"]).unlink()
        with self.assertRaisesRegex(ArtifactError, "Cannot read"):
            load_evidence(self.root)

    def test_changed_metric_after_cached_load(self):
        load_evidence(self.root)
        self.edit("metrics", b"changed")
        with self.assertRaisesRegex(ArtifactError, "SHA-256 mismatch"):
            load_evidence(self.root)

    def test_crlf_and_lf_checkouts(self):
        record = self.catalog["artifacts"]["metrics"]
        content = (self.root / record["path"]).read_bytes().replace(b"\r\n", b"\n")
        self.edit("metrics", content.replace(b"\n", b"\r\n"))
        self.assertEqual(load_evidence(self.root).customers, 6356)

    def test_malformed_json(self):
        self.edit("integrity", b"{invalid", repin=True)
        with self.assertRaisesRegex(ArtifactError, "Malformed"):
            load_evidence(self.root)

    def test_malformed_metric_schema(self):
        self.edit("metrics", b"wrong,columns\na,b\n", repin=True)
        with self.assertRaisesRegex(ArtifactError, "columns"):
            load_evidence(self.root)

    def test_invalid_metric_and_count(self):
        path = self.root / self.catalog["artifacts"]["metrics"]["path"]
        original = path.read_bytes()
        for old, new in [(b"0.8117294176037735", b"NaN"), (b"6356", b"1")]:
            with self.subTest(new=new):
                self.edit("metrics", original.replace(old, new), repin=True)
                with self.assertRaises(ArtifactError):
                    load_evidence(self.root)

    def test_duplicate_metric_model(self):
        path = self.root / self.catalog["artifacts"]["metrics"]["path"]
        self.edit("metrics", path.read_bytes().replace(b"DIRECT,", b"LATENT,"), repin=True)
        with self.assertRaisesRegex(ArtifactError, "duplicate"):
            load_evidence(self.root)

    def test_path_traversal(self):
        self.catalog["artifacts"]["metrics"]["path"] = "../outside.csv"
        (self.root / "app/artifact_catalog.json").write_text(json.dumps(self.catalog))
        with self.assertRaisesRegex(ArtifactError, "inside"):
            load_evidence(self.root)

    def test_missing_canonical_scores(self):
        self.catalog["predictions"] = None
        (self.root / "app/artifact_catalog.json").write_text(json.dumps(self.catalog))
        result = load_predictions(self.root)
        self.assertIsNone(result.frame)
        self.assertIsNone(result.source)
        self.assertIn("no canonical", result.message)

    def test_governance_final_count(self):
        record = self.catalog["artifacts"]["governance"]
        governance = pd.read_csv(self.root / record["path"], dtype=str, keep_default_na=False)
        final = governance.loc[governance["research_role"].eq("FINAL_TEST")]
        self.assertEqual(len(final), load_evidence(self.root).customers)
        self.assertFalse(governance["CONS_NO"].duplicated().any())
        self.assertTrue(final["original_split"].eq("test").all())


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.governance, self.kwargs = fixture()

    def validate(self):
        return validate_predictions(self.frame, self.governance, **self.kwargs)

    def test_ranking_ties_original_ids_and_literal_search(self):
        result = self.validate()
        self.assertEqual(result["CONS_NO"].tolist(), ["NA", "a[b", "001"])
        self.assertEqual(result["rank"].tolist(), [1, 2, 3])
        self.assertEqual(explore(result, "[", "Highest risk first")["rank"].tolist(), [2])
        self.assertEqual(explore(result, "na", "Highest risk first")["CONS_NO"].tolist(), ["NA"])
        self.assertEqual(explore(result, "", "Lowest risk first")["rank"].tolist(), [3, 1, 2])
        self.assertEqual(explore(result, "", "Customer identifier")["CONS_NO"].tolist(), ["001", "NA", "a[b"])
        self.assertTrue(explore(result, "absent", "Highest risk first").empty)

    def test_duplicate_identifier(self):
        self.frame.loc[1, "CONS_NO"] = "001"
        with self.assertRaisesRegex(ArtifactError, "Duplicate"):
            self.validate()

    def test_nonfinite_out_of_range_and_nonnumeric_scores(self):
        for score in ["nan", "inf", "-inf", "-0.01", "1.01", "invalid", ""]:
            with self.subTest(score=score):
                self.frame.loc[0, "risk_score"] = score
                with self.assertRaises(ArtifactError):
                    self.validate()

    def test_schema(self):
        self.frame = self.frame.drop(columns="risk_score")
        with self.assertRaisesRegex(ArtifactError, "columns"):
            self.validate()

    def test_cohort_size(self):
        self.kwargs["customers"] = 4
        with self.assertRaisesRegex(ArtifactError, "size"):
            self.validate()

    def test_training_record_substitution(self):
        self.frame.loc[0, "CONS_NO"] = "training"
        with self.assertRaisesRegex(ArtifactError, "split membership"):
            self.validate()

    def test_membership_order(self):
        self.frame = self.frame.iloc[::-1]
        with self.assertRaisesRegex(ArtifactError, "order/hash"):
            self.validate()

    def test_labels(self):
        for labels in [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "0"]]:
            with self.subTest(labels=labels):
                self.frame["FLAG"] = labels
                with self.assertRaises(ArtifactError):
                    self.validate()

    def test_empty_identifiers(self):
        self.frame["CONS_NO"] = self.frame["CONS_NO"].astype(object)
        for value in ["", "  ", None, 1]:
            with self.subTest(value=value):
                self.frame.loc[0, "CONS_NO"] = value
                with self.assertRaisesRegex(ArtifactError, "original strings"):
                    self.validate()

    def test_complete_future_export_adapter_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix=".fixture-", dir=Path(__file__).parent) as directory:
            root = Path(directory)
            self.frame.to_csv(root / "scores.csv", index=False)
            self.governance.to_csv(root / "roles.csv", index=False)
            def record(path):
                return {"path": path, "sha256": hashlib.sha256((root / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
            evidence = load_evidence()
            catalog = deepcopy(evidence.catalog)
            catalog["artifacts"]["governance"] = record("roles.csv")
            catalog["predictions"] = {**record("scores.csv"), "model_id": MODEL_ID,
                                      "source_stage": "27B", "research_role": "FINAL_TEST",
                                      "seeds": SEEDS, "ensemble": "arithmetic_mean",
                                      "model_freeze_sha256": catalog["artifacts"]["model"]["sha256"]}
            integrity = {"customer_membership_sha256": self.kwargs["membership_sha256"],
                         "labels_sha256": self.kwargs["labels_sha256"]}
            sample = replace(evidence, customers=3, theft_customers=1, catalog=catalog, integrity=integrity)
            with patch("app.data_loader.load_evidence", return_value=sample):
                loaded = load_predictions(root)
                self.assertEqual(loaded.frame["CONS_NO"].tolist(), ["NA", "a[b", "001"])
                self.assertEqual(loaded.source, "scores.csv")
                catalog["predictions"]["model_id"] = "FULL_48"
                self.assertIn("provenance", load_predictions(root).message)
                catalog["predictions"]["model_id"] = MODEL_ID
                (root / "scores.csv").unlink()
                self.assertIn("Cannot read", load_predictions(root).message)


if __name__ == "__main__":
    unittest.main()
