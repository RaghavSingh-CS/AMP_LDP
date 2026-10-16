"""Evidence, chart semantics and navigation regressions for Stage 3."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.config import REPOSITORY_ROOT as ROOT
from app.data_loader import ArtifactError
from app.performance_loader import AP_DEFINITION, METRICS, MODEL_ORDER, absolute_differences, load_curves, load_performance
from app.views.performance import comparison_chart, curve_chart


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".fixture-performance-", dir=ROOT / "tests/interface")
        self.root = Path(self.temp.name)
        catalog = json.loads((ROOT / "app/artifact_catalog.json").read_text())
        performance = json.loads((ROOT / "app/performance_catalog.json").read_text())
        paths = ["app/artifact_catalog.json", "app/performance_catalog.json"]
        paths += [r["path"] for r in catalog["artifacts"].values()]
        paths += [catalog["predictions"]["path"], catalog["predictions"]["provenance"]["path"]]
        paths += [performance[key]["path"] for key in ("comparators", "evaluator", "method_document", "historical_architectures", "curves")]
        provenance = json.loads((ROOT / performance["curves"]["path"]).read_text())
        paths += [provenance["coordinates"]["path"]]
        for relative in paths:
            target = self.root / relative
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(ROOT / relative, target)

    def tearDown(self):
        self.temp.cleanup()

    def rewrite(self, relative, mutate, *, catalog="app/performance_catalog.json", key=None):
        path = self.root / relative
        obj = json.loads(path.read_text())
        mutate(obj)
        path.write_text(json.dumps(obj) + "\n", encoding="utf-8", newline="\n")
        if key:
            cpath = self.root / catalog
            contents = json.loads(cpath.read_text())
            contents["artifacts"][key]["sha256"] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            cpath.write_text(json.dumps(contents), encoding="utf-8")

    def repin(self, key):
        path = self.root / "app/performance_catalog.json"
        c = json.loads(path.read_text())
        c[key]["sha256"] = hashlib.sha256((self.root / c[key]["path"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        path.write_text(json.dumps(c), encoding="utf-8")

    def test_frozen_values_prevalence_and_order(self):
        e = load_performance(self.root)
        self.assertEqual(e.comparison.model.tolist(), MODEL_ORDER)
        self.assertEqual(e.comparison.customers.tolist(), [6356] * 3)
        self.assertEqual(e.comparison.ROC_AUC.tolist(), [.8117294176037735, .7978625838889967, .7949673266082506])
        self.assertEqual(e.comparison.average_precision.tolist(), [.42186322254778824, .3853256447732082, .3684394274673042])
        self.assertEqual(e.prevalence, 542 / 6356)
        for key in METRICS:
            self.assertEqual(e.comparison.sort_values(key, ascending=False).model.tolist(), MODEL_ORDER)

    def test_absolute_deltas_use_full_precision(self):
        e = load_performance(self.root)
        d = absolute_differences(e)
        self.assertAlmostEqual(d.ROC_AUC.iloc[0], .0138668337147768)
        self.assertAlmostEqual(d.average_precision.iloc[0], .03653757777458004)
        self.assertAlmostEqual(d.ROC_AUC.iloc[1], .0167620909955229)
        self.assertAlmostEqual(d.average_precision.iloc[1], .05342379508048406)

    def test_comparison_chart_values_labels_and_axes(self):
        e = load_performance(self.root)
        for key, label in METRICS.items():
            chart = comparison_chart(e.comparison, key)
            pd.testing.assert_frame_equal(chart.data, e.comparison)
            spec = chart.to_dict()
            self.assertEqual(spec["encoding"]["x"]["field"], key)
            self.assertEqual(spec["encoding"]["x"]["title"], label)
            self.assertEqual(spec["encoding"]["x"]["scale"]["domain"], [0, 1])
            self.assertEqual(spec["encoding"]["y"]["sort"], MODEL_ORDER)

    def test_curve_provenance_alignment_and_geometry(self):
        e = load_performance(self.root)
        c = load_curves(e, self.root)
        self.assertIsNotNone(c.roc, c.message)
        self.assertEqual((len(c.roc), len(c.pr)), (835, 6050))
        self.assertEqual(c.provenance["AP_definition"], AP_DEFINITION)
        self.assertEqual(c.provenance["labels_sha256"], e.frozen.integrity["labels_sha256"])
        self.assertEqual(c.provenance["score_origin"], "reconstructed_inference_only")
        self.assertAlmostEqual(np.trapezoid(c.roc.true_positive_rate, c.roc.false_positive_rate), e.frozen.roc_auc, places=14)
        self.assertAlmostEqual(np.sum(np.diff(c.pr.recall) * c.pr.precision.to_numpy()[1:]), e.frozen.average_precision, places=14)
        self.assertNotAlmostEqual(np.trapezoid(c.pr.precision, c.pr.recall), e.frozen.average_precision, places=6)

    def test_curve_chart_keeps_all_points_and_step_semantics(self):
        e = load_performance(self.root)
        c = load_curves(e, self.root)
        for frame, pr in ((c.roc, False), (c.pr, True)):
            chart = curve_chart(frame.assign(point=range(len(frame))), pr=pr)
            spec = chart.to_dict()
            self.assertEqual(spec["mark"]["interpolate"], "step-before" if pr else "linear")
            self.assertEqual(len(chart.data.values), len(frame))
            for axis in ("x", "y"):
                self.assertEqual(spec["encoding"][axis]["scale"]["domain"], [0, 1])
            self.assertEqual(chart.data.values[0]["point"], 0)

    def test_absent_comparator_curves_are_explicit(self):
        e = load_performance(self.root)
        self.assertEqual(e.catalog["comparator_curves"], {"FULL_48": None, "DIRECT": None})
        self.assertEqual(len(e.comparison), 3)

    def test_missing_artifact_after_cached_load(self):
        load_performance(self.root)
        (self.root / "results/supervised/stage_27a/final_comparator_freeze.json").unlink()
        with self.assertRaisesRegex(ArtifactError, "Cannot read"):
            load_performance(self.root)

    def test_corrupt_comparator_hash_fails(self):
        load_performance(self.root)
        with (self.root / "results/supervised/stage_27a/final_comparator_freeze.json").open("ab") as f:
            f.write(b" ")
        with self.assertRaisesRegex(ArtifactError, "SHA-256"):
            load_performance(self.root)

    def test_malformed_catalog_and_json(self):
        for text in ("{", "[]", '{"schema_version":99}'):
            (self.root / "app/performance_catalog.json").write_text(text)
            with self.assertRaises(ArtifactError):
                load_performance(self.root)

    def test_rebound_incompatible_training_population_rejected(self):
        path = "results/supervised/stage_27a/final_comparator_freeze.json"
        self.rewrite(path, lambda d: d["comparators"]["FULL_48_HGB"].update(training_population="DEV_TRAIN+DEV_VAL"))
        self.repin("comparators")
        with self.assertRaisesRegex(ArtifactError, "training population"):
            load_performance(self.root)

    def test_rebound_checkpoint_mismatch_rejected(self):
        path = "results/supervised/stage_27a/final_comparator_freeze.json"
        self.rewrite(path, lambda d: d["comparators"]["FULL_48_HGB"].update(model_sha256="0" * 64))
        self.repin("comparators")
        with self.assertRaisesRegex(ArtifactError, "checkpoint"):
            load_performance(self.root)

    def test_wrong_metric_protocol_rejected(self):
        self.rewrite("results/supervised/stage_27a/stage27b_preregistration.json",
                     lambda d: d["metrics"].update(primary=["window ROC-AUC", "window PR-AUC"]),
                     catalog="app/artifact_catalog.json", key="protocol")
        # Keep terminal access linkage valid so the additional Stage 3 protocol guard is exercised.
        cp = self.root / "app/artifact_catalog.json"
        c = json.loads(cp.read_text())
        self.rewrite("results/supervised/stage_27b/final_access_event.json",
                     lambda d: d.update(authorized_by_preregistration_sha256=c["artifacts"]["protocol"]["sha256"]),
                     catalog="app/artifact_catalog.json", key="access")
        with self.assertRaisesRegex(ArtifactError, "customer evaluation protocol"):
            load_performance(self.root)

    def test_rebound_prevalence_mismatch_rejected(self):
        self.rewrite("results/supervised/stage_27b/final_integrity.json", lambda d: d.update(prevalence=.5),
                     catalog="app/artifact_catalog.json", key="integrity")
        with self.assertRaisesRegex(ArtifactError, "Prevalence"):
            load_performance(self.root)

    def test_missing_optional_curves_preserves_metrics(self):
        self.rewrite("app/performance_catalog.json", lambda d: d.update(curves=None))
        e = load_performance(self.root)
        self.assertEqual(len(e.comparison), 3)
        self.assertIsNone(load_curves(e, self.root).roc)

    def test_missing_curve_file_after_cache(self):
        e = load_performance(self.root)
        self.assertIsNotNone(load_curves(e, self.root).roc)
        (self.root / "app/derived/stage3/curves.json").unlink()
        self.assertIsNone(load_curves(e, self.root).roc)
        self.assertEqual(load_performance(self.root).frozen.customers, 6356)

    def test_rebound_wrong_curve_label_alignment_rejected(self):
        self.rewrite("app/derived/stage3/curve_provenance.json", lambda d: d.update(labels_sha256="0" * 64))
        self.repin("curves")
        c = load_curves(load_performance(self.root), self.root)
        self.assertIsNone(c.roc)
        self.assertIn("alignment", c.message)

    def test_rebound_wrong_ap_definition_rejected(self):
        self.rewrite("app/derived/stage3/curve_provenance.json", lambda d: d.update(AP_definition="trapezoidal PR area"))
        self.repin("curves")
        self.assertIsNone(load_curves(load_performance(self.root), self.root).pr)

    def test_corrupt_and_malformed_coordinates_rejected(self):
        path = self.root / "app/derived/stage3/curves.json"
        path.write_text("{}")
        self.assertIsNone(load_curves(load_performance(self.root), self.root).roc)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.rewrite("app/derived/stage3/curve_provenance.json", lambda d: d["coordinates"].update(sha256=digest))
        self.repin("curves")
        self.assertIn("unavailable", load_curves(load_performance(self.root), self.root).message)

    def test_protected_research_exact_bytes_unchanged(self):
        snapshot = json.loads((ROOT / "app/derived/stage3/research_snapshot.json").read_text())
        self.assertEqual(len(snapshot), 543)
        for relative, expected in snapshot.items():
            with (ROOT / relative).open("rb") as handle:
                self.assertEqual(hashlib.file_digest(handle, "sha256").hexdigest(), expected, relative)


class PerformanceAppTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=25).run()
        return app.radio[0].set_value("Model Performance").run()

    def test_metrics_curves_labels_and_navigation(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ["0.811729", "0.421863", "6,356", "542", "8.5274%"])
        self.assertEqual(app.dataframe[0].value.Model.tolist(), MODEL_ORDER)
        self.assertEqual(len(app.get("vega_lite_chart")), 4)
        prose = " ".join(m.value for m in [*app.markdown, *app.caption])
        self.assertIn("not trapezoidal PR area", prose)
        self.assertIn("no applicable frozen AMP-LDP binary operating threshold", prose)
        self.assertTrue(any("FULL_48 and DIRECT curves are unavailable" in x.value for x in app.info))
        app.radio[0].set_value("Customer Risk Explorer").run()
        app.text_input[0].set_value("no match").run()
        app.radio[0].set_value("Model Performance").run()
        app.radio[0].set_value("Customer Risk Explorer").run()
        self.assertEqual(app.text_input[0].value, "no match")
        app.radio[0].set_value("Overview").run()
        self.assertEqual(len(app.metric), 4)

    def test_unavailable_evidence_is_actionable(self):
        with patch("app.views.performance.load_performance", side_effect=ArtifactError("Missing fixture")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertIn("Model performance unavailable", app.error[0].value)
            self.assertEqual(len(app.metric), 0)

    def test_missing_curves_still_renders_comparison(self):
        from app.performance_loader import CurveResult
        with patch("app.views.performance.load_curves", return_value=CurveResult(None, None, None, "Evaluation curves unavailable: missing fixture")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value), 3)
            self.assertEqual(len(app.get("vega_lite_chart")), 2)
            self.assertIn("unavailable", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
