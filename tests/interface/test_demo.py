"""Stage 4 presenter, export, preflight, precision and navigation checks."""

import csv
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import REPOSITORY_ROOT as ROOT
from app.data_loader import ArtifactError, load_predictions
from app.demo import aggregate_summary, load_demo, select_examples
from app.performance_loader import CurveResult, load_curves, load_performance
from app.preflight import dependency_checks, run


def button(app, label):
    return next(b for b in app.button if b.label == label)


class PresenterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".fixture-demo-", dir=ROOT / "tests/interface")
        self.root = Path(self.temp.name)
        catalog = json.loads((ROOT / "app/artifact_catalog.json").read_text())
        performance = json.loads((ROOT / "app/performance_catalog.json").read_text())
        history = json.loads((ROOT / "app/consumption_catalog.json").read_text())
        paths = ["app/artifact_catalog.json", "app/performance_catalog.json", "app/consumption_catalog.json",
                 "app/demo_catalog.json", "requirements-interface.txt", ".streamlit/config.toml", "app/main.py"]
        paths += [r["path"] for r in catalog["artifacts"].values()]
        paths += [catalog["predictions"]["path"], catalog["predictions"]["provenance"]["path"]]
        paths += [r["path"] for r in performance.values() if isinstance(r, dict) and "path" in r]
        provenance = json.loads((ROOT / performance["curves"]["path"]).read_text())
        paths += [provenance["coordinates"]["path"], history["index"]["path"], history["provenance"]["path"]]
        for relative in paths:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)

    def tearDown(self):
        self.temp.cleanup()

    def change_demo(self, mutate):
        path = self.root / "app/demo_catalog.json"
        obj = json.loads(path.read_text())
        mutate(obj)
        path.write_text(json.dumps(obj), encoding="utf-8")

    def test_real_identifiers_and_selection_criteria(self):
        demo = load_demo(self.root)
        self.assertEqual(demo.examples.CONS_NO.tolist(), ["45C64D484FC7B1A767557CF0674330E0", "D5F510E7EFD856CDA64770568C25017B", "C21A9C87D49E6945FB4570F38F73C023"])
        self.assertEqual(demo.examples["rank"].tolist(), [1, 6356, 13])
        self.assertEqual(demo.examples.FLAG.tolist(), [1, 0, 0])
        scores = load_predictions(self.root).frame
        pd.testing.assert_frame_equal(demo.examples, select_examples(scores.sample(frac=1, random_state=42)))

    def test_all_parsed_scores_exactly_preserve_existing_csv_floats(self):
        with (ROOT / "app/derived/stage1b/customer_scores.csv").open() as handle:
            raw = {row["CONS_NO"]: float(row["risk_score"]) for row in csv.DictReader(handle)}
        scores = load_predictions(self.root).frame
        self.assertEqual(scores.CONS_NO.tolist(), sorted(raw, key=lambda key: (-raw[key], key)))
        self.assertEqual(scores.risk_score.tolist(), [raw[key] for key in scores.CONS_NO])

    def test_wrong_original_identifier_rejected(self):
        self.change_demo(lambda c: c["examples"][0].update(CONS_NO="invented"))
        with self.assertRaisesRegex(ArtifactError, "identifiers"):
            load_demo(self.root)

    def test_changed_selection_rule_rejected(self):
        self.change_demo(lambda c: c["examples"][0].update(criterion="Selected by live inference"))
        with self.assertRaisesRegex(ArtifactError, "selection rules"):
            load_demo(self.root)

    def test_wrong_score_or_provenance_binding_rejected(self):
        self.change_demo(lambda c: c["source_scores"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ArtifactError, "provenance"):
            load_demo(self.root)

    def test_label_selection_disclosure_required(self):
        self.change_demo(lambda c: c.update(selection_uses_evaluation_labels=False))
        with self.assertRaisesRegex(ArtifactError, "provenance"):
            load_demo(self.root)

    def test_missing_score_after_success_does_not_substitute_example(self):
        load_demo(self.root)
        (self.root / "app/derived/stage1b/customer_scores.csv").unlink()
        with self.assertRaises(ArtifactError):
            load_demo(self.root)

    def test_missing_and_malformed_demo_catalog(self):
        path = self.root / "app/demo_catalog.json"
        for text in ("{", "[]", '{"schema_version":9}'):
            path.write_text(text)
            with self.assertRaises(ArtifactError):
                load_demo(self.root)
        path.unlink()
        with self.assertRaises(ArtifactError):
            load_demo(self.root)

    def test_missing_dependencies_report_individual_failures(self):
        versions = {name: metadata.version(name) for name in ("streamlit", "pandas", "numpy", "pyarrow", "altair")}
        with patch("app.preflight.metadata.version", side_effect=lambda name: versions[name] if name != "streamlit" else (_ for _ in ()).throw(metadata.PackageNotFoundError(name))):
            checks = dependency_checks(self.root)
        self.assertEqual(sum(c["status"] == "FAIL" for c in checks), 1)
        self.assertIn("Missing Python dependency: streamlit", checks[0]["detail"])
        self.assertIn("requirements-interface.txt", checks[0]["recovery"])

    def test_wrong_dependency_version_is_failure(self):
        with patch("app.preflight.metadata.version", return_value="0.0.0"):
            self.assertTrue(all(c["status"] == "FAIL" for c in dependency_checks(self.root)))

    def test_preflight_reports_missing_source_without_repair(self):
        report = run(self.root)
        self.assertEqual(report["status"], "FAIL")
        history = next(c for c in report["checks"] if c["name"] == "history inputs and selected histories")
        self.assertEqual(history["status"], "FAIL")
        self.assertIn("data set.csv", history["detail"])
        self.assertIn("Do not rerun inference", history["recovery"])
        self.assertFalse((self.root / "data/raw/data set.csv").exists())

    def test_optional_missing_curves_are_warning_not_fake_fallback(self):
        (self.root / "app/derived/stage3/curves.json").unlink()
        with patch("app.preflight.history_check", return_value="Synthetic test-only history check; source absent in fixture"):
            report = run(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["optional_warnings"], 1)
        self.assertIn("unavailable", report["checks"][-1]["detail"])


class AggregateExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = load_performance()
        cls.curves = load_curves(cls.evidence)

    def test_verified_full_precision_metrics_definitions_and_sources(self):
        payload = json.loads(aggregate_summary(self.evidence, self.curves))
        self.assertEqual(payload["metrics"], self.evidence.comparison.to_dict("records"))
        self.assertEqual(payload["cohort"]["prevalence"], 542 / 6356)
        self.assertIn("non-interpolated", payload["metric_definitions"]["average_precision"])
        self.assertIn("historical", payload["cohort"]["name"])
        self.assertIn("Stage 13", payload["cohort"]["historical_access_disclosure"])
        self.assertEqual(payload["score_context"]["origin"], "reconstructed_inference_only")
        self.assertFalse(payload["score_context"]["original_persisted_stage27b_predictions"])
        self.assertEqual(payload["sources"]["metrics"], self.evidence.frozen.catalog["artifacts"]["metrics"])

    def test_export_excludes_every_original_identifier_and_customer_row(self):
        content = aggregate_summary(self.evidence, self.curves).decode()
        for identifier in load_predictions().frame.CONS_NO:
            self.assertNotIn(identifier, content)
        payload = json.loads(content)
        self.assertEqual(set(payload["metrics"][0]), {"model", "customers", "ROC_AUC", "average_precision"})
        for forbidden in ('"CONS_NO"', '"risk_score"', '"readings"', '"examples"'):
            self.assertNotIn(forbidden, content)

    def test_export_stable_and_optional_curve_absence_honest(self):
        self.assertEqual(aggregate_summary(self.evidence, self.curves), aggregate_summary(self.evidence, self.curves))
        payload = json.loads(aggregate_summary(self.evidence, CurveResult(None, None, None, "Missing test artifact")))
        self.assertEqual(payload["curves"]["AMP-LDP"], "unavailable")
        self.assertNotIn("curve_coordinates", payload["sources"])
        self.assertEqual(len(payload["metrics"]), 3)


class DemoAppTests(unittest.TestCase):
    def guide(self):
        return AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=30).run().radio[0].set_value("Demo Guide").run()

    def test_guide_five_steps_and_all_navigation_buttons(self):
        app = self.guide()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.subheader), 5)
        button(app, "Open Overview").click().run()
        self.assertEqual(app.radio[0].value, "Overview")
        app.radio[0].set_value("Demo Guide").run()
        button(app, "Open customer rankings").click().run()
        self.assertEqual(app.radio[0].value, "Customer Risk Explorer")
        self.assertEqual(len(app.dataframe[0].value), 6356)
        app.radio[0].set_value("Demo Guide").run()
        button(app, "Open Model Performance").click().run()
        self.assertEqual(app.radio[0].value, "Model Performance")
        self.assertEqual(len(app.get("download_button")), 1)

    def test_real_presets_open_correct_customer_with_exact_score(self):
        app = self.guide()
        for row in load_demo().examples.itertuples():
            app.radio[0].set_value("Demo Guide").run()
            app.selectbox[0].set_value(row.key).run()
            button(app, "Inspect verified example").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["amp_explorer"]["customer_id"], row.CONS_NO)
            self.assertEqual(int(app.metric[1].value.replace(",", "")), row.rank)
            exact = next(c.value for c in app.caption if c.value.startswith("Exact stored score: "))
            self.assertEqual(float(exact.split(": ")[1]), row.risk_score)
            self.assertEqual(len(app.date_input), 1)

    def test_guide_does_not_erase_search_sort_or_customer_interval(self):
        app = self.guide()
        button(app, "Open customer rankings").click().run()
        customer = load_demo().examples.CONS_NO.iloc[0]
        app.text_input[0].set_value(customer).run()
        app.selectbox[0].set_value("Lowest risk first").run()
        button(app, "Open customer inspector").click().run()
        interval = app.date_input[0].value
        app.radio[0].set_value("Demo Guide").run()
        app.selectbox[0].set_value("low_normal").run()
        button(app, "Inspect verified example").click().run()
        button(app, "Back to customer ranking").click().run()
        self.assertEqual(app.text_input[0].value, customer)
        self.assertEqual(app.selectbox[0].value, "Lowest risk first")
        button(app, "Open customer inspector").click().run()
        self.assertEqual(tuple(app.date_input[0].value), tuple(interval))

    def test_missing_examples_show_narrative_and_actionable_warning(self):
        with patch("app.views.demo_guide.load_demo", side_effect=ArtifactError("Missing fixture scores")):
            app = self.guide()
            self.assertFalse(app.exception)
            self.assertIn("Presenter examples unavailable", app.warning[0].value)
            self.assertEqual(len(app.subheader), 5)
            self.assertFalse(any(b.label == "Inspect verified example" for b in app.button))

    def test_artifact_loss_during_preset_click_returns_to_guide(self):
        app = self.guide()
        with patch("app.navigation.load_demo", side_effect=ArtifactError("Artifact removed during click")):
            button(app, "Inspect verified example").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, "Demo Guide")
        self.assertIn("Artifact removed", app.error[0].value)

    def test_presentation_export_configuration(self):
        self.guide()
        self.assertTrue(st.get_option("client.disableDataExport"))
        self.assertEqual(st.get_option("client.toolbarMode"), "viewer")


class OfflineReadinessTests(unittest.TestCase):
    def environment(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "tests/offline") + os.pathsep + str(ROOT)
        return env

    def test_fresh_preflight_with_network_blocked_and_guard_enforced(self):
        program = '''
import json,socket,sys
for action in [lambda:socket.create_connection(("example.com",443)), lambda:socket.getaddrinfo("example.com",443),lambda:socket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(b"x",("8.8.8.8",53))]:
 try:action()
 except OSError:pass
 else:raise AssertionError("Network guard not enforced")
from app.preflight import run
report=run()
assert not any(name in sys.modules for name in ("torch","sklearn","app.recovery.replay_stage1b"))
print(json.dumps(report))
'''
        proc = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=self.environment(), capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["optional_warnings"], 0)

    def test_stage4_protected_bytes_unchanged(self):
        import hashlib
        snapshot = json.loads((ROOT / "app/derived/stage4/research_snapshot.json").read_text())
        self.assertEqual(len(snapshot), 549)
        for path, expected in snapshot.items():
            with (ROOT / path).open("rb") as handle:
                self.assertEqual(hashlib.file_digest(handle, "sha256").hexdigest(), expected, path)


if __name__ == "__main__":
    unittest.main()
