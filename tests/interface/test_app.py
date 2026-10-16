"""Streamlit runtime tests, with synthetic data confined to mocked UI tests."""

import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from app.config import REPOSITORY_ROOT
from app.data_loader import ArtifactError, PredictionResult


class AppTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(REPOSITORY_ROOT / "app/main.py"), default_timeout=20).run()

    def test_overview_and_navigation(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ["6,356", "542", "0.8117", "0.4219"])
        app.radio[0].set_value("Customer Risk Explorer").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "6,356")
        self.assertFalse(app.text_input[0].disabled)
        self.assertFalse(app.selectbox[0].disabled)
        self.assertEqual(len(app.dataframe[0].value), 6356)
        self.assertTrue(app.dataframe[0].value["risk_score"].is_monotonic_decreasing)
        customer = app.dataframe[0].value["CONS_NO"].iloc[0]
        app.text_input[0].set_value(customer).run()
        self.assertEqual(app.dataframe[0].value["CONS_NO"].tolist(), [customer])
        self.assertEqual(app.dataframe[0].value["rank"].tolist(), [1])
        app.radio[0].set_value("Overview").run()
        self.assertEqual(len(app.metric), 4)

    def test_unavailable_explorer(self):
        with patch("app.views.explorer.load_predictions", return_value=PredictionResult(None, None, "Missing test artifact")):
            app = self.app()
            app.radio[0].set_value("Customer Risk Explorer").run()
            self.assertFalse(app.exception)
            self.assertIn("scores unavailable", app.warning[0].value)
            self.assertTrue(app.text_input[0].disabled)
            self.assertTrue(app.selectbox[0].disabled)
            self.assertEqual(len(app.dataframe), 0)

    def test_unavailable_overview(self):
        with patch("app.views.overview.load_evidence", side_effect=ArtifactError("Missing test artifact")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertIn("Frozen metrics unavailable", app.error[0].value)
            self.assertEqual(len(app.metric), 0)

    def test_explorer_table_search_and_sort(self):
        frame = pd.DataFrame({"rank": [1, 2, 3], "CONS_NO": ["NA", "a[b", "001"],
                              "risk_score": [0.9, 0.8, 0.1], "FLAG": [1, 0, 0]})
        with patch("app.views.explorer.load_predictions", return_value=PredictionResult(frame, "synthetic UI fixture", "Test only")):
            app = self.app()
            app.radio[0].set_value("Customer Risk Explorer").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value["rank"].tolist(), [1, 2, 3])
            app.selectbox[0].set_value("Lowest risk first").run()
            self.assertEqual(app.dataframe[0].value["rank"].tolist(), [3, 2, 1])
            app.text_input[0].set_value("[").run()
            self.assertEqual(app.dataframe[0].value["CONS_NO"].tolist(), ["a[b"])
            app.text_input[0].set_value("no match").run()
            self.assertTrue(app.dataframe[0].value.empty)
            self.assertIn("No matching", app.info[-1].value)
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
