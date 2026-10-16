"""Real Streamlit navigation, score identity and interval-state regressions."""

from datetime import timedelta
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import REPOSITORY_ROOT
from app.data_loader import ArtifactError, load_predictions


def button(app, label):
    return next(item for item in app.button if item.label == label)


def exact_score(app):
    caption = next(item.value for item in app.caption if item.value.startswith("Exact stored score: "))
    return float(caption.split(": ", 1)[1])


class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores = load_predictions().frame
        cls.normal = cls.scores.loc[cls.scores.FLAG.eq(0)].iloc[0]

    def opened(self):
        app = AppTest.from_file(str(REPOSITORY_ROOT / "app/main.py"), default_timeout=30).run()
        app.radio[0].set_value("Customer Risk Explorer").run()
        app.text_input[0].set_value(self.normal.CONS_NO).run()
        app.selectbox[0].set_value("Lowest risk first").run()
        button(app, "Open customer inspector").click().run()
        self.assertFalse(app.exception)
        return app

    def test_frozen_score_rank_and_return_context(self):
        app = self.opened()
        self.assertEqual(exact_score(app), float(self.normal.risk_score))
        self.assertEqual(int(app.metric[1].value.replace(",", "")), int(self.normal["rank"]))
        self.assertEqual(app.metric[2].value, "0")
        self.assertIn("Normal-labelled", [item.value for item in app.caption])
        self.assertEqual(app.session_state["amp_explorer"]["customer_id"], self.normal.CONS_NO)
        self.assertEqual(len(app.date_input), 1)
        button(app, "Back to customer ranking").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input[0].value, self.normal.CONS_NO)
        self.assertEqual(app.selectbox[0].value, "Lowest risk first")
        self.assertEqual(app.selectbox[1].value, self.normal.CONS_NO)

    def test_interval_persists_across_navigation_and_resets(self):
        app = self.opened()
        full = tuple(app.date_input[0].value)
        selected = (full[0] + timedelta(days=10), full[1] - timedelta(days=10))
        app.date_input[0].set_value(selected).run()
        self.assertFalse(app.exception)
        button(app, "Back to customer ranking").click().run()
        button(app, "Open customer inspector").click().run()
        self.assertEqual(tuple(app.date_input[0].value), selected)
        app.radio[0].set_value("Overview").run()
        app.radio[0].set_value("Customer Risk Explorer").run()
        self.assertEqual(tuple(app.date_input[0].value), selected)
        button(app, "Reset viewed interval").click().run()
        self.assertEqual(tuple(app.date_input[0].value), full)
        self.assertFalse(app.exception)

    def test_sequential_customer_inspection(self):
        app = self.opened()
        button(app, "Next rank").click().run()
        expected = self.scores.loc[self.scores["rank"].eq(int(self.normal["rank"]) + 1)].iloc[0]
        self.assertEqual(app.session_state["amp_explorer"]["customer_id"], expected.CONS_NO)
        self.assertEqual(exact_score(app), float(expected.risk_score))
        button(app, "Previous rank").click().run()
        self.assertEqual(app.session_state["amp_explorer"]["customer_id"], self.normal.CONS_NO)
        self.assertFalse(app.exception)

    def test_unavailable_history_preserves_score_and_back_navigation(self):
        with patch("app.views.inspector.load_history", side_effect=ArtifactError("Corrupt test history artifact")):
            app = self.opened()
            self.assertIn("history unavailable", app.warning[0].value)
            self.assertEqual(exact_score(app), float(self.normal.risk_score))
            self.assertEqual(len(app.date_input), 0)
            button(app, "Back to customer ranking").click().run()
            self.assertEqual(app.selectbox[1].value, self.normal.CONS_NO)
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
