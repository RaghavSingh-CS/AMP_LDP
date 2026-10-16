"""Source-value semantics, ID isolation, immutable-file checks and real histories."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from app.config import REPOSITORY_ROOT
from app.consumption_loader import load_history, parse_record, summarize, verify_file
from app.data_loader import ArtifactError, load_evidence, load_predictions
from app.views.inspector import consumption_chart


class SourceReadingTests(unittest.TestCase):
    def test_original_values_zeros_missing_and_statistics(self):
        readings = parse_record(b'0,,5,A,0\n', ["1/1/2014", "1/2/2014", "1/3/2014", "CONS_NO", "FLAG"], "A", 0)
        self.assertEqual(readings.consumption.iloc[0], 0)
        self.assertFalse(readings.missing.iloc[0])
        self.assertTrue(readings.missing.iloc[1])
        stats = summarize(readings)
        self.assertEqual((stats["observed"], stats["missing"], stats["zero_readings"]), (2, 1, 1))
        self.assertAlmostEqual(stats["missing_percent"], 100 / 3)
        self.assertEqual([stats[k] for k in ("mean", "median", "minimum", "maximum", "population_sd")], [2.5, 2.5, 0, 5, 2.5])

    def test_calendar_sorting_and_explicit_absent_column(self):
        readings = parse_record(b'7,2,A,1\n', ["1/3/2014", "1/1/2014", "CONS_NO", "FLAG"], "A", 1)
        self.assertTrue(readings.date.is_monotonic_increasing)
        self.assertEqual(readings.consumption.iloc[[0, 2]].tolist(), [2, 7])
        self.assertEqual(readings.status.iloc[1], "No source column")
        self.assertTrue(pd.isna(readings.consumption.iloc[1]))

    def test_no_cross_customer_or_label_substitution(self):
        headers = ["1/1/2014", "CONS_NO", "FLAG"]
        for content in [b"5,B,0\n", b"5,A,1\n"]:
            with self.subTest(content=content), self.assertRaises(ArtifactError):
                parse_record(content, headers, "A", 0)

    def test_invalid_source_values_or_schema(self):
        headers = ["1/1/2014", "CONS_NO", "FLAG"]
        for content in [b"-1,A,0\n", b"inf,A,0\n", b"text,A,0\n", b"5,A\n"]:
            with self.subTest(content=content), self.assertRaises(ArtifactError):
                parse_record(content, headers, "A", 0)

    def test_model_excluded_day_retains_source_reading(self):
        readings = parse_record(b"2,3,4,A,0\n", ["10/2/2014", "10/3/2014", "10/4/2014", "CONS_NO", "FLAG"], "A", 0)
        self.assertEqual(readings.consumption.tolist(), [2, 3, 4])
        self.assertEqual(readings.model_calendar.tolist(), [True, False, True])

    def test_chart_segments_do_not_bridge_missing_readings(self):
        readings = parse_record(b"1,,3,A,0\n", ["1/1/2014", "1/2/2014", "1/3/2014", "CONS_NO", "FLAG"], "A", 0)
        observed = readings.loc[~readings.missing]
        self.assertEqual(observed.segment.nunique(), 2)
        spec = consumption_chart(readings).to_dict()
        self.assertEqual(spec["vconcat"][0]["layer"][0]["encoding"]["detail"]["field"], "segment")
        self.assertEqual(spec["resolve"]["scale"]["x"], "shared")

    def test_all_missing_statistics_are_unavailable(self):
        readings = parse_record(b",,A,0\n", ["1/1/2014", "1/2/2014", "CONS_NO", "FLAG"], "A", 0)
        stats = summarize(readings)
        self.assertEqual(stats["missing_percent"], 100)
        self.assertIsNone(stats["mean"])
        consumption_chart(readings).to_dict()

    def test_file_hash_cache_does_not_hide_changes_or_deletion(self):
        with tempfile.TemporaryDirectory(prefix=".fixture-", dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_bytes(b"original")
            record = {"path": "source.csv", "sha256": hashlib.sha256(b"original").hexdigest()}
            self.assertEqual(verify_file(root, record), source)
            source.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ArtifactError, "SHA-256"):
                verify_file(root, record)
            source.unlink()
            with self.assertRaisesRegex(ArtifactError, "unavailable"):
                verify_file(root, record)


class RealHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores = load_predictions().frame
        cls.customers = [cls.scores.loc[cls.scores.FLAG.eq(label), "CONS_NO"].iloc[0] for label in (0, 1)]

    def test_all_histories_available_and_index_has_exact_score_membership(self):
        index = pd.read_csv(REPOSITORY_ROOT / "app/derived/stage2/history_index.csv", dtype={"CONS_NO": str})
        self.assertEqual(len(index), 6356)
        self.assertFalse(index.CONS_NO.duplicated().any())
        self.assertEqual(set(index.CONS_NO), set(self.scores.CONS_NO))
        self.assertEqual(index.set_index("CONS_NO").loc[self.scores.CONS_NO, "FLAG"].tolist(), self.scores.FLAG.tolist())
        self.assertTrue(index.observed.add(index.missing).eq(1035).all())

    def test_real_histories_both_classes_values_alignment_and_windows(self):
        catalog = json.loads((REPOSITORY_ROOT / "app/consumption_catalog.json").read_text())
        provenance = json.loads((REPOSITORY_ROOT / catalog["provenance"]["path"]).read_text())
        index = pd.read_csv(REPOSITORY_ROOT / catalog["index"]["path"], dtype={"CONS_NO": str}).set_index("CONS_NO")
        histories = [load_history(customer) for customer in self.customers]
        for customer, history in zip(self.customers, histories):
            with self.subTest(customer=customer):
                self.assertEqual(history.customer_id, customer)
                self.assertEqual(len(history.readings), 1035)
                self.assertTrue(history.readings.date.is_monotonic_increasing)
                self.assertTrue(history.windows.CONS_NO.eq(customer).all())
                self.assertEqual(len(history.windows), index.loc[customer, "usable_windows"])
                source = index.loc[customer]
                with (REPOSITORY_ROOT / catalog["source"]["path"]).open("rb") as handle:
                    handle.seek(int(source.byte_offset))
                    direct = parse_record(handle.read(int(source.byte_length)), provenance["source_headers"], customer, int(source.FLAG))
                pd.testing.assert_frame_equal(history.readings, direct)
                self.assertEqual(summarize(history.readings)["zero_readings"], int(source.zero_readings))
                self.assertEqual(history.readings.loc[history.readings.date.eq(pd.Timestamp("2016-09-18")), "status"].iloc[0], "No source column")
        self.assertFalse(histories[0].readings.consumption.equals(histories[1].readings.consumption))

    def test_invalid_or_nonfinal_customer_fails(self):
        for customer in ["", None, "not-a-customer", "../../data/raw"]:
            with self.subTest(customer=customer), self.assertRaises(ArtifactError):
                load_history(customer)

    def test_real_zero_readings_remain_observed(self):
        index = pd.read_csv(REPOSITORY_ROOT / "app/derived/stage2/history_index.csv", dtype={"CONS_NO": str})
        customer = index.loc[index.zero_readings.gt(0), "CONS_NO"].iloc[0]
        history = load_history(customer)
        zeros = history.readings.consumption.eq(0)
        self.assertGreater(int(zeros.sum()), 0)
        self.assertTrue(history.readings.loc[zeros, "status"].eq("Observed").all())
        self.assertTrue(history.readings.loc[history.readings.missing, "consumption"].isna().all())

    def test_missing_catalog_and_corrupt_catalog_fail_clearly(self):
        with tempfile.TemporaryDirectory(prefix=".fixture-", dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "app").mkdir()
            with patch("app.consumption_loader.load_predictions", return_value=load_predictions()):
                with self.assertRaisesRegex(ArtifactError, "unavailable"):
                    load_history(self.customers[0], root)
                (root / "app/consumption_catalog.json").write_text("{broken")
                with self.assertRaisesRegex(ArtifactError, "unavailable"):
                    load_history(self.customers[0], root)

    def test_repinned_index_cannot_link_another_real_customer(self):
        # Test only: tamper with a copied metadata index; original readings stay read-only.
        with tempfile.TemporaryDirectory(prefix=".fixture-", dir=Path(__file__).parent) as directory:
            root = Path(directory)
            catalog = json.loads((REPOSITORY_ROOT / "app/consumption_catalog.json").read_text())
            provenance = json.loads((REPOSITORY_ROOT / catalog["provenance"]["path"]).read_text())
            index = pd.read_csv(REPOSITORY_ROOT / catalog["index"]["path"], dtype=str)
            customer = self.customers[0]
            target = index.index[index.CONS_NO.eq(customer)][0]
            other = index.loc[index.CONS_NO.ne(customer) & index.FLAG.eq("0")].iloc[0]
            for name in ("byte_offset", "byte_length", "record_sha256"):
                index.loc[target, name] = other[name]
            index_path = root / catalog["index"]["path"]
            index_path.parent.mkdir(parents=True)
            index.to_csv(index_path, index=False, lineterminator="\n")
            catalog["index"]["sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
            provenance["index"] = deepcopy(catalog["index"])
            provenance_path = root / catalog["provenance"]["path"]
            provenance_path.write_text(json.dumps(provenance))
            catalog["provenance"]["sha256"] = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            (root / "app").mkdir(exist_ok=True)
            (root / "app/consumption_catalog.json").write_text(json.dumps(catalog))
            with patch("app.consumption_loader.load_predictions", return_value=load_predictions()), \
                 patch("app.consumption_loader.load_evidence", return_value=load_evidence()), \
                 patch("app.consumption_loader.verify_file", side_effect=lambda _, record: REPOSITORY_ROOT / record["path"]):
                with self.assertRaisesRegex(ArtifactError, "different customer"):
                    load_history(customer, root)


if __name__ == "__main__":
    unittest.main()
