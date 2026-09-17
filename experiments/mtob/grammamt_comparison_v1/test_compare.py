"""Tests for matching, structural parsing, scoring and safe output boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare as c


def record(raw, prediction="", error=None):
    return {"idx": 0, "source": "source", "reference": "reference",
            "translation": {"raw": raw, "prediction": prediction, "error": error}}


class ComparisonTests(unittest.TestCase):
    def test_plain_multisentence(self):
        text = "She left. He stayed.\nThey met later."
        view = c.baseline_view(record(text), "shot")
        self.assertEqual(view["raw"], text)
        self.assertEqual(view["extracted"], text)

    def test_chain_structural_extraction(self):
        raw = "Gloss: PRON PAST walk\nFINAL_TRANSLATION: She walked.\nShe came home."
        view = c.baseline_view(record(raw), "chain_gloss")
        self.assertEqual(view["extracted"], "She walked.\nShe came home.")
        a, b = view["extracted_span"]
        self.assertEqual(raw[a:b], view["extracted"])

    def test_explanation(self):
        raw = "FINAL_TRANSLATION: She left.\n\n## Explanation:\nA past form."
        view = c.baseline_view(record(raw), "shot")
        self.assertIn("Explanation", view["raw"])
        self.assertEqual(view["extracted"], "She left.")
        a, b = view["extracted_span"]
        self.assertEqual(raw[a:b], view["extracted"])

    def test_missing_chain_translation(self):
        view = c.baseline_view(record("Gloss: repeated gloss", error="truncated;empty_or_unextractable_translation"), "chain_gloss")
        self.assertEqual((view["raw"], view["extracted"]), ("", ""))

    def test_failure_blank(self):
        for error in ("reasoning_violation", "refusal"):
            view = c.baseline_view(record("FINAL_TRANSLATION: not valid", error=error), "shot")
            self.assertEqual(view["raw"], "")
            self.assertEqual(view["extracted"], "")

    def test_ambiguous_final_fails(self):
        with self.assertRaises(ValueError):
            c.baseline_view(record("FINAL_TRANSLATION: one\nFINAL_TRANSLATION: two"), "shot")

    def test_ambiguous_explanation_preserved(self):
        raw = "Translation: First candidate.\nTranslation: Second candidate."
        view = c.baseline_view(record(raw), "shot")
        self.assertTrue(view["ambiguous_extraction"])
        self.assertEqual(view["extracted"], raw)

    def test_reference_blind(self):
        a = record("Translation: Actual text.\nNotes:\nDetails.")
        b = copy.deepcopy(a)
        b["reference"] = "A totally different gold answer."
        self.assertEqual(c.baseline_view(a, "shot")["extracted"], c.baseline_view(b, "shot")["extracted"])

    def test_complete_matching_required(self):
        expected = [{"source": "s", "reference": "r"}]
        cfg = {"test": expected}
        cfg["fingerprint"] = c.digest(cfg)
        row = {"idx": 0, "source": "s", "reference": "r", "fingerprint": cfg["fingerprint"]}
        c.validate_baseline([row], cfg, expected)
        for rows in ([], [row, row], [{**row, "reference": "wrong"}], [{**row, "idx": True}]):
            with self.assertRaises(ValueError):
                c.validate_baseline(rows, cfg, expected)

    def test_zero_baseline_percent(self):
        self.assertIsNone(c.relative_change(1, 0))
        self.assertEqual(c.relative_change(3, 2), 50)

    def test_test_family_and_no_duplicate_pairs(self):
        pairs = c.all_pairs()
        self.assertEqual(len(pairs), 90)
        self.assertEqual(len({(r["left"], r["right"]) for r in pairs}), 90)
        self.assertEqual(c.SETTINGS["bootstrap"]["family_size"], 360)
        self.assertEqual(len(c.baseline_keys()), 24)

    def test_metric_compatibility(self):
        predictions = ["One two three four five.", ""]
        references = ["One two three four five.", "Another correct sentence here."]
        metrics = c.evaluate(predictions, references)
        p, r = c.verify_score_inputs(predictions, references, metrics)
        self.assertEqual(len(p), 2)
        self.assertEqual(p[1], "")
        self.assertEqual(len(r), 2)

    def test_write_boundary(self):
        with self.assertRaises(ValueError):
            c.save(c.OUT.parent / "forbidden.json", {})

    def test_holm_family(self):
        adjusted = c.holm([0.0001, 0.01, 0.03] + [1.0] * 357)
        self.assertAlmostEqual(adjusted[0], 0.036)
        self.assertEqual(adjusted[1], 1)

    def test_stale_audit_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "provenance.json").write_text('{"fingerprint":"old"}')
            with patch.object(c, "OUT", out), patch.object(c, "provenance", return_value={"fingerprint": "new"}):
                with self.assertRaisesRegex(ValueError, "changed"):
                    c.checked_audit()


if __name__ == "__main__":
    unittest.main()
