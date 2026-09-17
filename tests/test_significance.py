import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_baseline_significance import aligned_predictions, baseline_key, holm


class SignificanceTests(unittest.TestCase):
    def test_holm_known_values(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.8, 0.9]), [1.0, 1.0])

    def test_condition_mapping(self):
        self.assertEqual(baseline_key("qwen3_context_modelgloss"), "model_gloss")
        self.assertEqual(baseline_key("grammar_image_chain"), "chain_gloss")
        self.assertEqual(baseline_key("grammar_text_shot"), "gloss_shot")
        with self.assertRaises(ValueError):
            baseline_key("unknown")

    def test_pairing_and_no_failed_sentence_dropping(self):
        expected = [{"source":"a", "reference":"b"}, {"source":"c", "reference":"d"}]
        rows = [dict(ex, idx=i, shot={"prediction":"" if i else "b", "error":"truncated" if i else None})
                for i,ex in enumerate(expected)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            preds, quality = aligned_predictions(path, "shot", expected)
            self.assertEqual(preds, ["b", ""])
            self.assertEqual(quality, {"recorded_errors":1, "empty_predictions":1})
            with self.assertRaises(ValueError):
                aligned_predictions(path, "shot", list(reversed(expected)))
            with self.assertRaises(ValueError):
                aligned_predictions(path, "shot", expected[:1])


if __name__ == "__main__":
    unittest.main()
