import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xxl_scorer", ROOT / "runners/scoring/score_gold_v2_xcomet_xxl.py")
xxl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xxl)


class XXLTests(unittest.TestCase):
    def test_isolated_output(self):
        cfg = {"metrics": "metrics/matched_gold_v2/qwen3/test.json"}
        self.assertEqual(xxl.output_path(cfg), xxl.OUTPUT / "qwen3/test.json")
        with self.assertRaises(ValueError):
            xxl.output_path({"metrics": "metrics/matched_gold_v2/../../../escape.json"})

    def test_checkpoint_and_prediction_bound_reuse(self):
        identity = {"model": xxl.MODEL, "records": 2, "results_sha256": "a", "checkpoint_sha256": "b"}
        saved = {"provenance": identity, "xcomet_xxl": 0.5,
                 "segments": [{"idx": 0, "score": 0.4}, {"idx": 1, "score": 0.6}]}
        self.assertTrue(xxl.reusable(saved, identity))
        for key in ("results_sha256", "checkpoint_sha256", "model"):
            self.assertFalse(xxl.reusable(saved, dict(identity, **{key: "different"})))
        saved["segments"][1]["idx"] = 0
        self.assertFalse(xxl.reusable(saved, identity))

    def test_invalid_scores_not_reused(self):
        identity = {"model": xxl.MODEL, "records": 1}
        saved = {"provenance": identity, "xcomet_xxl": 0.4, "segments": [{"idx": 0, "score": 0.4}]}
        saved["segments"][0]["score"] = float("nan")
        self.assertFalse(xxl.reusable(saved, identity))


if __name__ == "__main__":
    unittest.main()
