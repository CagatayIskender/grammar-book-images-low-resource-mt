import copy
import importlib.metadata
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xl_analysis", ROOT / "runners/scoring/analyze_gold_v2_xcomet_xl.py")
xl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xl)


class XLTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"id": "fixture", "fingerprint": "f", "test": [{}, {}]}
        self.metric = {"fingerprint": "f", "translation": {"xcomet": 0.5},
            "xcomet_segments": {"translation": [{"idx": 0, "score": 0.4}, {"idx": 1, "score": 0.6}]},
            "xcomet_provenance": {"model": "Unbabel/XCOMET-XL", "records": 2,
                "results_sha256": "r", "checkpoint_sha256": "a" * 64,
                "scorer_sha256": xl.sha256(ROOT / "runners/score_verified.py"),
                "comet_version": importlib.metadata.version("unbabel-comet")}}

    def test_accept_aligned_scores(self):
        self.assertEqual(xl.verify_metric(self.metric, self.cfg, "r"), "a" * 64)

    def test_reject_old_aggregate_only(self):
        del self.metric["xcomet_segments"]
        with self.assertRaises(ValueError):
            xl.verify_metric(self.metric, self.cfg, "r")

    def test_reject_stale_or_wrong_model(self):
        for key in ("model", "results_sha256", "scorer_sha256"):
            changed = copy.deepcopy(self.metric)
            changed["xcomet_provenance"][key] = "wrong"
            with self.assertRaises(ValueError):
                xl.verify_metric(changed, self.cfg, "r")

    def test_reject_bad_segments(self):
        for segments in ([{"idx": 0, "score": 0.5}],
                         [{"idx": 0, "score": 0.4}, {"idx": 0, "score": 0.6}],
                         [{"idx": 0, "score": float("nan")}, {"idx": 1, "score": 0.6}],
                         [{"idx": 0, "score": 0.1}, {"idx": 1, "score": 0.2}]):
            self.metric["xcomet_segments"]["translation"] = segments
            with self.assertRaises(ValueError):
                xl.verify_metric(self.metric, self.cfg, "r")


if __name__ == "__main__":
    unittest.main()
