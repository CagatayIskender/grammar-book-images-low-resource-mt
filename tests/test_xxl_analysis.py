import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xxl_analysis", ROOT / "runners/scoring/analyze_gold_v2_xcomet_xxl.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class XXLAnalysisTests(unittest.TestCase):
    def test_identity_and_known_shift(self):
        same = analysis.paired_test([0, 0.3, 0.7], [0, 0.3, 0.7], 1000, 12)
        self.assertEqual(same["p"], 1)
        self.assertEqual(same["delta"], 0)
        shift = analysis.paired_test([0, 0.2, 0.4], [0.1, 0.3, 0.5], 1000, 12)
        self.assertAlmostEqual(shift["delta"], 0.1)
        self.assertAlmostEqual(shift["delta_ci_low"], 0.1)
        self.assertAlmostEqual(shift["p"], 1 / 1001)

    def test_deterministic_and_symmetric(self):
        a, b = [0, 0.1, 0.7, 0.2], [0.5, 0.1, 0.3, 0.4]
        first = analysis.paired_test(a, b, 1000, 24)
        self.assertEqual(first, analysis.paired_test(a, b, 1000, 24))
        reverse = analysis.paired_test(b, a, 1000, 24)
        self.assertAlmostEqual(first["p"], reverse["p"])
        self.assertAlmostEqual(first["delta"], -reverse["delta"])
        with self.assertRaises(ValueError):
            analysis.paired_test([0.1], [0.2, 0.3], 1000, 24)

    def test_holm_and_mismatched_inputs(self):
        for a, b in zip(analysis.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(a, b)
        with self.assertRaises(ValueError):
            analysis.check_pair({"support": ["one"]}, {"support": ["two"]})

    def test_full_plan_with_synthetic_equal_scores(self):
        systems = {}
        for item in json.loads((ROOT / "configs/matched_gold_v2/catalog.json").read_text()):
            cfg = json.loads((ROOT / item["config"]).read_text())
            systems[cfg["id"]] = (cfg, [0.5] * len(cfg["test"]))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(analysis, "OUT", output), patch.object(analysis, "load_inputs", return_value=(systems, {})):
                with contextlib.redirect_stdout(io.StringIO()):
                    analysis.main(samples=100)
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(len(result["comparisons"]), 630)
            self.assertEqual(len(result["material_pairs"]), 216)
            self.assertTrue(all(r["p_holm"] == 1 for r in result["comparisons"]))
            tsez = [r for r in result["comparisons"] if r["language"] == "Tsez" and r["model"] == "qwen3"]
            self.assertEqual({r["records"] for r in tsez if r["cohort"] == "native"}, {445})
            self.assertEqual({r["records"] for r in tsez if r["cohort"] == "common99"}, {99})


if __name__ == "__main__":
    unittest.main()
