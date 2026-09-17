import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
import run_sampled_context as sampled
from experiment_io import atomic_json, sha256


class SampledTests(unittest.TestCase):
    def test_strict_parser(self):
        item = {"raw":"Gloss: child-ABS\nFINAL_TRANSLATION: The child.",
                "truncated":False, "raw_with_special_tokens":"", "repetition_warning":False}
        cfg = {"condition":"chain_gloss"}
        self.assertEqual(sampled.parse_output(cfg, item), ("child-ABS", "The child."))
        for key, value in (("truncated", True), ("repetition_warning", True),
                           ("raw_with_special_tokens", "<think>reason</think>")):
            with self.assertRaises(ValueError):
                sampled.parse_output(cfg, dict(item, **{key:value}))

    def test_path_does_not_overwrite_greedy(self):
        cfg = {"id":"example", "results":"results/chain_gloss_v2/example.jsonl"}
        self.assertTrue(str(sampled.result_path(cfg, False)).endswith("example_sampled_v1.jsonl"))
        self.assertIn("sampled_v1", str(sampled.result_path(cfg, True)))

    def test_fingerprint_includes_policy(self):
        a = {"policy":{"temperature":0.7}, "seed":1}
        b = copy.deepcopy(a)
        b["policy"]["temperature"] = 0.8
        self.assertNotEqual(sampled.fingerprint_of(a), sampled.fingerprint_of(b))

    def test_preflight_rejects_partial_stale_or_modified_output(self):
        cfg = {"id":"example", "prediction_key":"chain"}
        examples = [{"source":str(i), "reference":"reference"} for i in range(3)]
        provenance = {"test":examples, "policy":"sample"}
        fingerprint = sampled.fingerprint_of(provenance)
        with tempfile.TemporaryDirectory() as directory, patch.object(sampled, "ROOT", Path(directory)):
            path = sampled.result_path(cfg, True)
            path.parent.mkdir(parents=True)
            rows = [dict(ex, idx=i, fingerprint=fingerprint, chain={"prediction":"translation", "error":None})
                    for i, ex in enumerate(examples)]
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            atomic_json(path.with_suffix(".provenance.json"), provenance)
            status = {"status":"complete", "fingerprint":fingerprint, "records":3, "expected":3,
                      "results_sha256":sha256(path)}
            atomic_json(path.with_suffix(".status.json"), status)
            self.assertTrue(sampled.preflight_ready(cfg, provenance))
            self.assertFalse(sampled.preflight_ready(cfg, dict(provenance, policy="changed")))
            path.write_text("".join(json.dumps(r)+"\n" for r in rows[:2]))
            self.assertFalse(sampled.preflight_ready(cfg, provenance))
            path.write_text("".join(json.dumps(dict(r, chain={"prediction":"changed"}))+"\n" for r in rows))
            self.assertFalse(sampled.preflight_ready(cfg, provenance))


if __name__ == "__main__":
    unittest.main()
