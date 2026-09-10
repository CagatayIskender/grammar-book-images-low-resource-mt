import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners"))
from experiment_io import build_chain_prompt, dataset, parse_chain, validate_records


class ExperimentTests(unittest.TestCase):
    def test_chain_format(self):
        self.assertEqual(parse_chain("Gloss: man-ERG go-PST\nFINAL_TRANSLATION: The man went."), ("man-ERG go-PST", "The man went."))
        for raw in ("FINAL_TRANSLATION: hello", "Gloss: x\nFINAL_TRANSLATION:", "<think>hi</think>\nGloss: x\nFINAL_TRANSLATION: y", "Gloss: x\nFINAL_TRANSLATION: y\nExtra"):
            with self.assertRaises(ValueError):
                parse_chain(raw)

    def test_prompt_allows_gloss_not_reasoning(self):
        system, user = build_chain_prompt([], "Tsez", "SOURCE", "GRAMMAR")
        self.assertIn("Gloss: <generated gloss>", user)
        self.assertIn("FINAL_TRANSLATION:", user)
        self.assertNotIn("Do not include notes, glosses", user)
        self.assertIn("gloss", system)

    def test_chain_examples_omit_missing_glosses(self):
        support = [
            {"source":"a", "reference":"A", "gloss":"  "},
            {"source":"b", "reference":"B"},
            {"source":"c", "reference":"C", "gloss":"word-ERG"},
        ]
        _, user = build_chain_prompt(support, "Tsez", "TARGET", "GRAMMAR")
        self.assertIn("Tsez sentence: a\nFINAL_TRANSLATION: A", user)
        self.assertIn("Tsez sentence: b\nFINAL_TRANSLATION: B", user)
        self.assertIn("Tsez sentence: c\nGloss: word-ERG\nFINAL_TRANSLATION: C", user)
        self.assertNotRegex(user, r"(?m)^Gloss:\s*$")
        self.assertNotIn("English translation:", user)
        self.assertIn("English lexical meanings and grammatical abbreviations", user)
        self.assertIn("Do not copy the source sentence as the gloss", user)

    def test_revised_chain_applies_to_all_70_conditions(self):
        from run_audited_context import prompt_for
        configs = json.loads((ROOT / "configs/experiment_catalog.json").read_text())
        configs = [c for c in configs if c["condition"] == "chain_gloss"]
        self.assertEqual(len(configs), 70)
        for cfg in configs:
            support = dataset(cfg["language"], "train")[:21]
            ex = {"source":"TEST_SOURCE", "reference":"MUST_NOT_LEAK", "gloss":"MUST_NOT_LEAK_GLOSS"}
            _, user = prompt_for(cfg, support, ex, "GRAMMAR", "MUST_NOT_USE_MODEL_GLOSS")
            self.assertEqual(user.count("FINAL_TRANSLATION:"), len(support) + 1)
            self.assertNotRegex(user, r"(?m)^Gloss:\s*$")
            self.assertNotIn("MUST_NOT_", user)
            for item in support:
                self.assertIn(item["source"], user)
                self.assertIn(item["reference"], user)

    def test_full_dataset_required(self):
        expected = dataset("Tsez")
        rows = [dict(e, idx=i, fingerprint="a") for i,e in enumerate(expected[:9])]
        validate_records(rows, expected, fingerprint="a")
        with self.assertRaises(ValueError):
            validate_records(rows, expected, complete=True)
        with self.assertRaises(ValueError):
            validate_records(rows, expected, fingerprint="b")
        rows[1]["idx"] = 0
        with self.assertRaises(ValueError):
            validate_records(rows, expected)

    def test_catalog(self):
        configs = json.loads((ROOT / "configs/experiment_catalog.json").read_text())
        self.assertEqual(len(configs), 210)
        self.assertEqual(sum(c["condition"]=="chain_gloss" for c in configs), 70)
        self.assertEqual(sum(c["repair"] for c in configs), 5)
        for field in ("id", "results", "metrics", "job"):
            self.assertEqual(len({c[field] for c in configs}), len(configs))
        for c in configs:
            self.assertEqual(c["gpu_count"], 1)
            self.assertEqual(c["dtype"], "float32")
            self.assertEqual(len(dataset(c["language"])), c["expected_records"])
            for p in c["context_files"]:
                self.assertTrue((ROOT / p).is_file(), p)
            self.assertTrue((ROOT / c["job"]).is_file())
            if c["material"]=="pdfpages_impactful_jpg":
                count = 6 if c["source"]=="pdf2_rigsby" else 9
                self.assertEqual(len(c["context_files"]), count)

    def test_syntax(self):
        for root in ("runners", "scripts"):
            for p in (ROOT / root).rglob("*.py"):
                if not any(x in p.parts for x in ("cache", "__pycache__")):
                    ast.parse(p.read_text(), filename=str(p))
            for p in (ROOT / root).rglob("*.sh"):
                subprocess.run(["bash", "-n", str(p)], check=True, capture_output=True)

    def test_no_multigpu_active_jobs(self):
        import re
        for root in ("scripts", "experiments/mtob/jobs"):
            for p in (ROOT / root).rglob("*.sh"):
                text = p.read_text()
                for match in re.findall(r"#SBATCH\s+--(?:gres=gpu:|gpus=|gpus-per-node=)(\d+)", text):
                    self.assertEqual(match, "1", str(p))

    def test_stale_score_not_skipped(self):
        from experiment_io import sha256
        from score_verified import score_targets
        rows = [dict(e, idx=i, prediction_block={"prediction":"test"}) for i,e in enumerate(dataset("Gitksan"))]
        with tempfile.TemporaryDirectory() as folder:
            result, metric = Path(folder)/"test.jsonl", Path(folder)/"test.json"
            result.write_text("".join(json.dumps(r)+"\n" for r in rows))
            target = {"language":"Gitksan", "results":str(result), "metrics":str(metric)}
            m = {"prediction_block":{"xcomet":0.5}, "xcomet_provenance":{"model":"Unbabel/XCOMET-XL", "records":37, "results_sha256":"stale"}}
            metric.write_text(json.dumps(m))
            self.assertEqual(score_targets([target], dry_run=True)[0], 1)
            m["xcomet_provenance"]["results_sha256"] = sha256(result)
            metric.write_text(json.dumps(m))
            self.assertEqual(score_targets([target], dry_run=True)[0], 0)
            result.write_text(json.dumps(rows[0])+"\n")
            self.assertEqual(len(score_targets([target], dry_run=True)[1]), 1)

    def test_legacy_context_paths_exist(self):
        import re
        for p in (ROOT / "scripts/jobs").rglob("*.sh"):
            for value in re.findall(r'''(?:--grammar_image_dir|--grammar_text_file)\s+["']([^"']+)''', p.read_text()):
                if "$" not in value:
                    self.assertTrue(Path(value).exists(), f"{p}: {value}")

    def test_mtob_inputs(self):
        root = ROOT / "experiments/mtob"
        for c in json.loads((root / "configs/sources.json").read_text()).values():
            self.assertTrue((root / c["pdf"]).is_file(), c["pdf"])
            self.assertTrue((root / c["test_file"]).is_file(), c["test_file"])


if __name__ == "__main__":
    unittest.main()
