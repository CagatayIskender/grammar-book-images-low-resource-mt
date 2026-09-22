"""Read-only checks for the complete corrected matrix and actual user prompts."""
import copy
import json
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners/matched_gold_v2"))
import protocol as p


class GoldSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "configs/matched_gold_v2/catalog.json").read_text())
        cls.configs = [json.loads((ROOT / item["config"]).read_text()) for item in cls.catalog]

    def test_all_351_conditions_have_actual_gold_examples(self):
        self.assertEqual(len(self.configs), 351)
        for cfg in self.configs:
            p.validate_support(cfg)
            active = dict(cfg, current_gloss=cfg.get("predicted_glosses", [""])[0])
            _, user = p.prompt(active, cfg["test"][0]["source"])
            for e in cfg["support"]:
                self.assertTrue(e["gloss"].strip())
                self.assertIn(f"Gloss: {e['gloss']}\n", user)

    def test_empty_gloss_rejected(self):
        cfg = copy.deepcopy(self.configs[0])
        cfg["support"][0]["gloss"] = " "
        with self.assertRaises(ValueError):
            p.prompt(cfg, cfg["test"][0]["source"])

    def test_tampered_gold_rejected(self):
        cfg = copy.deepcopy(self.configs[0])
        cfg["support"][0]["gloss"] = "invented gloss"
        with self.assertRaises(ValueError):
            p.validate_support(cfg)

    def test_only_support_and_provenance_changed(self):
        for cfg in self.configs:
            old = json.loads((ROOT / cfg["previous_config"]).read_text())
            for key in ("test", "policy", "context_files", "grammar_text", "model_id", "method"):
                self.assertEqual(cfg[key], old[key])
            self.assertEqual(cfg.get("predicted_glosses"), old.get("predicted_glosses"))
            from prepare import parse_igt
            manifest = json.loads((ROOT / cfg['gold_support_manifest']).read_text())
            training = parse_igt(Path(manifest['training_file']).read_text())
            self.assertEqual(cfg['support'], [training[i] for i in manifest['training_indices']])
            if cfg['language'] != 'Lezgi':
                self.assertEqual([(r['source'], r['reference']) for r in cfg['support']],
                                 [(r['source'], r['reference']) for r in old['support']])

    def test_no_gold_training_source_overlaps_test(self):
        for cfg in self.configs:
            self.assertFalse({r['source'] for r in cfg['support']} & {r['source'] for r in cfg['test']})
        cfg = copy.deepcopy(self.configs[0])
        cfg['test'][0]['source'] = cfg['support'][0]['source']
        with self.assertRaises(ValueError): p.validate_support(cfg)

    def test_gold_test_annotation_is_not_sent(self):
        for cfg in self.configs:
            active = copy.deepcopy(cfg)
            active["current_gloss"] = cfg.get("predicted_glosses", [""])[0]
            before = p.prompt(active, active["test"][0]["source"])
            active["test"][0]["gloss"] = "GOLD_TEST_SENTINEL_NOT_ALLOWED"
            self.assertEqual(before, p.prompt(active, active["test"][0]["source"]))
            self.assertNotIn("GOLD_TEST_SENTINEL_NOT_ALLOWED", before[1])

    def test_modelgloss_uses_predicted_test_gloss(self):
        for cfg in self.configs:
            if cfg["method"] == "modelgloss":
                self.assertEqual(len(cfg["test"]), len(cfg["predicted_glosses"]))
                predicted = cfg["predicted_glosses"][0]
                self.assertTrue(predicted.strip())
                _, user = p.prompt(dict(cfg, current_gloss=predicted), cfg["test"][0]["source"])
                self.assertIn(f"Gloss: {predicted}\n", user)

    def test_cohorts_and_single_gpu_policy(self):
        expected = {"Gitksan": 37, "Natugu": 99, "Lezgi": 87, "Tsez": 445}
        for cfg in self.configs:
            n = 99 if cfg["model"] == "gemini25flashlite" and cfg["language"] == "Tsez" else expected[cfg["language"]]
            self.assertEqual(len(cfg["test"]), n)
            if cfg["model"] != "gemini25flashlite":
                self.assertEqual(cfg["policy"]["gpu_count"], 1)
                self.assertEqual(cfg["policy"]["dtype"], "float32")
                self.assertFalse(cfg["policy"]["enable_thinking"])

    def test_baseline_context_support_identity(self):
        bases = {(c['model'], c['language'], c['method']): c for c in self.configs if c['material'] == 'baseline'}
        for cfg in self.configs:
            base = bases[(cfg['model'], cfg['language'], cfg['method'])]
            self.assertEqual(cfg['support'], base['support'])
            self.assertEqual(cfg.get('predicted_glosses'), base.get('predicted_glosses'))

    def test_missing_or_wrong_prompt_evidence_rejected(self):
        cfg = self.configs[0]
        active = dict(cfg, current_gloss=cfg.get('predicted_glosses', [''])[0])
        system, user = p.prompt(active, cfg['test'][0]['source'])
        attempt = {'raw': 'FINAL_TRANSLATION: A translation.', 'user_prompt_sha256': p.digest([system, user]),
                   'prompt_evidence': {'system': system, 'user': user, 'support_count': 21,
                                       'nonempty_gold_glosses': 21, 'support_sha256': cfg['gold_support_sha256']}}
        row = p.result_row(cfg, 0, attempt, {'type': 'test'})
        p.validate_rows([row], cfg)
        row['translation']['attempt']['prompt_evidence']['user'] = 'wrong'
        with self.assertRaises(ValueError): p.validate_rows([row], cfg)

    def test_generation_adapter_and_resume_without_network(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('gold_run_test', ROOT / 'runners/matched_gold_v2/run.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        calls = []
        class FakeAPI:
            def __init__(self, cfg): pass
            def generate(self, system, user, images, cfg):
                self_cfg = cfg
                for e in self_cfg['support']:
                    assert f"Gloss: {e['gloss']}\n" in user
                calls.append(user)
                return {'raw': 'Gloss: test\nFINAL_TRANSLATION: A generated translation.', 'truncated': False}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = copy.deepcopy(next(c for c in self.configs if c['model'] == 'gemini25flashlite' and c['material'] == 'baseline'))
            cfg['test'] = cfg['test'][:2]
            if 'predicted_glosses' in cfg: cfg['predicted_glosses'] = cfg['predicted_glosses'][:2]
            cfg['fingerprint'] = p.digest({k:v for k,v in cfg.items() if k != 'fingerprint'})
            (root / 'cfg.json').write_text(json.dumps(cfg))
            (root / 'group.json').write_text(json.dumps(['cfg.json']))
            # Only fixture storage/backend are substituted; actual prompt and row checks run.
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'API', FakeAPI), patch.object(runner, 'verify_inputs', p.validate_support):
                self.assertEqual(runner.run_group('group.json', 100), 0)
                self.assertEqual(runner.run_group('group.json', 100), 0)
            self.assertEqual(len(calls), 2)
            rows = [json.loads(line) for line in (root / cfg['results']).read_text().splitlines()]
            p.validate_rows(rows, cfg, complete=True)


if __name__ == '__main__':
    unittest.main()
