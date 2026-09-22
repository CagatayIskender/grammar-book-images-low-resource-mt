"""Path migration must preserve evidence and reject unrelated code changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('layout_test', ROOT / 'runners/artifact_layout.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class ArtifactLayoutTests(unittest.TestCase):
    def test_mapping_and_idempotence(self):
        for old, new in (
            ('metrics/matched_gold_v2/qwen3/test.json', 'metrics/matched_gold_v2/lexical_and_xcomet_xl/qwen3/test.json'),
            ('metrics/matched_gold_v2_xcomet_xxl/qwen3/test.json', 'metrics/matched_gold_v2/xcomet_xxl/qwen3/test.json'),
            ('metrics/matched_v1/test.json', 'metrics/matched_v1/test.json')):
            self.assertEqual(layout.metric_path(ROOT, old), ROOT / new)
            self.assertEqual(layout.metric_path(ROOT, ROOT / new), ROOT / new)
        for unsafe in ('../bad', 'metrics/matched_gold_v2/../../../bad', '/outside.json'):
            with self.assertRaises(ValueError):
                layout.metric_path(ROOT, unsafe)

    def test_all_moved_metrics_keep_their_hashes(self):
        records = json.loads((ROOT / 'docs/metric_layout_migration.json').read_text())['artifacts']
        self.assertEqual(len(records), 702)
        self.assertEqual(len({r['new_path'] for r in records}), 702)
        for entry in records:
            with self.subTest(path=entry['old_path']):
                new = layout.metric_path(ROOT, entry['old_path'])
                self.assertEqual(new, ROOT / entry['new_path'])
                self.assertFalse((ROOT / entry['old_path']).exists())
                self.assertEqual(hashlib.sha256(new.read_bytes()).hexdigest(), entry['sha256'])

    def test_approved_code_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / 'runner.py'
            snapshot = root / 'archive/code_before_metric_layout/runner.py'
            snapshot.parent.mkdir(parents=True)
            (root / 'docs').mkdir()
            snapshot.write_text('original')
            current.write_text('path adapter')
            old = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            new = hashlib.sha256(current.read_bytes()).hexdigest()
            record = dict(path='runner.py', snapshot=str(snapshot.relative_to(root)),
                          original_sha256=old, runtime_sha256=new)
            (root / 'docs/metric_layout_migration.json').write_text(json.dumps({'code_migrations': [record]}))
            self.assertEqual(layout.recorded_code_path(root, 'runner.py', old), snapshot)
            self.assertTrue(layout.equivalent_scorer(root, 'runner.py', old, new))
            self.assertFalse(layout.equivalent_scorer(root, 'runner.py', 'unrelated', new))
            current.write_text('unapproved code')
            with self.assertRaises(ValueError):
                layout.recorded_code_path(root, 'runner.py', old)
            self.assertFalse(layout.equivalent_scorer(root, 'runner.py', old, new))
            current.write_text('path adapter')
            snapshot.write_text('tampered original')
            with self.assertRaises(ValueError):
                layout.recorded_code_path(root, 'runner.py', old)


if __name__ == '__main__':
    unittest.main()
