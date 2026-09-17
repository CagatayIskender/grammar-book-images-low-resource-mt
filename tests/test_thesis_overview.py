import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('thesis_overview', ROOT/'scripts/build_thesis_overview.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ThesisOverviewTests(unittest.TestCase):
    def test_complete_matrix_and_overview_totals(self):
        status = json.loads((ROOT/'docs/matched_v1/status.json').read_text())
        conditions = [x for x in status['conditions'] if x['model'] in module.MODELS]
        self.assertEqual(len(conditions), 351)
        groups = module.groups(conditions)
        self.assertEqual(len(groups), 7)
        for index, (_, items) in enumerate(groups):
            expected = 4 if index==0 else 5 if index==6 else 6
            for model in module.MODELS:
                for method in module.METHODS:
                    selected = [x for x in items if x['model']==model and x['method']==method]
                    self.assertEqual(len(selected), expected)

    def test_budget_requires_evidence_and_active_job_takes_priority(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'group.json').write_text(json.dumps(['cfg.json']))
            job = dict(group='group.json', name='gpt56luna_budget')
            item = dict(id='example', model='gpt56luna', state='pending', config='cfg.json')
            with patch.object(module, 'ROOT', root):
                rows, missing = module.classify([item], [job], [])
                self.assertEqual(missing, ['example'])
                self.assertEqual(rows[0]['display_state'], 'Not queued')
                rows, missing = module.classify([item], [job], [], ['gpt56luna'])
                self.assertEqual(missing, [])
                self.assertEqual(rows[0]['display_state'], 'Budget limit')
                rows, missing = module.classify([item], [job],
                    [dict(name='matched_gpt56luna_budget', job_id='123')], ['gpt56luna'])
                self.assertEqual(rows[0]['display_state'], 'Queued')

    def test_duplicate_job_membership_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'group.json').write_text(json.dumps(['cfg.json']))
            with patch.object(module, 'ROOT', root):
                with self.assertRaises(ValueError):
                    module.classify([], [dict(group='group.json', name='one'),
                                         dict(group='group.json', name='two')], [])

    def test_image_outputs_are_nonblank(self):
        from PIL import Image, ImageStat
        for name in ('experiment_matrix_overview', 'experiment_matrix_detailed'):
            path = ROOT/'docs/matched_v1/figures'/f'{name}.jpg'
            with Image.open(path) as image:
                self.assertEqual(image.width, 1900)
                self.assertGreater(ImageStat.Stat(image.convert('L')).stddev[0], 20)


if __name__ == '__main__':
    unittest.main()
