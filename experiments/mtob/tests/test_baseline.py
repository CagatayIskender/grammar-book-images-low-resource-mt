import unittest
import tempfile
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from baseline import percentage, policy
from mtob_grammar.config import ROOT, sources
from mtob_grammar.experiment import output_paths, CONDITIONS
from mtob_grammar.prompts import appendix_c_prompt


class BaselineTests(unittest.TestCase):
    def test_only_context_removed(self):
        for s in sources().values():
            for retry in (False, True):
                original = appendix_c_prompt(s['language'], s['location'], 'SOURCE', 'UNIQUE_CONTEXT', refusal=retry)
                baseline = appendix_c_prompt(s['language'], s['location'], 'SOURCE', '', refusal=retry)
                self.assertEqual(original.replace('UNIQUE_CONTEXT', ''), baseline)
                self.assertEqual(baseline.count('SOURCE'), 2)

    def test_paths_and_existing_matrix_unchanged(self):
        self.assertEqual(CONDITIONS, ('ge', 'gs', 'gl'))
        for p in output_paths('qwen3', 'natugu', 'baseline'):
            self.assertIn(ROOT / 'baseline_v1', p.parents)
        for p in output_paths('qwen3', 'natugu', 'ge'):
            self.assertNotIn(ROOT / 'baseline_v1', p.parents)

    def test_percentage(self):
        self.assertEqual(percentage(20, 24), 20)
        self.assertEqual(percentage(.5, .4, True), 19.999999999999996)
        self.assertIsNone(percentage(0, 5))

    def test_frozen_settings(self):
        p = policy('qwen35', 'tsez')
        self.assertEqual(p['source']['test_n'],445)
        self.assertEqual(p['temperature'],.05)
        self.assertEqual(p['max_new_tokens'],256)
        self.assertEqual(p['gpus'],1)
        self.assertEqual(p['dtype'],'float32')
        self.assertFalse(p['thinking'])

    def test_generation_and_resume_do_not_rescue_finished_failures(self):
        from mtob_grammar import experiment
        from mtob_grammar.data import TranslationExample
        calls = []
        def generate(prompt, seed):
            calls.append((prompt, seed))
            text = 'A translated sentence.' if len(calls) == 1 else 'I cannot translate this.'
            return SimpleNamespace(text=text, metadata={})
        model = {'model_id': 'test/model'}
        source = {'test_n': 2, 'language': 'Test', 'location': 'Somewhere', 'test_file': 'unused'}
        examples = [TranslationExample(0,'source one','reference one'), TranslationExample(1,'source two','reference two')]
        with tempfile.TemporaryDirectory(dir=ROOT/'baseline_v1') as directory, ExitStack() as stack:
            path = Path(directory)/'results.jsonl'
            for name, value in [('model_config', model), ('source_config', source), ('parse_igt', examples),
                                ('input_path', Path('unused')), ('output_paths', (path,Path(directory)/'metrics.json')),
                                ('create_backend', SimpleNamespace(generate=generate))]:
                stack.enter_context(patch.object(experiment,name,return_value=value))
            stack.enter_context(patch.object(experiment,'evaluate',side_effect=AssertionError('No inline scoring')))
            enc = SimpleNamespace(encode=lambda text: text.split())
            result = experiment.run('qwen3','sample','baseline',encoding=enc)
            self.assertEqual(result['records'],2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows], ['ok','refusal'])
            self.assertEqual(len(calls),5)
            self.assertTrue(all('grammar book' not in p for p,_ in calls))
            result = experiment.run('qwen3','sample','baseline',encoding=enc)
            self.assertEqual(result['state'],'already_complete')
            self.assertEqual(len(calls),5)


if __name__ == '__main__':
    unittest.main()
