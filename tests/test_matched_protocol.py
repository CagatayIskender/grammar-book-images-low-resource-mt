import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runners/matched'))
from protocol import prompt,parse,validate_rows,result_row,policy


class ProtocolTests(unittest.TestCase):
    def cfg(self,method='shot',kind='none'):
        return dict(method=method,language='Natugu',context_kind=kind,grammar_text='GRAMMAR' if kind=='text' else '',
                    support=[dict(source='EXAMPLE',gloss='GLOSS',reference='ENGLISH')],current_gloss='PREDICTED',
                    fingerprint='F',test=[dict(source='SOURCE',reference='REFERENCE')])

    def test_all_prompts_are_ablations(self):
        for method in ('shot','chain_gloss','modelgloss'):
            base=prompt(self.cfg(method),'SOURCE')
            for kind in ('text','image'):
                actual=prompt(self.cfg(method,kind),'SOURCE')
                self.assertEqual(base[0],actual[0])
                self.assertNotIn('REFERENCE',actual[1])
                self.assertNotIn('grammar reference',base[1])
                self.assertEqual(base[1].count('SOURCE'),1)
                if method=='chain_gloss':
                    block='The attached images are pages from a grammar reference for this language.\n' if kind=='image' else 'Here is a grammar reference summary for Natugu:\nGRAMMAR\n'
                    cleaned=actual[1].replace(block,'',1).replace('Use the grammar reference to produce the gloss first','Produce the gloss first',1)
                else:
                    block='\nThe attached images are pages from a grammar reference for this language.\n' if kind=='image' else '\nHere is a grammar reference summary for Natugu. Use it as supporting context:\nGRAMMAR\n'
                    cleaned=actual[1].replace(block,'',1).replace('Use the grammar reference as supporting context.\n','',1)
                self.assertEqual(cleaned,base[1])

    def test_parser_recovers_multiline_without_generation(self):
        out=parse('Gloss: one\ntwo\nFINAL_TRANSLATION: A sentence.','chain_gloss')
        self.assertEqual(out['prediction'],'A sentence.')
        self.assertEqual(out['generated_gloss'],'one\ntwo')
        self.assertIn('multiline_gloss',out['flags'])

    def test_empty_truncated_reasoning(self):
        for raw,truncated in [('',False),('Gloss: foo',False),('FINAL_TRANSLATION: partial',True),('<think>analysis</think>\nFINAL_TRANSLATION: text',False)]:
            self.assertEqual(parse(raw,'chain_gloss',truncated)['prediction'],'')

    def test_sparse_resume_and_duplicates(self):
        cfg=self.cfg();row=result_row(cfg,0,{'raw':'FINAL_TRANSLATION: OK'},'test')
        validate_rows([row],cfg,True)
        with self.assertRaises(ValueError):validate_rows([row,row],cfg)
        row['fingerprint']='different'
        with self.assertRaises(ValueError):validate_rows([row],cfg)

    def test_special_thinking_not_hidden(self):
        row=result_row(self.cfg(),0,dict(raw='FINAL_TRANSLATION: OK',raw_with_special_tokens='<think>x</think>'),'test')
        self.assertEqual(row['translation']['error'],'reasoning_violation')

    def test_methods_share_policy(self):
        for model in ('qwen3','qwen35','gemini25flashlite'):
            self.assertEqual(policy(model)['semantic_attempts'],1)
        self.assertEqual(policy('qwen3'),policy('qwen35'))

    def test_no_prompt_repair_in_execution(self):
        source=(ROOT/'runners/matched/run.py').read_text()
        self.assertNotIn('user +=',source)
        self.assertNotIn('Follow the requested output format exactly',source)

    def test_budget(self):
        import tempfile
        from run import API
        with tempfile.TemporaryDirectory() as directory:
            api=API.__new__(API);api.cap=4;api.ledger=Path(directory)/'budget.json'
            api.reserve(3)
            with self.assertRaises(RuntimeError):api.reserve(2)
            api.settle(3,0.5);api.reserve(3)
            data=json.loads(api.ledger.read_text())
            self.assertEqual(data['reserved_upper_bound_usd'],3.5)
            self.assertEqual(data['reported_cost_usd'],0.5)

    def test_tsez_cohorts(self):
        catalog=json.loads((ROOT/'configs/matched_v1/catalog.json').read_text())
        conditions=[json.loads((ROOT/item['config']).read_text()) for item in catalog if item['language']=='Tsez']
        full=next(cfg['test'] for cfg in conditions if cfg['model']=='qwen3')
        self.assertEqual(len(full),445)
        for cfg in conditions:
            expected=full if cfg['model'] in ('qwen3','qwen35') else full[:99]
            self.assertEqual(cfg['test'],expected)
            self.assertEqual(len(cfg.get('predicted_glosses',cfg['test'])),len(expected))

    def test_qwen_jobs_use_one_gpu_and_full_tsez(self):
        index=json.loads((ROOT/'configs/matched_v1/jobs.json').read_text())
        for job in index:
            if job['model'] not in ('qwen3','qwen35'):continue
            self.assertIn('#SBATCH --gres=gpu:1\n',(ROOT/job['job']).read_text())
            if job['language']=='tsez':
                paths=json.loads((ROOT/job['group']).read_text())
                self.assertLessEqual(len(paths),3)
                for path in paths:self.assertEqual(len(json.loads((ROOT/path).read_text())['test']),445)


if __name__=='__main__':unittest.main()
