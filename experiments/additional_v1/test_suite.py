"""Local checks only: no network, model loading, API, or submission."""
import unittest
import copy
import numpy as np
from common import *
from generate import active_prompt, validate
from analyze import bootstrap_scores, paired_summary, holm, metric_objects


class Tests(unittest.TestCase):
    def test_scope(self):
        with self.assertRaises(ValueError):destination('../outside.json')

    def test_cluster_bootstrap(self):
        refs=['a small example sentence','another example here','a small example sentence']
        hyp=['a small example sentence','different example here','small sentence']
        src=['duplicate','other','duplicate'];n=11;seed=23
        for metric in metric_objects().values():
            observed,values,clusters=bootstrap_scores(metric,hyp,refs,src,n,seed)
            self.assertEqual(clusters,2)
            rng=np.random.default_rng(seed);groups=[[0,2],[1]]
            for value,picks in zip(values,rng.integers(0,2,size=(n,2))):
                idx=[i for k in picks for i in groups[k]]
                self.assertAlmostEqual(value,metric.corpus_score([hyp[i] for i in idx],[[refs[i] for i in idx]]).score,places=9)
            self.assertAlmostEqual(observed,metric.corpus_score(hyp,[refs]).score)
            self.assertEqual(paired_summary(values,values,0)['p_raw'],1)

    def test_holm(self):
        rows=[dict(family='a',p_raw=p) for p in (.01,.04,.03)]
        holm(rows);np.testing.assert_allclose([r['p_holm'] for r in rows],[.03,.06,.06])

    def test_masks(self):
        cfg=config(baseline('qwen3','Lezgi','shot'))
        self.assertEqual(len(mask(cfg,'valid84')),84)
        self.assertEqual(len(mask(cfg,'sensitivity83')),83)

    def test_all_parent_prompts(self):
        for model in MODELS:
            for language in LANGUAGES:
                for method in ('shot','chain_gloss','modelgloss'):
                    cfg=config(baseline(model,language,method))
                    system,user=active_prompt(cfg,dict(oracle=False),0)
                    reference=original.prompt(dict(cfg,current_gloss=cfg.get('predicted_glosses',[''])[0]),cfg['test'][0]['source'])
                    self.assertEqual((system,user),reference)
                    changed=copy.deepcopy(cfg)
                    changed['test'][0]['reference']='FORBIDDEN_REFERENCE_SENTINEL'
                    changed['test'][0]['gloss']='FORBIDDEN_GOLD_SENTINEL'
                    self.assertEqual((system,user),active_prompt(changed,dict(oracle=False),0))
                    self.assertTrue(all(f"Gloss: {s['gloss']}\n" in user for s in cfg['support']))
                cfg=config(baseline(model,language,'modelgloss'))
                system,user=active_prompt(cfg,dict(oracle=True),0)
                altered=copy.deepcopy(cfg);altered['current_gloss']=cfg['test'][0]['gloss']
                self.assertEqual((system,user),original.prompt(altered,cfg['test'][0]['source']))

    def test_resume_identity(self):
        cfg=config(baseline('qwen3','Natugu','shot'));entry=dict(oracle=False,fingerprint='test')
        system,user=active_prompt(cfg,entry,0)
        row=dict(idx=0,source=cfg['test'][0]['source'],reference=cfg['test'][0]['reference'],fingerprint='test',
            translation={'attempt':{'prompt_evidence':{'system':system,'user':user}}})
        validate([row],cfg,entry)
        with self.assertRaises(ValueError):validate([row,row],cfg,entry)
        with self.assertRaises(ValueError):validate([row],cfg,entry,True)
        bad=copy.deepcopy(row);bad['translation']['attempt']['prompt_evidence']['user']+='extra'
        with self.assertRaises(ValueError):validate([bad],cfg,entry)


if __name__=='__main__':unittest.main()
