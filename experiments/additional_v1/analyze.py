"""Paired source-cluster bootstrap using SacreBLEU's own corpus-statistic scorer."""
import argparse
from collections import defaultdict
import importlib.metadata
import os
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from common import *


def metric_objects():
    return {'BLEU': BLEU(), 'chrF++': CHRF(word_order=2)}


def cluster_stats(metric, hyps, refs, sources):
    stats=np.asarray(metric._extract_corpus_statistics(hyps,[refs]),dtype=np.float64)
    groups={}
    for i, source in enumerate(sources): groups.setdefault(source,[]).append(i)
    return np.asarray([stats[indices].sum(axis=0) for indices in groups.values()]), list(groups)


def bootstrap_scores(metric, hyps, refs, sources, samples, seed):
    stats, groups=cluster_stats(metric,hyps,refs,sources)
    rng=np.random.default_rng(seed)
    values=np.empty(samples)
    for start in range(0,samples,256):
        n=min(256,samples-start)
        picks=rng.integers(0,len(groups),size=(n,len(groups)))
        totals=stats[picks].sum(axis=1)
        values[start:start+n]=[metric._compute_score_from_stats(x.tolist()).score for x in totals]
    observed=metric.corpus_score(hyps,[refs]).score
    aggregate=metric._compute_score_from_stats(stats.sum(axis=0).tolist()).score
    if not np.isclose(observed,aggregate,atol=1e-10): raise ValueError('Corpus-statistic mismatch')
    return observed, values, len(groups)


def paired_summary(left, right, observed):
    delta=np.asarray(left)-np.asarray(right)
    centered=delta-observed
    p=(1+np.count_nonzero(np.abs(centered)>=abs(observed)-1e-12))/(len(delta)+1)
    low,high=np.quantile(delta,[.025,.975])
    return dict(delta=observed, ci_low=float(low), ci_high=float(high), p_raw=float(p))


def holm(rows):
    families=defaultdict(list)
    for r in rows: families[r['family']].append(r)
    for group in families.values():
        ordered=sorted(group,key=lambda r:r['p_raw']); running=0
        for rank,row in enumerate(ordered):
            running=max(running,(len(group)-rank)*row['p_raw'])
            row['p_holm']=min(1.,running); row['family_tests']=len(group)
    return rows


def distribution(cfg, rows, indices, metric_name, plan):
    hyps=[rows[i]['translation']['prediction'] for i in indices]
    refs=[rows[i]['reference'] for i in indices];sources=[rows[i]['source'] for i in indices]
    # Identical source/ref/ID cohorts share draws across systems and packages.
    cohort_hash=digest([indices,sources,refs])
    seed=int(digest([plan['seed'],cohort_hash])[:16],16)
    identity=dict(result_sha256=sha256(ROOT/cfg['results']), indices=indices, metric=metric_name,
        sacrebleu=importlib.metadata.version('sacrebleu'), samples=plan['samples'], seed=seed,
        code_sha256=sha256(Path(__file__)), lock_sha256=sha256(BASE/'configs/lock.json'))
    key=digest(identity); path=destination(f'cache/bootstrap_{key}.npz')
    metric=metric_objects()[metric_name]
    if path.exists():
        with np.load(path) as saved:
            if saved['identity'].item()!=json.dumps(identity,sort_keys=True): raise ValueError('Stale bootstrap cache')
            return float(saved['observed']),saved['values'].copy(),int(saved['clusters'])
    observed,values,clusters=bootstrap_scores(metric,hyps,refs,sources,plan['samples'],seed)
    tmp=path.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp,identity=json.dumps(identity,sort_keys=True),observed=observed,values=values,clusters=clusters)
    tmp.replace(path)
    return observed,values,clusters


def main(package):
    require_audit(); plan=load(BASE/'configs/plan.json')
    outputs=[]; configs={}; records={}
    contrasts=[r for r in plan['contrasts'] if r['package']==package]
    for num, pair in enumerate(contrasts,1):
        for side in ('left','right'):
            name=pair[side]
            if name not in configs:
                configs[name]=load(ROOT/name); records[name]=original_rows(configs[name])
        left,right=(configs[pair[k]] for k in ('left','right'))
        if left['support']!=right['support'] or left['policy']!=right['policy'] or left['test']!=right['test']:
            raise ValueError('Unmatched comparison inputs')
        indices=mask(left,pair['cohort'])
        for metric in plan['metrics']:
            a,da,clusters=distribution(left,records[pair['left']],indices,metric,plan)
            b,db,other=distribution(right,records[pair['right']],indices,metric,plan)
            if clusters!=other: raise ValueError('Cluster mismatch')
            outputs.append(dict(pair, metric=metric,n=len(indices),clusters=clusters,
                left_score=a,right_score=b,**paired_summary(da,db,a-b),
                ci_type='pointwise95_percentile',resampling='source_cluster',
                left_failures=sum(not records[pair['left']][i]['translation']['prediction'] for i in indices),
                right_failures=sum(not records[pair['right']][i]['translation']['prediction'] for i in indices)))
        print(f'[{package} {num}/{len(contrasts)}] {pair["id"]}',flush=True)
    holm(outputs)
    if len(outputs)!=len(contrasts)*2: raise ValueError('Missing planned tests')
    table(f'reports/{package}_comparisons.tsv',outputs)
    save(f'reports/{package}_analysis.json',dict(plan_sha256=sha256(BASE/'configs/plan.json'),
        original_hashes_sha256=sha256(BASE/'reports/original_hashes.json'),rows=outputs,
        status='complete',post_hoc=True,ci='pointwise, not Holm-adjusted',samples=plan['samples']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',choices=['A','B'],required=True)
    main(p.parse_args().package)
