"""Matched paired tests and common-cohort model/material tables, no sentence dropping."""
import argparse
import collections
import csv
import json
import os
from protocol import ROOT, verify_inputs, validate_rows
from experiment_io import read_records, atomic_json, sha256
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest


def holm_with_missing(values, planned):
    result=[1.0]*len(values);running=0
    for rank,i in enumerate(sorted(range(len(values)),key=lambda i:values[i])):
        running=max(running,(planned-rank)*values[i]);result[i]=min(1.0,running)
    return result


def run(samples,models,cohort):
    output=ROOT/'docs/matched_v1'/f'analysis_{cohort}'
    output.mkdir(parents=True,exist_ok=True)
    catalog=[x for x in json.loads((ROOT/'configs/matched_v1/catalog.json').read_text()) if x['model'] in models]
    baselines={};systems=[];excluded=[];inputs={};scores=[];by_case={}
    for item in catalog:
        cfg=json.loads((ROOT/item['config']).read_text());path=ROOT/cfg['results']
        try:
            verify_inputs(cfg);rows=read_records(path);validate_rows(rows,cfg,complete=True)
        except (ValueError,OSError) as exc:
            excluded.append(dict(id=cfg['id'],reason=str(exc)));continue
        rows=sorted(rows,key=lambda r:r['idx'])
        if cohort=='common99' and cfg['language']=='Tsez':rows=rows[:99]
        refs=[r['reference'] for r in rows];hyps=[r['translation']['prediction'] for r in rows]
        inputs[cfg['results']]=sha256(path)
        group=(cfg['model'],cfg['language'],cfg['method'])
        data=(cfg,rows,hyps,refs)
        by_case[(cfg['model'],cfg['language'],cfg['source'],cfg['variant'],cfg['method'],cfg['material'])]=data
        if cfg['material']=='baseline':baselines[group]=data
        else:systems.append(data)
        scores.append(dict(id=cfg['id'],model=cfg['model'],language=cfg['language'],source=cfg['source'],variant=cfg['variant'],
                           method=cfg['method'],material=cfg['material'],records=len(rows),errors=sum(bool(r['translation']['error']) for r in rows),
                           bleu=BLEU().corpus_score(hyps,[refs]).score,chrf=CHRF(word_order=2).corpus_score(hyps,[refs]).score))
    groups=collections.defaultdict(list)
    for data in systems:
        cfg,rows,hyps,refs=data;group=(cfg['model'],cfg['language'],cfg['method']);base=baselines.get(group)
        if not base:
            excluded.append(dict(id=cfg['id'],reason='Missing complete matched baseline'));continue
        if refs!=base[3] or [r['source'] for r in rows]!=[r['source'] for r in base[1]] or cfg['policy']!=base[0]['policy'] or cfg['support']!=base[0]['support']:
            raise ValueError('Unmatched contrast')
        if cfg['method']=='modelgloss' and cfg['predicted_glosses']!=base[0]['predicted_glosses']:
            raise ValueError('Predicted gloss mismatch')
        groups[group].append(data)
    os.environ['SACREBLEU_SEED']='20260915';comparisons=[]
    for group,items in groups.items():
        base=baselines[group]
        names=[('Baseline',base[2])]+[(x[0]['id'],x[2]) for x in items]
        print('[PAIRED]',group,len(items),flush=True)
        signatures,values=PairedTest(names,{'BLEU':BLEU(),'chrF++':CHRF(word_order=2)},[base[3]],test_type='bs',n_samples=samples,n_jobs=1)()
        for metric,results in values.items():
            if metric=='System':continue
            for i,(cfg,rows,hyps,refs) in enumerate(items,1):
                value=results[i]
                comparisons.append(dict(id=cfg['id'],model=cfg['model'],language=cfg['language'],method=cfg['method'],
                    material=cfg['material'],source=cfg['source'],variant=cfg['variant'],metric=metric,
                    baseline_score=float(results[0].score),score=float(value.score),delta=float(value.score-results[0].score),
                    p=float(value.p_value),system_ci_half_width=float(value.ci),records=len(rows),signature=str(signatures[metric]),
                    errors=sum(bool(r['translation']['error']) for r in rows)))
    # Family fixed before results: all selected-model material-vs-baseline contrasts x two metrics.
    planned=sum(x['material']!='baseline' for x in catalog)*2
    adjusted=holm_with_missing([x['p'] for x in comparisons],planned)
    for row,p in zip(comparisons,adjusted):row['p_holm']=p
    material_tests=[];planned_material=0
    for item in catalog:
        if item['material'] not in ('cheatsheet_txt','summary_tables_txt'):continue
        planned_material+=2
        case=(item['model'],item['language'],item['source'],item['variant'],item['method'])
        text=by_case.get(case+(item['material'],));image=by_case.get(case+(item['material'].replace('_txt','_jpg'),))
        if not text or not image:continue
        if (text[3]!=image[3] or text[0]['policy']!=image[0]['policy']
                or text[0]['support']!=image[0]['support']
                or text[0].get('predicted_glosses')!=image[0].get('predicted_glosses')
                or [r['source'] for r in text[1]]!=[r['source'] for r in image[1]]):
            raise ValueError('Material pair mismatch')
        print('[MATERIAL PAIR]',case,item['material'],flush=True)
        sig,values=PairedTest([('Baseline',text[2]),('JPG',image[2])],{'BLEU':BLEU(),'chrF++':CHRF(word_order=2)},[text[3]],test_type='bs',n_samples=samples,n_jobs=1)()
        for metric,results in values.items():
            if metric=='System':continue
            material_tests.append(dict(model=item['model'],language=item['language'],source=item['source'],variant=item['variant'],
                method=item['method'],text_condition=item['material'],metric=metric,txt_score=float(results[0].score),jpg_score=float(results[1].score),
                delta_jpg_minus_txt=float(results[1].score-results[0].score),p=float(results[1].p_value),records=len(text[1]),signature=str(sig[metric])))
    for row,p in zip(material_tests,holm_with_missing([x['p'] for x in material_tests],planned_material)):row['p_holm']=p
    for path,h in inputs.items():
        if sha256(ROOT/path)!=h:raise ValueError('Results changed during analysis')
    for name,values in [('scores',scores),('comparisons',comparisons),('material_pairs',material_tests)]:
        if values:
            with (output/f'{name}.tsv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]),delimiter='\t');writer.writeheader();writer.writerows(values)
        else:
            (output/f'{name}.tsv').write_text('')
    atomic_json(output/'results.json',dict(cohort=cohort,models=models,samples=samples,seed=20260915,
        planned_holm_tests=planned,planned_material_holm_tests=planned_material,material_pairs=material_tests,
        missing_tests_treated_as_p1=True,comparisons=comparisons,excluded=excluded,inputs=inputs,
        limitations=['CIs are per-system, not paired-delta intervals.','No failed sentences removed.',
        'Common99 uses first 99 Tsez sentences, not a random sample.','Cross-model tables are descriptive; no causal architecture or interaction claim.',
        'Model policies differ across local/API backends but are fixed within model/method/material.',
        'Historical API imports have limited prompt/finish-reason provenance.','Not all text/JPG pairs are certified content-equivalent.',
        'No human adequacy evaluation or minimum practically important difference is supplied.',
        'Native Tsez uses 445 Qwen rows and 99 API rows; use common99 for cross-model Tsez comparisons.'],
        minimum_raw_p=1/(samples+1),holm_first_rejection_threshold=.05/max(1,planned),
        bootstrap_resolution_sufficient=1/(samples+1)<.05/max(1,planned)))
    lines=['# Matched Grammar-Context Results','',f'Paired bootstrap: {samples} resamples. Cohort: {cohort}.',
           'Failures remain in the denominator. Incomplete experiments are excluded, not replaced by a smaller intersection.','',
           '| Model | Complete experiments | Planned experiments | Context contrasts | Holm-positive metric tests |',
           '| --- | ---: | ---: | ---: | ---: |']
    for model in models:
        lines.append(f"| {model} | {sum(x['model']==model for x in scores)} | {sum(x['model']==model for x in catalog)} | {sum(x['model']==model for x in comparisons)//2} | {sum(x['model']==model and x['p_holm']<.05 and x['delta']>0 for x in comparisons)} |")
    lines+=['','This is a partial matrix whenever complete and planned counts differ. Luna may stop at the user-approved budget.',
            'Use analysis_common99/scores.tsv for descriptive cross-model comparisons on identical source cohorts.',
            'Native Tsez: Qwen3/Qwen3.5 use 445 rows; Gemini/Luna use the first 99 rows.',
            'comparisons.tsv measures context minus the same-model, same-method baseline.',
            'material_pairs.tsv tests TXT versus JPG; a representation-only interpretation additionally requires content equivalence.',
            'No model-interaction significance or causal architecture claim is made. Local and API decoding policies differ.',
            'Original GRAMMAMT prompt strategies are adapted, not reproduced verbatim. All methods retain the same first 21 support examples.',
            'No significant difference is not evidence of equivalence. See results.json for hashes, missing conditions and limitations.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('[SUMMARY]',len(comparisons),'metric comparisons;',len(excluded),'excluded',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=100000)
    p.add_argument('--models',nargs='+',default=['qwen3','qwen35','gemini25flashlite'])
    p.add_argument('--cohort',choices=['native','common99'],default='native');a=p.parse_args();run(a.samples,a.models,a.cohort)
