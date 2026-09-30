"""Validate and publish retained evidence; explicitly retire unexecuted conditions.

Run with --clean to remove only the enumerated failed/abandoned job artifacts.
The frozen scientific plan is preserved so exclusion and multiplicity remain visible.
"""
import argparse
import csv
import math
from pathlib import Path
import shutil
import sys
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from generate import validate
from analyze import metric_objects
sys.path.insert(0,str(BASE/'scoring'))
from score_xcomet import reusable


def read_table(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def main(clean=False):
    verify_suite()
    plan=load(BASE/'configs/plan.json')
    keep=[e for e in plan['generation'] if e['model']=='qwen3']
    excluded=[e for e in plan['generation'] if e['model']=='qwen35']
    hashes={};matrix=[];data={};total=Counter()
    lexical=read_table(BASE/'metrics/generation_scores.tsv')
    for e in excluded:
        path=BASE/'results'/f'{e["id"]}.jsonl'
        if path.exists() and path.stat().st_size:
            raise ValueError('Refusing to delete unexpected Qwen3.5 predictions; review needed')
    for e in keep:
        parent=load(ROOT/e['parent']);original.verify_inputs(parent)
        result=BASE/'results'/f'{e["id"]}.jsonl';rows=sorted(read_records(result),key=lambda r:r['idx'])
        validate(rows,parent,e,True);data[e['id']]=(parent,rows)
        hashes[str(result.relative_to(BASE))]=sha256(result)
        cohorts=['valid84','sensitivity83'] if parent['language']=='Lezgi' else ['native','common99'] if parent['language']=='Tsez' else ['native']
        for cohort in cohorts:
            idx=mask(parent,cohort)
            for name,metric in metric_objects().items():
                actual=metric.corpus_score([rows[i]['translation']['prediction'] for i in idx],[[rows[i]['reference'] for i in idx]]).score
                found=[r for r in lexical if (r['id'],r['cohort'],r['metric'])==(e['id'],cohort,name)]
                if len(found)!=1 or abs(actual-float(found[0]['score']))>1e-9:raise ValueError('Lexical score mismatch')
        for size in ('xl','xxl'):
            path=BASE/f'metrics/xcomet_{size}/{e["id"]}.json';artifact=load(path)
            identity=artifact['provenance']
            if identity['results_sha256']!=sha256(result) or identity['fingerprint']!=e['fingerprint'] or identity['records']!=len(rows):raise ValueError('COMET identity mismatch')
            if not reusable(artifact,identity):raise ValueError('COMET scores invalid')
            if identity['model']!='Unbabel/XCOMET-'+size.upper():raise ValueError('Wrong COMET model')
            hashes[str(path.relative_to(BASE))]=sha256(path)
            summaries=read_table(BASE/f'reports/xcomet_{size}_scores.tsv')
            for cohort in cohorts:
                idx=mask(parent,cohort);mean=sum(artifact['segments'][i]['score'] for i in idx)/len(idx)
                found=[r for r in summaries if (r['id'],r['cohort'])==(e['id'],cohort)]
                if len(found)!=1 or abs(float(found[0]['score'])-mean)>1e-12:raise ValueError('COMET mask summary mismatch')
        empty=sum(not r['translation']['prediction'] for r in rows)
        truncated=sum(bool(r['translation']['attempt'].get('truncated')) for r in rows)
        total.update(records=len(rows),empty=empty,truncated=truncated)
        matrix.append(dict(id=e['id'],model=e['model'],language=e['language'],records=len(rows),expected=len(parent['test']),
            empty=empty,truncated=truncated,lexical='verified',xcomet_xl='verified',xcomet_xxl='verified',
            results=str(result.relative_to(BASE)),results_sha256=sha256(result)))
    if len(matrix)!=37 or total['records']!=6049:raise ValueError('Retained scope changed')
    for package,count in [('A',72),('B',360)]:
        obj=load(BASE/f'reports/{package}_analysis.json')
        if obj['status']!='complete' or len(obj['rows'])!=count:raise ValueError('Incomplete A/B analysis')
    comparisons=read_table(BASE/'metrics/generation_comparisons.tsv')
    if len(comparisons)!=60 or any(r['model']!='qwen3' for r in comparisons):raise ValueError('Wrong comparison scope')
    if any(r['p_holm'] for r in comparisons if r['package'] in ('F','G')):raise ValueError('Do not shrink incomplete inference families')
    for pair in plan['generation_comparisons']:
        if pair['model']!='qwen3':continue
        left,lr=data[pair['left']];right,rr=data[pair['right']]
        if left['support']!=right['support'] or left['test']!=right['test']:raise ValueError('Unmatched pairs')
    changes=[];checked=0
    # Reader-facing documentation is deliberately revised; scientific artifacts are not.
    for name,checksum in load(BASE/'reports/original_hashes.json').items():
        p=ROOT/name
        scientific=(Path(name).is_absolute() or name.startswith(('configs/','results/','metrics/','runners/','materials/','inputs/','experiments/mtob/')))
        if scientific:
            checked+=1
            if not p.is_file() or sha256(p)!=checksum:changes.append(name)
    if changes:raise ValueError('Original scientific artifacts changed: '+str(changes[:5]))
    table('publication/condition_catalog.tsv',matrix)
    save('publication/artifact_hashes.json',hashes)
    save('publication/retained_manifest.json',dict(status='validated_retained_scope',conditions=keep,
        condition_count=len(keep),counts=dict(total),original_scientific_files_verified=checked,
        original_scientific_changes=[],original_planned_conditions=74,excluded_conditions=len(excluded),
        exclusion_reason='All new Qwen3.5 jobs failed before generation because the wrong environment was activated.',
        statistical_policy='Keep completed A/B families; F/G descriptive only with original incomplete families disclosed; I descriptive.',
        no_new_generation=True,no_new_hypothesis_tests=True))
    lines=['# Retained Supplementary Findings','',
        'Validated scope: 37 new Qwen3 conditions, 6,049 sentence-condition records; 192 truncated/empty outputs remain in all denominators.',
        'Both XCOMET collections and lexical scores were checked against the retained prediction hashes and cohort masks.',
        'Completed A/B/C analyses use historical results from both Qwen models; they are not new Qwen3.5 generations.','',
        '## Qwen3 Diagnostics','',
        'chrF++ differences on native cohorts (Lezgi valid84). F: 1024 minus 512 tokens. G: gold minus predicted gloss.',
        'These are descriptive, not Holm-significant claims: the original F/G families remain incomplete.','',
        '| Language | F: budget difference | G: oracle difference | I: context difference, three seeds |',
        '| --- | ---: | ---: | --- |']
    for lang in LANGUAGES:
        selected=[r for r in comparisons if r['language']==lang and r['metric']=='chrF++' and r['cohort'] in ('native','valid84')]
        values={p:[r for r in selected if r['package']==p] for p in ('F','G','I')}
        seeds=sorted(values['I'],key=lambda r:int(r['seed']))
        lines.append(f"| {lang} | {float(values['F'][0]['delta']):+.3f} | {float(values['G'][0]['delta']):+.3f} | "+', '.join(f"{float(r['delta']):+.3f}" for r in seeds)+' |')
    lines+=['','The larger budget changes little here; gold gloss is not a guaranteed performance upper bound.',
        'The three selected context differences are positive for Gitksan, Lezgi and Tsez and negative for Natugu across the tested seeds.',
        'This is useful diagnostic evidence, not proof of universal grammar benefits or general seed robustness.','',
        '## Completed Stored-Prediction Analyses','',
        'A: 46 of 72 metric tests have Holm-adjusted p < .05. This compares no-book methods, not grammar additions.',
        'B: four of 360 tests have Holm-adjusted p < .05 (two contrasts evaluated under two overlapping masks).',
        'Both significant context-baseline tests are negative chrF++ differences for Qwen3.5 Lezgi Cyrillic Chain-gloss cheat-sheet JPG.',
        'The two significant format tests favor Cyrillic summary-tables JPG over TXT on BLEU; they are not baseline gains.',
        'The 84/83 masks are sensitivity views, not independent replications. Exact estimates and p-values remain in reports/A_comparisons.tsv and reports/B_comparisons.tsv.','',
        'No XCOMET significance tests were performed. All completed learned-metric scores remain available, regardless of direction.']
    destination('publication/FINDINGS.md').write_text('\n'.join(lines)+'\n')
    removal=[]
    jobs=load(BASE/'configs/jobs.json')
    bad_groups={g for g in plan['groups'] if g.split('_')[1]=='qwen35'}
    for group in bad_groups:
        removal.extend([BASE/f'jobs/{group}.sh',BASE/f'configs/groups/{group}.json',BASE/f'results/{group}.lock'])
    for e in excluded:
        removal.extend((BASE/'results').glob(e['id']+'.*'))
    receipt_path=BASE/'reports/submitted_jobs.json'
    if receipt_path.exists():
        receipts=load(receipt_path)
        for key,value in receipts.items():
            if key in bad_groups or key=='report':removal.append(BASE/f'logs/{value["job_id"]}.out')
    removal.extend([BASE/'jobs/report.sh',BASE/'jobs/score_xcomet_xl.sh',BASE/'jobs/score_xcomet_xxl.sh'])
    for size in ('xl','xxl'):
        removal.extend([BASE/f'reports/xcomet_{size}_missing.json',BASE/f'reports/xcomet_{size}_status.json'])
    removal.extend([BASE/'reports/completion.json',BASE/'reports/generation_status.tsv',BASE/'reports/REPORT.md',
        BASE/'reports/submission_remaining.json',BASE/'reports/submitted_xcomet_jobs.json'])
    abandoned=ROOT/'experiments/gemini_decoding_v1'
    if abandoned.exists():removal.extend(p for p in abandoned.rglob('*') if p.is_file())
    removal=sorted(set(p for p in removal if p.exists()))
    if clean or not (BASE/'publication/cleanup_manifest.json').exists():
        save('publication/cleanup_manifest.json',dict(files=[str(p.relative_to(ROOT)) for p in removal],
            deletion_authorized=True,preserve_all_nonempty_qwen3_predictions=True))
    if clean:
        for p in removal:
            if not (p.resolve().is_relative_to(BASE) or p.resolve().is_relative_to(abandoned)):raise ValueError('Deletion outside scope')
            p.unlink()
        if abandoned.exists():
            for p in sorted(abandoned.rglob('*'),key=lambda p:len(p.parts),reverse=True):
                if p.is_dir():p.rmdir()
            abandoned.rmdir()
        save('configs/jobs.json',{k:v for k,v in jobs.items() if k not in bad_groups and k!='report'})
        if receipt_path.exists():save(receipt_path,{k:v for k,v in receipts.items() if k not in bad_groups and k!='report'})
        for name,checksum in hashes.items():
            if sha256(BASE/name)!=checksum:raise ValueError('Retained artifact changed during cleanup')
    print(dict(validated_conditions=len(matrix),counts=dict(total),files_to_remove=len(removal),cleaned=clean))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--clean',action='store_true');main(parser.parse_args().clean)
