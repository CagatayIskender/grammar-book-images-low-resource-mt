"""Close the supplementary study without hiding missing or human-gated work."""
from common import *
from generate import validate
from analyze import distribution, paired_summary, holm, metric_objects


def main():
    require_audit();plan=load(BASE/'configs/plan.json')
    complete={};statuses=[];scores=[]
    for entry in plan['generation']:
        parent=load(ROOT/entry['parent']);path=BASE/'results'/f'{entry["id"]}.jsonl'
        status=dict(id=entry['id'],records=0,expected=len(parent['test']),status='missing',detail='')
        try:
            rows=read_records(path);status['records']=len(rows)
            validate(rows,parent,entry,complete=True)
            if not path.exists():raise ValueError('Missing result')
            status['status']='complete'
            complete[entry['id']]=(dict(parent,results=str(path)),rows)
            cohorts=['valid84','sensitivity83'] if parent['language']=='Lezgi' else ['native','common99'] if parent['language']=='Tsez' else ['native']
            for cohort in cohorts:
                idx=mask(parent,cohort)
                for name,metric in metric_objects().items():
                    scores.append(dict(id=entry['id'],model=entry['model'],language=parent['language'],cohort=cohort,
                        metric=name,n=len(idx),seed=entry['seed'],budget=entry['budget'],oracle=entry['oracle'],
                        score=metric.corpus_score([rows[i]['translation']['prediction'] for i in idx],[[rows[i]['reference'] for i in idx]]).score,
                        empty=sum(not rows[i]['translation']['prediction'] for i in idx),
                        truncated=sum(rows[i]['translation']['attempt'].get('truncated',False) for i in idx),sha256=sha256(path)))
        except (ValueError,OSError,KeyError) as exc:
            status.update(status='partial_or_invalid',detail=str(exc))
        statuses.append(status)
    table('reports/generation_status.tsv',statuses)
    if scores:table('metrics/generation_scores.tsv',scores)
    comparisons=[]
    for pair in plan['generation_comparisons']:
        if pair['left'] not in complete or pair['right'] not in complete:continue
        left,lr=complete[pair['left']];right,rr=complete[pair['right']]
        if left['test']!=right['test'] or left['support']!=right['support']:raise ValueError('Unmatched new comparison')
        cohorts=['valid84','sensitivity83'] if left['language']=='Lezgi' else ['native','common99'] if left['language']=='Tsez' else ['native']
        for cohort in cohorts:
            idx=mask(left,cohort)
            for name in plan['metrics']:
                if pair['package']=='I':
                    metric=metric_objects()[name]
                    a=metric.corpus_score([lr[i]['translation']['prediction'] for i in idx],[[lr[i]['reference'] for i in idx]]).score
                    b=metric.corpus_score([rr[i]['translation']['prediction'] for i in idx],[[rr[i]['reference'] for i in idx]]).score
                    summary=dict(delta=a-b,ci_low='',ci_high='',p_raw='',p_holm='',family_tests='')
                else:
                    a,da,_=distribution(left,lr,idx,name,plan);b,db,_=distribution(right,rr,idx,name,plan)
                    summary=paired_summary(da,db,a-b)
                comparisons.append(dict(package=pair['package'],family=pair['package'],model=pair['model'],language=pair['language'],
                    seed=pair.get('seed',''),left=pair['left'],right=pair['right'],cohort=cohort,metric=name,n=len(idx),
                    left_score=a,right_score=b,**summary))
    # A missing comparison must not silently reduce the multiplicity family.
    for family in ('F','G'):
        family_rows=[r for r in comparisons if r['family']==family]
        expected=sum((4 if p['language'] in ('Lezgi','Tsez') else 2) for p in plan['generation_comparisons'] if p['package']==family)
        if len(family_rows)==expected:holm(family_rows)
        else:
            for r in family_rows:r.update(p_holm='',family_tests=expected)
    if comparisons:table('metrics/generation_comparisons.tsv',comparisons)
    changes=[]
    for name,expected in load(BASE/'reports/original_hashes.json').items():
        if not (ROOT/name).exists() or sha256(ROOT/name)!=expected:changes.append(name)
    audit_files={name:(BASE/f'reports/{name}').exists() for name in ('A_analysis.json','B_analysis.json','C_qwen3.json','C_qwen35.json')}
    report=dict(generation_complete=len(complete),generation_expected=len(plan['generation']),analysis_files=audit_files,
        original_changes=changes,human_gates=plan['human_gates'],main_study_unchanged=not changes,
        post_hoc=True,limitations=['Additional runs are not part of the original 351-condition matrix.',
        'Seed comparisons are descriptive; confidence intervals are pointwise.',
        'Missing tests do not imply nonsignificance or equivalence.'])
    save('reports/completion.json',report)
    text=['# Additional Study Status','',f"Complete generation conditions: {len(complete)}/{len(plan['generation'])}.",'',
          'Original results remain separate. See generation_status.tsv and metrics/generation_comparisons.tsv.','',
          '## Exclusions and Limitations','']+[f'- {key}: {value}' for key,value in plan['human_gates'].items()]
    destination('reports/REPORT.md').write_text('\n'.join(text)+'\n')
    if changes:raise RuntimeError('Original artifact hash changed')
    if len(complete)!=len(plan['generation']) or not all(audit_files.values()):raise SystemExit(2)


if __name__=='__main__':main()
