"""Prepare a complete, versioned matrix and import eligible first responses only."""
import collections
import argparse
import csv
import json
from pathlib import Path
import sys
from protocol import ROOT, MODELS, METHODS, digest, policy, result_row, write_rows, sha256
from experiment_io import atomic_json, dataset, read_records
from run_grammamt_openrouter_context import load_glosslm_predictions


def reuse(cfg):
    model, method, lang = cfg['model'], cfg['method'], cfg['language'].lower()
    if model in ('qwen3','qwen35') and method=='chain_gloss':
        if cfg['material']=='baseline':
            old = ROOT / f'results/baseline/{model}/{lang}/grammar/original/chain_gloss_sampled_v1.jsonl'
        else:
            old = ROOT / cfg['old_config']['results']
            old = Path(str(old).replace('/chain_gloss_v2/qwen3/', f'/chain_gloss_v2/{model}/'))
            old = old.with_name(old.stem+'_sampled_v1.jsonl')
        pr = old.with_suffix('.provenance.json')
        if not pr.exists() or not old.exists():
            return []
        provenance = json.loads(pr.read_text())
        recorded_model=provenance.get('model',provenance.get('config',{}).get('model_id'))
        if recorded_model!=cfg['model_id']:return []
        if provenance.get('support')!=cfg['support'] or provenance.get('test',[])[:len(cfg['test'])]!=cfg['test']:
            return []
        if provenance.get('dtype')!='float32' or provenance.get('enable_thinking') is not False:
            return []
        sampling = provenance.get('sampling',provenance.get('policy',{}))
        if any(sampling.get(k)!=cfg['policy'][k] for k in ('temperature','top_p','top_k','presence_penalty','repetition_penalty')):
            return []
        required = ('runners/run_sampled_context.py','runners/run_audited_context.py','runners/experiment_io.py','runners/fp32_attention.py','runners/diagnose_tsez_chain.py')
        if any(provenance.get('code_hashes',{}).get(p)!=sha256(ROOT/p) for p in required):
            return []
        if cfg['material']!='baseline' and provenance.get('context_hashes')!=cfg['context_hashes']:
            return []
        old_fp=digest(provenance)
        output=[]
        for row in read_records(old):
            i=row['idx']
            if i>=len(cfg['test']):continue
            if row.get('fingerprint')!=old_fp or row['source']!=cfg['test'][i]['source'] or row['reference']!=cfg['test'][i]['reference']:
                raise ValueError('Historical first-response provenance mismatch')
            block=next(v for v in row.values() if isinstance(v,dict) and 'prediction' in v)
            a=block.get('attempts',[])
            if not a or a[0].get('seed')!=cfg['policy']['seed']+2*i:
                continue
            output.append(result_row(cfg,i,a[0],dict(type='historical_first_attempt',path=str(old.relative_to(ROOT)),sha256=sha256(old),excluded_later_attempts=len(a)-1)))
        return output
    if model=='gemini25flashlite' and method in ('shot','modelgloss') and cfg['variant']=='original':
        prefix='modelgloss_' if method=='modelgloss' else ''
        if cfg['material']=='baseline':
            old=ROOT/f'results/baseline/{model}/{lang}/grammar/original/results_{model}_{lang}_{prefix}baseline.jsonl'
        else:
            source=cfg['source'] if lang!='gitksan' else 'gitksan_'+cfg['source']
            if lang!='gitksan':source=lang
            old=ROOT/f'results/curated_v1/{model}/{lang}/{cfg["source"]}/original/results_{model}_{source}_curated_v1_{prefix}{cfg["material"]}.jsonl'
        audit=ROOT/'docs/comparability_audit/audit.json'
        if not old.exists() or not audit.exists():return []
        entry=next((r for r in json.loads(audit.read_text())['files'] if r['path']==str(old.relative_to(ROOT))),None)
        if not entry or entry.get('api_fingerprint_reconstructed') is not True or entry['input_sha256']!=sha256(old):
            return []
        rows=read_records(old)
        if len(rows)!=len(cfg['test']):return []
        migration={r['old_path']:r for r in csv.DictReader((ROOT/'docs/migration_manifest.tsv').open(),delimiter='\t')}
        recorded_paths=rows[0].get('grammar_image_paths',[])+([rows[0]['grammar_text_file']] if rows[0].get('grammar_text_file') else [])
        resolved=[]
        for name in recorded_paths:
            rel=name.split('/GRAMMAMT/',1)[-1];item=migration.get(rel)
            if item is None or sha256(ROOT/item['new_path'])!=item['sha256_before']:return []
            resolved.append(item['new_path'])
        if sorted(resolved)!=sorted(cfg['context_files']):return []
        output=[]
        for i,row in enumerate(rows):
            if row['source']!=cfg['test'][i]['source'] or row['reference']!=cfg['test'][i]['reference']:raise ValueError('API alignment')
            block=next(v for v in row.values() if isinstance(v,dict) and 'prediction' in v)
            if method=='modelgloss' and block.get('glosslm_pred_gloss')!=cfg['predicted_glosses'][i]:raise ValueError('Gloss mismatch')
            api=block.get('api',{})
            reasoning=(api.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens',0)
            output.append(result_row(cfg,i,dict(raw=block['raw'],api=api,reasoning_violation=bool(reasoning)),
                dict(type='audited_api_response',path=str(old.relative_to(ROOT)),sha256=sha256(old),limitations='No historical per-request prompt hash or finish_reason; audited settings and migration hashes.')))
        return output
    return []


def main(refresh=False):
    if refresh and (ROOT/'docs/matched_v1/submissions.json').exists():
        raise ValueError('Cannot refresh a submitted protocol')
    templates={}
    for path in sorted((ROOT/'configs/experiments').glob('*.json')):
        c=json.loads(path.read_text())
        if c['model']=='qwen3' and c['condition']=='chain_gloss':
            templates[(c['language'],c['source'],c['variant'],c['material'])]=c
    if len(templates)!=35:raise ValueError(f'Unexpected material matrix: {len(templates)}')
    code=['runners/matched/'+n for n in ('protocol.py','prepare.py','run.py')]
    code+=['runners/experiment_io.py','runners/run_sampled_context.py','runners/run_audited_context.py',
           'runners/fp32_attention.py','runners/diagnose_tsez_chain.py','runners/openrouter/run_grammamt_openrouter_context.py']
    hashes={p:sha256(ROOT/p) for p in code}
    catalog=[];groups=collections.defaultdict(list)
    for model in MODELS:
        cases=[(key,c) for key,c in templates.items()]
        cases += [((lang,'grammar','original','baseline'),None) for lang in ('Gitksan','Lezgi','Natugu','Tsez')]
        for (lang,source,variant,material),old in cases:
            for method in METHODS:
                test=dataset(lang)
                if lang=='Tsez':test=test[:99]
                files=old['context_files'] if old else []
                kind=old['context_kind'] if old else 'none'
                cfg=dict(model=model,model_id=MODELS[model],language=lang,source=source,variant=variant,material=material,
                         method=method,context_kind=kind,context_files=files,context_hashes={p:sha256(ROOT/p) for p in files},
                         grammar_text=(ROOT/files[0]).read_text().strip() if kind=='text' else '',
                         support=dataset(lang,'train')[:21],test=test,policy=policy(model),code_hashes=hashes,old_config=old,
                         protocol='matched_v1: fixed initial prompt; one semantic attempt; failed translations retained as empty')
                if method=='modelgloss':cfg['predicted_glosses']=load_glosslm_predictions(lang,ROOT/'runners/openrouter/cache')[:len(test)]
                if method=='modelgloss' and (len(cfg['predicted_glosses'])!=len(test) or any(not s for s in cfg['predicted_glosses'])):
                    raise ValueError('Missing predicted gloss')
                ident=f'{model}_{lang.lower()}_{source}_{variant}_{method}_{material}'
                cfg['id']=ident
                tail=f'matched_v1/{model}/{lang.lower()}/{source}/{variant}/{method}_{material}'
                cfg['results']='results/'+tail+'.jsonl';cfg['metrics']='metrics/'+tail+'.json'
                cfg['fingerprint']=digest(cfg)
                path=ROOT/'configs/matched_v1'/f'{ident}.json'
                changed=path.exists() and json.loads(path.read_text())!=cfg
                if changed and not refresh:raise ValueError('Frozen protocol changed; new revision required')
                if changed and any(r.get('origin',{}).get('type')=='generated_first_attempt' for r in read_records(ROOT/cfg['results'])):
                    raise ValueError('Cannot refresh generated experiments')
                atomic_json(path,cfg)
                reused=reuse(cfg)
                dest=ROOT/cfg['results']
                if reused and (not dest.exists() or changed):write_rows(dest,reused)
                rows=read_records(dest)
                catalog.append(dict(id=ident,model=model,language=lang,source=source,variant=variant,method=method,material=material,
                                    expected=len(test),records=len(rows),config=str(path.relative_to(ROOT)),results=cfg['results'],metrics=cfg['metrics']))
                groups[(model,lang.lower(),method)].append(str(path.relative_to(ROOT)))
    atomic_json(ROOT/'configs/matched_v1/catalog.json',catalog)
    for (model,lang,method),configs in groups.items():
        configs.sort(key=lambda p:('baseline' not in p,p))
        name=f'{model}_{lang}_{method}'
        atomic_json(ROOT/'configs/matched_v1/groups'/f'{name}.json',configs)
    print('Matrix',len(catalog),'conditions; imported complete',sum(x['records']==x['expected'] for x in catalog))
    print('By model',dict(collections.Counter(x['model'] for x in catalog)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-unsubmitted',action='store_true')
    main(parser.parse_args().refresh_unsubmitted)
