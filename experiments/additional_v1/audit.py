"""Offline input audit and feasibility gates; no inference or external requests."""
import ast
from collections import Counter
import re
import subprocess
from common import *


def main():
    verify_suite()
    frozen = {}
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in tracked:
        path = ROOT / name
        if name and path.is_file() and not path.is_relative_to(BASE):
            frozen[name] = sha256(path)
    data_root = ROOT.parent / 'Database/2023glossingST/data'
    for language in LANGUAGES:
        for path in (data_root / language).glob('*'):
            if path.is_file(): frozen[str(path)] = sha256(path)
    aligned, coverage, material_rows, samples = [], [], [], []
    seen_vectors, seen_materials = set(), set()
    for num, item in enumerate(catalog(), 1):
        cfg = config(item); rows = original_rows(cfg)
        for name in (item['config'], cfg['results'], cfg['gold_support_manifest'], *cfg['context_files']):
            frozen[name] = sha256(ROOT/name)
        gold = dataset(cfg['language'])
        if len(gold) != len(cfg['test']): raise ValueError('Gold count mismatch')
        for idx, (a, b) in enumerate(zip(gold, cfg['test'])):
            if any(a.get(k) != b.get(k) for k in ('source','reference','gloss')):
                raise ValueError(f'Gold/test version mismatch: {cfg["id"]}/{idx}')
            if not a.get('gloss','').strip(): raise ValueError('Oracle needs complete gold glosses')
        for cohort in (('valid84','sensitivity83') if cfg['language']=='Lezgi' else ('native',)):
            selected = mask(cfg, cohort)
            aligned.append(dict(id=cfg['id'], cohort=cohort, records=len(selected),
                unique_sources=len({cfg['test'][i]['source'] for i in selected}),
                failures=sum(not rows[i]['translation']['prediction'].strip() for i in selected)))
        if cfg['method'] == 'modelgloss':
            vector = digest([cfg['language'], [(r['source'], r['reference']) for r in cfg['test']], cfg['predicted_glosses']])
            if vector not in seen_vectors:
                seen_vectors.add(vector)
                coverage.append(dict(kind='modelgloss', id=vector, language=cfg['language'], records=len(rows),
                    recoverable=sum(bool(g.strip()) for g in cfg['predicted_glosses']),
                    gate='source/order alignment verified; original cache lineage not independently certified; no formal accuracy claim'))
        if cfg['method'] == 'chain_gloss':
            coverage.append(dict(kind='chain_gloss', id=cfg['id'], language=cfg['language'], records=len(rows),
                recoverable=sum(bool(r['translation'].get('generated_gloss','').strip()) and not r['translation']['attempt'].get('truncated') for r in rows),
                gate='formal accuracy skipped: no manual extraction validation/evaluator'))
            # Deterministic hash sample, not selected for quality or significance.
            chosen=sorted(rows,key=lambda r:digest([20260925,cfg['id'],r['idx']]))[:3]
            samples.extend(dict(condition=cfg['id'], idx=r['idx'], source=r['source'],
                raw=r['translation']['attempt']['raw'], extracted=r['translation'].get('generated_gloss',''),
                reviewed=False) for r in chosen)
        for name in cfg['context_files']:
            if name in seen_materials: continue
            seen_materials.add(name)
            path = ROOT / name
            material_rows.append(dict(path=name, sha256=sha256(path), bytes=path.stat().st_size,
                kind=path.suffix, assessment='inventory only; source faithfulness not human-verified'))
        print(f'[AUDIT {num}/234] {cfg["id"]}',flush=True)
    table('reports/record_alignment.tsv', aligned)
    table('reports/gloss_feasibility.tsv', coverage)
    table('reports/material_inventory.tsv', material_rows)
    save('reviews/chain_extraction_sample.json', samples)
    official = ROOT.parent / 'Database/2023glossingST/baseline/src/eval.py'
    data_code = official.with_name('data.py')
    for p in (official, data_code): frozen[str(p)] = sha256(p)
    save('reports/gloss_evaluator_sources.json', dict(evaluator=str(official), evaluator_sha256=sha256(official),
        data_parser=str(data_code), data_parser_sha256=sha256(data_code),
        inspected_functions=[n.name for n in ast.parse(official.read_text()).body if isinstance(n,ast.FunctionDef)],
        decision='No official-equivalent gloss accuracy claimed without full normalization and manual extraction validation. Gold alignment does permit oracle inputs.'))
    # Existing MTOB evidence is inspected, never silently accepted as a current control.
    mtob = []
    for path in (ROOT/'experiments/mtob/results').rglob('*.jsonl'):
        if not ('qwen' in str(path).lower()): continue
        rows=read_records(path)
        mtob.append(dict(path=str(path.relative_to(ROOT)), records=len(rows),
            first_record_keys=sorted(rows[0]) if rows else [],
            gate='not certified: must establish old/current runtime, retries, and Natugu contamination policy'))
    save('reports/mtob_comparability_gate.json', dict(status='blocked_before_generation', candidates=mtob,
        required=['verified old/current model and processor revisions','prespecified verified Natugu overlap mask',
                  'matched retry policy or jointly regenerated contexts'], baseline_conditions=8))
    statuses = {'A':'ready','B':'ready','C':'ready; reconstructions labelled separately',
        'D':'alignment/coverage complete; formal accuracy skipped without reviewer',
        'E':'inventory complete; human source audit skipped by user',
        'F':'eight Qwen pairs, 512/1024; no Gemini pilot', 'G':'eight oracle/control pairs; gold identity verified',
        'H':'not submitted: reviewed matched-content inputs unavailable',
        'I':'eight prespecified model/language selections, three seeds each',
        'J':'skipped: no manually verified transcription targets',
        'K':'skipped by user: no source-language evaluator',
        'L':'not submitted: comparability/contamination gate unresolved'}
    save('reports/package_status.json', statuses)
    for name, expected in frozen.items():
        if sha256(ROOT/name)!=expected: raise ValueError(f'Original changed during audit: {name}')
    save('reports/original_hashes.json', frozen)
    save('reports/input_audit.json', dict(status='passed', conditions=len(catalog()),
        gold_alignment=True, original_files=len(frozen), lock_sha256=sha256(BASE/'configs/lock.json'),
        original_hashes_sha256=sha256(BASE/'reports/original_hashes.json')))
    print('[PASSED] Input/gold alignment; original artifacts unchanged',flush=True)


if __name__=='__main__': main()
