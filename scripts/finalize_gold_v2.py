"""Build a read-only-input thesis handover; no generation, network, or model loading."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runners/matched_gold_v2'))
sys.path.insert(0, str(ROOT / 'runners/scoring'))
import protocol as p
from score_gold_v2_xcomet_xxl import output_path, provenance, reusable
from analyze_gold_v2_xcomet_xl import verify_metric
from sacrebleu.metrics import BLEU, CHRF
from PIL import Image, ImageDraw, ImageFont

OUT = ROOT / 'grammar_context_evaluation'
MODELS = ['qwen3', 'qwen35', 'gemini25flashlite']
NAMES = ['Qwen3-VL-8B', 'Qwen3.5-9B', 'Gemini 2.5 Flash Lite']
METHODS = ['shot', 'chain_gloss', 'modelgloss']
HASHES, STATS = {}, {}


def read(path):
    path = Path(path).resolve()
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed while reading: {path}')
    key = os.path.relpath(path, ROOT)
    HASHES[key] = hashlib.sha256(data).hexdigest()
    STATS[key] = (after.st_size, after.st_mtime_ns)
    return data


def checksum(path):
    key = os.path.relpath(Path(path).resolve(), ROOT)
    if key not in HASHES:
        read(path)
    return HASHES[key]


def load(path):
    return json.loads(read(path))


def tsv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def reference_indices(test, remove_overlap=False):
    invalid = [i for i, row in enumerate(test) if row['reference'].strip().lower() in ('nan', '')]
    if invalid != [37, 62, 81]:
        raise ValueError(f'Unexpected Lezgi reference exclusions: {invalid}')
    excluded = set(invalid) | ({11} if remove_overlap else set())
    return [i for i in range(len(test)) if i not in excluded]


def audit():
    catalog = load(ROOT / 'configs/matched_gold_v2/catalog.json')
    analyses = {}
    expected_hashes = {}
    for name in ('native', 'common99', 'xcomet_xl', 'xcomet_xxl'):
        base = ROOT / 'docs/matched_gold_v2' / f'analysis_{name}'
        a = load(base / 'results.json')
        if a.get('excluded') or len(a['comparisons']) != 630 or len(a['material_pairs']) != 216:
            raise ValueError(f'Incomplete statistical family: {name}')
        if a['samples'] != 100000:
            raise ValueError('Unexpected bootstrap count')
        for file, h in a['inputs'].items():
            if file in expected_hashes and expected_hashes[file] != h:
                raise ValueError('Analyses used different inputs')
            expected_hashes[file] = h
        for suffix in ('comparisons', 'material_pairs', 'scores'):
            rows = list(csv.DictReader(read(base / f'{suffix}.tsv').decode().splitlines(), delimiter='\t'))
            if suffix != 'scores':
                saved = a[suffix]
                if len(rows) != len(saved):
                    raise ValueError('TSV/JSON row mismatch')
                for row, source in zip(rows, saved):
                    for key, value in source.items():
                        if row[key] != str(value):
                            raise ValueError(f'TSV/JSON content mismatch: {name}/{key}')
        analyses[name] = a
    conditions, lezgi, scores = [], {}, []
    xl_checkpoints, xxl_checkpoints = set(), set()
    for num, item in enumerate(catalog, 1):
        cfg = load(ROOT / item['config'])
        if cfg['family'] != 'matched_gold_v2' or p.digest({k:v for k,v in cfg.items() if k != 'fingerprint'}) != cfg['fingerprint']:
            raise ValueError('Invalid configuration identity')
        for file, h in {**cfg['code_hashes'], **cfg['context_hashes']}.items():
            if checksum(ROOT / file) != h:
                raise ValueError(f'Frozen input changed: {file}')
        p.validate_support(cfg)
        checksum(ROOT / cfg['gold_support_manifest'])
        rows = [json.loads(line) for line in read(ROOT / cfg['results']).splitlines() if line.strip()]
        p.previous.validate_rows(rows, cfg, complete=True)
        rows.sort(key=lambda x:x['idx'])
        for row in rows:
            i = row['idx']
            active = dict(cfg, current_gloss=cfg.get('predicted_glosses', [''] * len(rows))[i])
            system, user = p.previous.prompt(active, row['source'])
            evidence = {'system': system, 'user': user, 'support_count': 21,
                        'nonempty_gold_glosses': 21, 'support_sha256': cfg['gold_support_sha256']}
            attempt = row['translation']['attempt']
            if (row.get('gold_support_sha256') != cfg['gold_support_sha256'] or
                    attempt.get('user_prompt_sha256') != p.digest([system, user]) or
                    attempt.get('prompt_evidence') != evidence):
                raise ValueError(f'Actual prompt mismatch: {cfg["id"]}/{i}')
        result_hash = checksum(ROOT / cfg['results'])
        metrics = load(ROOT / cfg['metrics'])
        xl_checkpoints.add(verify_metric(metrics, cfg, result_hash))
        xxl = load(output_path(cfg))
        checkpoint = xxl['provenance']['checkpoint_sha256']
        if xxl.get('id') != cfg['id'] or not reusable(xxl, provenance(cfg, result_hash, checkpoint)):
            raise ValueError(f'Invalid XXL artifact: {cfg["id"]}')
        xxl_checkpoints.add(checkpoint)
        hyps, refs = [r['translation']['prediction'] for r in rows], [r['reference'] for r in rows]
        lexical = {'bleu': BLEU().corpus_score(hyps, [refs]).score,
                   'chrf': CHRF(word_order=2).corpus_score(hyps, [refs]).score}
        if metrics['results_sha256'] != result_hash:
            raise ValueError('Stale basic metrics')
        for key, value in lexical.items():
            if not math.isclose(value, metrics['translation'][key], abs_tol=1e-8):
                raise ValueError(f'Basic metric mismatch: {cfg["id"]}/{key}')
        c = {k:cfg[k] for k in ('id','model','language','source','variant','method','material')}
        c.update(config=item['config'], results=cfg['results'], metrics=cfg['metrics'],
                 xxl_metrics=str(output_path(cfg).relative_to(ROOT)), records=len(rows), expected=len(cfg['test']),
                 state='complete', xl_verified=True, xxl_verified=True,
                 empty=sum(not h.strip() for h in hyps),
                 truncated=sum(bool(r['translation']['attempt'].get('truncated')) for r in rows),
                 missing_gloss=sum(cfg['method']=='chain_gloss' and not r['translation'].get('generated_gloss','').strip() for r in rows),
                 reasoning=sum('reasoning_violation' in (r['translation'].get('error') or '') for r in rows),
                 errors=sum(bool(r['translation'].get('error')) for r in rows))
        conditions.append(c)
        xl = [x['score'] for x in metrics['xcomet_segments']['translation']]
        xxl_values = [x['score'] for x in xxl['segments']]
        for cohort in ('native', 'common99'):
            n = 99 if cohort == 'common99' and cfg['language']=='Tsez' else len(rows)
            scores.append(dict(c, cohort=cohort, evaluated_records=n,
                               bleu=BLEU().corpus_score(hyps[:n], [refs[:n]]).score,
                               chrf_plus_plus=CHRF(word_order=2).corpus_score(hyps[:n], [refs[:n]]).score,
                               xcomet_xl=sum(xl[:n])/n, xcomet_xxl=sum(xxl_values[:n])/n))
        if cfg['language'] == 'Lezgi':
            lezgi[cfg['id']] = (cfg, hyps, xl, xxl_values)
        print(f'[AUDIT {num}/351] {cfg["id"]}', flush=True)
    if len(conditions) != 351 or sum(c['records'] for c in conditions) != 40731:
        raise ValueError('Wrong matrix dimensions')
    if len(xl_checkpoints) != 1 or len(xxl_checkpoints) != 1:
        raise ValueError('Mixed checkpoints')
    for file, h in expected_hashes.items():
        if checksum(ROOT / file) != h:
            raise ValueError(f'Statistics input hash mismatch: {file}')
    return conditions, scores, analyses, lezgi


def sensitivity(lezgi):
    rows = []
    for cfg, hyps, xl, xxl in lezgi.values():
        for cohort, remove in (('valid_reference84', False), ('valid_reference_no_overlap83', True)):
            indices = reference_indices(cfg['test'], remove)
            refs = [cfg['test'][i]['reference'] for i in indices]
            hs = [hyps[i] for i in indices]
            rows.append(dict(id=cfg['id'], model=cfg['model'], method=cfg['method'], material=cfg['material'],
                variant=cfg['variant'], cohort=cohort, records=len(indices), empty=sum(not h.strip() for h in hs),
                bleu=BLEU().corpus_score(hs,[refs]).score, chrf_plus_plus=CHRF(word_order=2).corpus_score(hs,[refs]).score,
                xcomet_xl=sum(xl[i] for i in indices)/len(indices), xcomet_xxl=sum(xxl[i] for i in indices)/len(indices)))
    baselines = {(r['model'],r['method'],r['cohort']):r for r in rows if r['material']=='baseline'}
    contrasts = []
    for row in rows:
        if row['material']=='baseline':
            continue
        b = baselines[row['model'],row['method'],row['cohort']]
        for metric in ('bleu','chrf_plus_plus','xcomet_xl','xcomet_xxl'):
            contrasts.append(dict(id=row['id'],cohort=row['cohort'],model=row['model'],method=row['method'],
                material=row['material'],variant=row['variant'],records=row['records'],metric=metric,
                baseline_score=b[metric],score=row[metric],delta=row[metric]-b[metric]))
    return rows, contrasts


def font(size, bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf', size)


def text(draw, box, value, size=22, fill='#17212b', bold=False):
    x,y,w,h=box; f=font(size,bold); bounds=draw.textbbox((0,0),value,font=f)
    if bounds[2]-bounds[0]>w-8 or bounds[3]-bounds[1]>h-4:
        raise ValueError(f'Figure text overflow: {value}')
    draw.text((x+4,y+4-bounds[1]),value,font=f,fill=fill)


def figures(conditions, analyses):
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    keys=list(dict.fromkeys((c['language'],c['source'],c['variant'],c['material']) for c in conditions))
    keys.sort(key=lambda k:(k[3]!='baseline',k[0],k[1],k[2],k[3]))
    lookup={(c['model'],c['method'],c['language'],c['source'],c['variant'],c['material']):c for c in conditions}
    short={'cheatsheet_txt':'Cheat sheet TXT','cheatsheet_jpg':'Cheat sheet JPG','summary_tables_txt':'Summary tables TXT',
           'summary_tables_jpg':'Summary tables JPG','summary_text_txt':'Summary text TXT','pdfpages_impactful_jpg':'Selected pages JPG','baseline':'No-book baseline'}
    width,left,col=2340,610,188
    image=Image.new('RGB',(width,310+len(keys)*67+170),'white');d=ImageDraw.Draw(image)
    text(d,(35,20,2250,60),'Final experiment matrix | matched_gold_v2',36,bold=True)
    text(d,(35,88,2250,40),'351/351 record-complete conditions | XL and XXL verified | Coverage is not translation quality.',24)
    for mi,name in enumerate(NAMES):
        text(d,(left+mi*3*col,157,3*col,45),name,25,bold=True)
        for j,label in enumerate(('Gloss-shot','Chain-gloss','ModelGloss')):
            text(d,(left+(3*mi+j)*col,214,col,35),label,20)
    for ri,key in enumerate(keys):
        y=280+67*ri
        lang,src,var,mat=key
        label=f'{lang} / '+({'pdf1_brown':'Brown PDF1','pdf2_rigsby':'Rigsby PDF2'}.get(src,var))
        text(d,(35,y,560,30),label,21,bold=True);text(d,(35,y+31,560,30),short[mat],20)
        for mi,model in enumerate(MODELS):
            for j,method in enumerate(METHODS):
                c=lookup[(model,method)+key];x=left+(mi*3+j)*col
                d.rectangle((x,y,x+col-7,y+60),fill='#14756b')
                text(d,(x+7,y+3,col-16,30),f'{c["records"]}/{c["expected"]}',22,'white',True)
                text(d,(x+7,y+33,col-16,26),f'Empty: {c["empty"]}',17,'white')
    y=290+67*len(keys)
    for i,line in enumerate(('21 gold-glossed training supports in every condition. Test gold gloss is not a target input.',
        'Tsez: Qwen 445 rows; Gemini first 99. Gitksan sources share one baseline per model/method.',
        'Lezgi: 87 generated rows include 3 invalid references; use separately labelled 84/83-row sensitivity tables.',
        'Source: grammar_context_evaluation/condition_catalog.tsv and grammar_context_evaluation/audit.json. Luna and MTOB are outside this matrix.')):
        text(d,(35,y+i*33,2270,32),line,21)
    save_figure(image,directory/'experiment_matrix_detailed')
    image=Image.new('RGB',(1550,740),'white');d=ImageDraw.Draw(image)
    text(d,(35,25,1460,60),'Production and evaluation complete',36,bold=True)
    text(d,(35,95,1460,40),'matched_gold_v2 | 351 conditions | 40,731 sentence-condition records',25)
    headers=['Model','Conditions','Records','Empty outputs','XL / XXL']
    xs=[35,485,720,960,1220];ws=[440,225,230,255,285]
    for x,w,label in zip(xs,ws,headers):text(d,(x,170,w,40),label,23,bold=True)
    for i,(model,label) in enumerate(zip(MODELS,NAMES)):
        cs=[c for c in conditions if c['model']==model];n=sum(c['records'] for c in cs);empty=sum(c['empty'] for c in cs)
        vals=[label,f'{len(cs)}/117',f'{n:,}',f'{empty} ({100*empty/n:.2f}%)','117 / 117']
        for x,w,v in zip(xs,ws,vals):text(d,(x,235+85*i,w,60),v,23)
        d.line((35,300+85*i,1500,300+85*i),fill='#ced5d9',width=2)
    for i,line in enumerate(('Complete coverage is not a claim that all model responses are successful.',
        'Full-cohort statistical tests are complete; Lezgi reference-quality limits remain explicit.',
        'No universal performance gain is established. Nonsignificance is not equivalence.',
        'Source: grammar_context_evaluation/audit.json. English handover; historical studies are not pooled.')):
        text(d,(35,520+i*42,1465,38),line,23)
    save_figure(image,directory/'experiment_matrix_overview')


def save_figure(image, path):
    image.save(path.with_suffix('.jpg'),quality=95,subsampling=0)
    image.save(path.with_suffix('.pdf'),resolution=150)


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    conditions,scores,analyses,lezgi=audit()
    filtered,contrasts=sensitivity(lezgi)
    summary=[]
    for name,a in analyses.items():
        for family in ('comparisons','material_pairs'):
            rows=a[family];delta='delta_jpg_minus_txt' if family=='material_pairs' and name in ('native','common99') else 'delta'
            summary.append(dict(analysis=name,family=family,tests=len(rows),
                significant_positive=sum(r['p_holm']<.05 and r[delta]>0 for r in rows),
                significant_negative=sum(r['p_holm']<.05 and r[delta]<0 for r in rows)))
    for filename,rows in (('condition_catalog',conditions),('scores',scores),('significance_summary',summary),
                          ('lezgi_sensitivity_scores',filtered),('lezgi_sensitivity_contrasts',contrasts)):
        tsv(OUT/f'{filename}.tsv',rows)
    figures(conditions,analyses)
    for file,(size,mtime) in STATS.items():
        st=(ROOT/file).stat()
        if (st.st_size,st.st_mtime_ns)!=(size,mtime):raise ValueError(f'Input changed: {file}')
    audit_report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),family='matched_gold_v2',
        conditions=len(conditions),records=sum(c['records'] for c in conditions),
        xl_verified=351,xxl_verified=351,actual_message_records_verified=40731,
        original_inputs_unchanged_during_build=True,inputs=HASHES,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistical_summary=summary,conditions_detail=conditions,
        limitations=['Qwen processor-serialized token sequences not independently reconstructed.',
        'This validates saved message evidence, not human translation quality.',
        'Image contents have not been independently OCR-audited for test overlap.',
        'Lezgi 84/83-row sensitivity is descriptive; no filtered-cohort p-values are claimed.',
        'Full 87-row Lezgi significance is reference-contaminated; not confirmatory evidence.',
        'Historical-file manifest verification is separate from this corrected-input audit.'])
    (OUT/'audit.json').write_text(json.dumps(audit_report,ensure_ascii=False,indent=2)+'\n')
    lines=['# Final Completion and Comparability Audit','',
        f'Verified: {audit_report["timestamp_utc"]}. Scope: matched_gold_v2 only.','',
        'All 351 conditions / 40,731 records pass complete-index, source/reference, frozen configuration,',
        '21 gold-support and saved actual system/user-message verification. Both XL and XXL artifacts',
        'match these predictions. Saved analysis input hashes and TSV/JSON contents agree.',
        'Basic BLEU and chrF++ were recomputed and matched. No model was loaded; no API or generation was run.','',
        '## Status by Model','', '| Model | Conditions | Records | Empty | Truncated | Missing chain gloss |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        cs=[c for c in conditions if c['model']==model]
        lines.append('| '+model+' | '+str(len(cs))+' | '+' | '.join(str(sum(c[k] for c in cs)) for k in ('records','empty','truncated','missing_gloss'))+' |')
    lines+=['','## Statistical Output Completeness','',
        '| Analysis | Family | Tests | Significant positive | Significant negative |','| --- | --- | ---: | ---: | ---: |']
    lines += [f'| {r["analysis"]} | {r["family"]} | {r["tests"]} | {r["significant_positive"]} | {r["significant_negative"]} |' for r in summary]
    lines+=['','Counts are tests, not independent discoveries. Cohorts overlap; correction families are separate.',
        'Full-cohort Lezgi tests include invalid references and must not establish a primary improvement claim.',
        'No significant difference is not equivalence. No cross-model interaction is tested.','',
        '## Reference-Quality Decision','',
        'The frozen 87-row Lezgi analyses are preserved as protocol outputs, not silently replaced.',
        'Use the 84-row valid-reference sensitivity for descriptive Lezgi performance; indices 37, 62 and 81',
        'are excluded solely because their source-dataset references are literal `nan`.',
        'The 83-row sensitivity additionally removes index 11, whose reference occurs in a longer training support.',
        'The same index masks apply to every baseline/context/model/method; empty predictions remain included.',
        'Scores are recomputed from the same predictions. XL/XXL use the corresponding saved sentence scores.',
        'These post hoc sensitivity results are descriptive only: no filtered-cohort significance or universal gain is claimed.',
        'Partial training/test overlap, duplicate rows and unassessed book/test overlap remain limitations.','',
        '## Readiness','',
        '**Share with caveats.** Production, scoring and planned tests are complete. No generation job remains',
        'necessary for this frozen study. Write findings as effects of this adapted context package, not exact',
        'GRAMMAMT replication or proven improvements across languages. Lezgi conclusions are descriptive.',
        'See audit.json for scope, hashes and remaining limits; see README.md for handover navigation.']
    (OUT/'COMPLETION_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:audit_report[k] for k in ('conditions','records','xl_verified','xxl_verified')}),flush=True)


if __name__=='__main__':
    build()
