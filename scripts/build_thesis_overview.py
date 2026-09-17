"""Render standalone thesis matrix images and audit queued-condition coverage.

Uses validated local status and Slurm metadata; never submits or generates.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/matched_v1/figures'
MODELS = ['qwen3', 'qwen35', 'gemini25flashlite']
MODEL_NAMES = ['Qwen3', 'Qwen3.5', 'Gemini 2.5 Flash Lite']
METHODS = ['shot', 'chain_gloss', 'modelgloss']
METHOD_NAMES = ['Gloss-shot', 'Chain-gloss', 'ModelGloss']
MATERIALS = ['cheatsheet_txt', 'cheatsheet_jpg', 'summary_tables_txt',
             'summary_tables_jpg', 'summary_text_txt', 'pdfpages_impactful_jpg']
MATERIAL_NAMES = ['Cheat sheet TXT', 'Cheat sheet JPG', 'Summary tables TXT',
                  'Summary tables JPG', 'Summary text TXT', 'PDF pages JPG']
SOURCES = [('Gitksan', 'pdf1_brown', 'original', 'Gitksan / Brown'),
           ('Gitksan', 'pdf2_rigsby', 'original', 'Gitksan / Rigsby'),
           ('Lezgi', 'grammar', 'original', 'Lezgi'),
           ('Natugu', 'grammar', 'original', 'Natugu'),
           ('Tsez', 'grammar', 'original', 'Tsez'),
           ('Lezgi', 'grammar', 'cyrillic', 'Lezgi / Cyrillic')]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
INK = '#1f2933'
LOW, HIGH = (242, 245, 247), (28, 100, 135)
LEFT, COL, WIDTH = 432, 159, 1900


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def color(ratio):
    return tuple(round(a + ratio * (b-a)) for a, b in zip(LOW, HIGH))


def put(draw, xy, text, size=22, bold=False, fill=INK, max_width=None):
    f = font(size, bold)
    if max_width is not None and draw.textlength(text, font=f) > max_width:
        raise ValueError(f'Text would overflow: {text}')
    draw.text(xy, text, font=f, fill=fill)


def center(draw, box, text, size=22, bold=False, fill=INK):
    x, y, w, h = box
    f = font(size, bold)
    bbox = draw.textbbox((0, 0), text, font=f)
    if bbox[2]-bbox[0] > w-8 or bbox[3]-bbox[1] > h-4:
        raise ValueError(f'Cell would overflow: {text}')
    draw.text((x+(w-(bbox[2]-bbox[0]))/2-bbox[0], y+(h-(bbox[3]-bbox[1]))/2-bbox[1]),
              text, font=f, fill=fill)


def classify(conditions, jobs, queue, budget_stopped=()):
    active = {x['name']: x for x in queue}
    membership = {}
    for job in jobs:
        for config in json.loads((ROOT/job['group']).read_text()):
            if config in membership:
                raise ValueError(f'Duplicate condition in jobs: {config}')
            membership[config] = 'matched_'+job['name']
    missing, rows = [], []
    for item in conditions:
        entry = dict(item)
        name = membership.get(item['config'])
        entry['queued_job'] = active.get(name, {}).get('job_id')
        if item['state'] == 'complete':
            entry['display_state'] = 'Complete'
        elif item['state'] == 'invalid':
            entry['display_state'] = 'Invalid'
            missing.append(item['id'])
        elif item['state'] == 'resource_unavailable':
            entry['display_state'] = 'Memory limit'
        elif name in active:
            entry['display_state'] = 'Queued'
        elif item['model'] in budget_stopped:
            entry['display_state'] = 'Budget limit'
        else:
            entry['display_state'] = 'Not queued'
            missing.append(item['id'])
        rows.append(entry)
    return rows, missing


def stopped_by_budget():
    path = ROOT/'docs/matched_v1/submissions.json'
    latest = {}
    if not path.exists():
        return []
    for job in json.loads(path.read_text()):
        latest[job['model']] = job['job_id']
    stopped = []
    for model, jid in latest.items():
        log = ROOT/'slurm_outputs'/f'{jid}.out'
        if not log.exists():
            continue
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size-16384))
            tail = handle.read().decode('utf-8', errors='replace')
        if 'API budget reached; no further requests sent' in tail:
            stopped.append(model)
    return stopped


def header(draw, title, stamp, subtitle):
    put(draw, (40, 26), title, 36, True)
    put(draw, (40, 80), subtitle, 23)
    put(draw, (40, 118), 'Snapshot: '+stamp+' | matched_v1 | Coverage, not translation quality.', 20)
    for i, name in enumerate(MODEL_NAMES):
        x = LEFT+i*3*COL
        center(draw, (x, 173, 3*COL, 39), name, 24, True)
        draw.line((x+7, 221, x+3*COL-7, 221), fill=INK, width=2)
        for j, method in enumerate(METHOD_NAMES):
            center(draw, (x+j*COL, 229, COL, 36), method, 17)


def rows_for(conditions, language, source, variant, materials):
    return [x for x in conditions if x['language']==language and x['source']==source
            and x['variant']==variant and x['material'] in materials]


def groups(conditions):
    result = [('Baseline / four languages', [x for x in conditions if x['material']=='baseline'])]
    result.extend((label, rows_for(conditions, lang, src, variant, MATERIALS))
                  for lang, src, variant, label in SOURCES)
    if sum(len(rows) for _, rows in result) != len(conditions):
        raise ValueError('Overview omitted or duplicated conditions')
    return result


def overview(conditions, stamp):
    image = Image.new('RGB', (WIDTH, 1150), 'white')
    draw = ImageDraw.Draw(image)
    header(draw, 'Experiment matrix / record coverage', stamp,
           '3 models x 117 conditions = 351 planned conditions. Cells: complete / planned conditions.')
    for row_index, (label, items) in enumerate(groups(conditions)):
        y = 282+row_index*73
        put(draw, (40, y+19), label, 24, True, max_width=LEFT-55)
        for mi, model in enumerate(MODELS):
            for si, method in enumerate(METHODS):
                subset = [x for x in items if x['model']==model and x['method']==method]
                complete = sum(x['state']=='complete' for x in subset)
                total = len(subset)
                if not total:
                    raise ValueError('Unplanned empty overview cell')
                ratio = complete/total
                x = LEFT+(3*mi+si)*COL
                draw.rectangle((x+2, y+2, x+COL-2, y+68), fill=color(ratio))
                center(draw, (x, y+3, COL, 62), f'{complete}/{total}', 25, True,
                       fill='white' if ratio>.65 else INK)
    y = 817
    put(draw, (40, y+12), 'TOTAL COMPLETE', 21, True)
    for i, model in enumerate(MODELS):
        total = sum(x['model']==model and x['state']=='complete' for x in conditions)
        center(draw, (LEFT+i*3*COL, y, 3*COL, 52), f'{total} / 117', 28, True)
    put(draw, (40, 892), 'Original materials: 6 types. Lezgi Cyrillic: 5 types (no separate PDF-page condition).', 22)
    put(draw, (40, 929), 'Tsez: Qwen 445, Gemini 99 sentences. Use the same first 99 for cross-model comparisons.', 22)
    put(draw, (40, 966), 'Complete records do not imply error-free translation. Empty predictions are marked * in the detailed matrix.', 22)
    put(draw, (40, 1003), 'Luna is excluded from the primary thesis scope. Its historical results remain unchanged.', 22)
    put(draw, (40, 1040), 'Source: validated configurations and results. This is an experiment count, not a Slurm job count.', 20)
    return image


def detailed(conditions, stamp):
    definitions = [(lang+' / baseline', lang, 'grammar', 'original', 'baseline')
                   for lang in ['Gitksan', 'Lezgi', 'Natugu', 'Tsez']]
    for lang, source, variant, label in SOURCES:
        for material, name in zip(MATERIALS, MATERIAL_NAMES):
            if variant=='cyrillic' and material=='pdfpages_impactful_jpg':
                continue
            definitions.append((label+' / '+name, lang, source, variant, material))
    lookup = {(x['model'], x['method'], x['language'], x['source'], x['variant'], x['material']): x
              for x in conditions}
    if len(definitions)*len(MODELS)*len(METHODS)!=len(conditions) or len(lookup)!=len(conditions):
        raise ValueError('Detailed matrix dimensions are inconsistent')
    image = Image.new('RGB', (WIDTH, 2925), 'white')
    draw = ImageDraw.Draw(image)
    header(draw, 'Experiment matrix / all primary conditions', stamp,
           'Each cell is one experiment: available / expected records; second line: execution status.')
    for ri, (label, lang, source, variant, material) in enumerate(definitions):
        y = 280+ri*62
        if ri in (4, 10, 16, 22, 28, 34):
            draw.line((40, y-2, WIDTH-35, y-2), fill=INK, width=2)
        # Two lines retain every source and material label without tiny text.
        group, item_label = label.rsplit(' / ', 1)
        put(draw, (40, y+6), group, 20, True, max_width=LEFT-55)
        put(draw, (40, y+32), item_label, 19, max_width=LEFT-55)
        for mi, model in enumerate(MODELS):
            for si, method in enumerate(METHODS):
                data = lookup[model, method, lang, source, variant, material]
                ratio = data['records']/data['expected']
                x = LEFT+(mi*3+si)*COL
                draw.rectangle((x+2, y+2, x+COL-2, y+58), fill=color(ratio))
                fg = 'white' if ratio>.65 else INK
                label = f"{data['records']}/{data['expected']}"+('*' if data['empty'] else '')
                center(draw, (x, y+3, COL, 28), label, 19, True, fg)
                center(draw, (x, y+31, COL, 24), data['display_state'], 14, fill=fg)
    put(draw, (40, 2740), '* At least one empty/unextractable prediction; failed predictions remain in the evaluation denominator.', 22)
    put(draw, (40, 2778), 'Color encodes record coverage only: light = incomplete, dark = complete. It does not encode quality.', 22)
    put(draw, (40, 2816), 'Gitksan PDF1/PDF2 share the same language/method baseline; that baseline is not counted twice.', 22)
    put(draw, (40, 2854), 'Luna is outside the primary scope. Metric validity and statistical significance require separate checks.', 22)
    return image


def main(without_slurm=False):
    status = json.loads((ROOT/'docs/matched_v1/status.json').read_text())
    if not status['validated_hashes']:
        raise ValueError('Run scripts/status_matched.py --validate first')
    queue = []
    if not without_slurm:
        output = subprocess.check_output(['squeue', '-u', os.environ.get('USER','ge92kun2'),
            '-h', '-o', '%i|%j|%T|%b|%E'], text=True)
        for line in output.splitlines():
            jid, name, state, gpu, dependencies = line.split('|', 4)
            if name.startswith('matched_'):
                queue.append(dict(job_id=jid, name=name, state=state, gpu=gpu, dependencies=dependencies))
    jobs = json.loads((ROOT/'configs/matched_v1/jobs.json').read_text())
    budget_stopped = stopped_by_budget()
    primary = [item for item in status['conditions'] if item['model'] in MODELS]
    conditions, missing = classify(primary, jobs, queue, budget_stopped)
    for item in queue:
        if item['gpu'] not in ('N/A', '(null)', 'gres/gpu:1'):
            raise ValueError('Unexpected GPU request: '+str(item))
    if without_slurm:
        for item in conditions:
            if item['display_state'] in ('Not queued','Queued'):
                item['display_state'] = 'Incomplete'
    stamp = datetime.fromisoformat(status['timestamp_utc']).astimezone(ZoneInfo('Europe/Berlin')).strftime('%d.%m.%Y %H:%M %Z')
    OUT.mkdir(parents=True, exist_ok=True)
    for name, renderer in [('experiment_matrix_overview', overview), ('experiment_matrix_detailed', detailed)]:
        picture = renderer(conditions, stamp)
        picture.save(OUT/(name+'.jpg'), quality=95, subsampling=0)
        picture.save(OUT/(name+'.pdf'), resolution=150)
    counts = Counter((x['model'], x['display_state']) for x in conditions)
    payload = dict(status_timestamp=status['timestamp_utc'], queue_checked=not without_slurm,
        queue=queue, missing_job_conditions=missing if not without_slurm else None,
        budget_stopped_models=budget_stopped,
        conditions=conditions, all_jobs_single_gpu=True if not without_slurm else None,
        generated_from='docs/matched_v1/status.json', included_models=MODELS,
        excluded_models=['gpt56luna'])
    (OUT/'matrix_data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Experiment and Queue Status', '', 'Snapshot: '+stamp, '',
             'Primary scope: Qwen3, Qwen3.5 and Gemini. Luna is excluded.',
             'Record completeness alone does not certify translation quality or XCOMET availability.',
             'See [the final audit](completion_audit_2026-09-17.md) for metric and comparability verification.', '',
             '| Model | Complete | Queued (incomplete/partial) | Budget-limited | Other |',
             '| --- | ---: | ---: | ---: | ---: |']
    for model, label in zip(MODELS, MODEL_NAMES):
        complete, queued, budget = (counts[model,key] for key in ['Complete','Queued','Budget limit'])
        lines.append(f'| {label} | {complete} | {queued} | {budget} | {117-complete-queued-budget} |')
    lines += ['', f'Uncovered conditions outside budget/memory limits: {len(missing)}.'
              if not without_slurm else 'Slurm was not checked.', '',
              'Each GPU job requests one GPU. Independent generation jobs may run concurrently.',
              'No new generation is required when all primary conditions are complete.', '',
              '| Job ID | Name | State | GPU |', '| --- | --- | --- | --- |']
    for job in queue:
        lines.append(f"| {job['job_id']} | {job['name']} | {job['state']} | {job['gpu']} |")
    (ROOT/'docs/matched_v1/CURRENT_STATUS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(conditions=len(conditions), queued_jobs=len(queue),
                         missing_jobs=len(missing) if not without_slurm else None, outputs=str(OUT)), indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--without-slurm', action='store_true')
    main(parser.parse_args().without_slurm)
