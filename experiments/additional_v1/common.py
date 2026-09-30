"""Isolated supplementary study: original artifacts are read-only."""
import csv
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / 'runners'))
sys.path.insert(0, str(ROOT / 'runners/matched_gold_v2'))
import protocol as original
from experiment_io import dataset, read_records, sha256
sys.path.insert(0, str(BASE))

MODELS = ('qwen3', 'qwen35')
LANGUAGES = ('Gitksan', 'Lezgi', 'Natugu', 'Tsez')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def destination(path):
    path = (BASE / path).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError('Write outside the additional study rejected')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save(path, value):
    target = destination(path)
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(target)


def table(path, rows):
    if not rows:
        raise ValueError(f'Empty table: {path}')
    with destination(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def catalog():
    return [x for x in load(ROOT / 'configs/matched_gold_v2/catalog.json') if x['model'] in MODELS]


def config(item):
    return load(ROOT / item['config'])


def baseline(model, language, method):
    return next(x for x in catalog() if (x['model'], x['language'], x['method'], x['material']) ==
                (model, language, method, 'baseline'))


def original_rows(cfg, verify=True):
    if cfg['model'] not in MODELS:
        raise ValueError('Gemini/API models excluded from this study')
    if verify:
        original.verify_inputs(cfg)
    rows = read_records(ROOT / cfg['results'])
    original.validate_rows(rows, cfg, complete=True)
    return sorted(rows, key=lambda r: r['idx'])


def mask(cfg, cohort):
    indices = list(range(len(cfg['test'])))
    if cfg['language'] == 'Lezgi':
        if cohort not in ('valid84', 'sensitivity83'):
            raise ValueError('Lezgi requires an explicit reference-quality mask')
        invalid = [i for i, r in enumerate(cfg['test']) if r['reference'].strip().lower() in ('', 'nan')]
        if invalid != [37, 62, 81]:
            raise ValueError('Lezgi reference exclusions changed')
        if cfg['test'][11]['reference'] not in cfg['support'][19]['reference']:
            raise ValueError('Lezgi overlap evidence changed')
        excluded = set(invalid) | ({11} if cohort == 'sensitivity83' else set())
        indices = [i for i in indices if i not in excluded]
    elif cohort == 'common99':
        if cfg['language'] != 'Tsez':
            raise ValueError('Do not double-count identical non-Tsez cohorts')
        indices = indices[:99]
    elif cohort != 'native':
        raise ValueError(cohort)
    return indices


def verify_suite():
    lock = load(BASE / 'configs/lock.json')
    for name, expected in lock['code'].items():
        if sha256(BASE / name) != expected:
            raise ValueError(f'Supplementary code changed after specification: {name}')
    for name, expected in lock.get('dependencies', {}).items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Imported runner changed after specification: {name}')
    if sha256(BASE / 'configs/plan.json') != lock['plan_sha256']:
        raise ValueError('Analysis plan changed after specification')
    return lock


def require_audit():
    lock = verify_suite()
    audit = load(BASE / 'reports/input_audit.json')
    if audit['status'] != 'passed' or audit['lock_sha256'] != sha256(BASE / 'configs/lock.json'):
        raise ValueError('Successful matching input audit required')
    return lock, audit
