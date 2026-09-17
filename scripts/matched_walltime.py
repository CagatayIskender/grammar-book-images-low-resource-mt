"""Conservative continuation limits; never modify generation configurations."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runners/matched'))
from protocol import validate_rows
from experiment_io import read_records


def limits(config_paths):
    configs = [json.loads((ROOT / p).read_text()) for p in config_paths]
    first = configs[0]
    if first['model'] not in ('qwen3', 'qwen35'):
        return '10:00:00', 34200
    remaining = 0
    for cfg in configs:
        rows = read_records(ROOT / cfg['results'])
        validate_rows(rows, cfg)
        remaining += len(cfg['test']) - len(rows)
    hours = choose_hours(first['language'].lower(), first['method'],
                         [c['material'] for c in configs], remaining)
    return f'{hours:02d}:00:00', hours * 3600 - (600 if hours == 1 else 1800)


def choose_hours(language, method, materials, remaining):
    if method == 'chain_gloss' and 0 < remaining <= 100 and language != 'tsez':
        return 1
    if language == 'gitksan':
        return 4
    if language in ('lezgi', 'natugu'):
        return 6
    if language == 'tsez' and materials == ['summary_text_txt']:
        return 3
    return 10
