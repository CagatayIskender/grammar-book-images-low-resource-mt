"""Verify corrected outputs, actual prompts, metrics and historical preservation."""
import json
from collections import Counter
from protocol import ROOT, verify_inputs, validate_rows
from prepare import historical_hashes
from experiment_io import read_records, sha256, atomic_json


def main():
    docs = ROOT / 'docs/matched_gold_v2'
    if historical_hashes() != json.loads((docs / 'historical_hashes.json').read_text()):
        raise ValueError('Historical results/configs/metrics changed')
    catalog = json.loads((ROOT / 'configs/matched_gold_v2/catalog.json').read_text())
    report, counts = [], Counter()
    for item in catalog:
        cfg = json.loads((ROOT / item['config']).read_text())
        entry = {'id': cfg['id'], 'model': cfg['model'], 'expected': len(cfg['test'])}
        try:
            verify_inputs(cfg)
            rows = read_records(ROOT / cfg['results'])
            validate_rows(rows, cfg)
            entry['records'] = len(rows)
            entry['state'] = 'complete' if len(rows) == len(cfg['test']) else 'partial' if rows else 'not_started'
            entry['empty_predictions'] = sum(not r['translation']['prediction'].strip() for r in rows)
            entry['xcomet_verified'] = False
            path = ROOT / cfg['metrics']
            if entry['state'] == 'complete' and path.exists():
                metrics = json.loads(path.read_text())
                provenance = metrics.get('xcomet_provenance', {})
                import math
                value = metrics.get('translation', {}).get('xcomet')
                entry['xcomet_verified'] = (metrics.get('fingerprint') == cfg['fingerprint']
                    and provenance.get('results_sha256') == sha256(ROOT / cfg['results'])
                    and provenance.get('records') == len(rows) and provenance.get('model') == 'Unbabel/XCOMET-XL'
                    and isinstance(value, (int, float)) and math.isfinite(value))
        except (ValueError, OSError, KeyError) as exc:
            entry.update(state='invalid', error=str(exc))
        counts[entry['state']] += 1
        report.append(entry)
    atomic_json(docs / 'status.json', {'counts': dict(counts), 'conditions': report, 'historical_hashes_unchanged': True})
    print(dict(counts), 'verified XCOMET', sum(r.get('xcomet_verified', False) for r in report))


if __name__ == '__main__':
    main()
