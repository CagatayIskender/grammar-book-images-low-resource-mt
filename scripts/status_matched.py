"""Audit the frozen matched matrix without submitting or generating anything."""
import argparse
import collections
import csv
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runners/matched'))
from protocol import validate_rows, verify_inputs
from experiment_io import atomic_json, read_records


def main(validate=False):
    catalog = json.loads((ROOT / 'configs/matched_v1/catalog.json').read_text())
    details = []
    summary = collections.defaultdict(collections.Counter)
    cohorts = collections.defaultdict(set)
    for item in catalog:
        cfg = json.loads((ROOT / item['config']).read_text())
        result = ROOT / cfg['results']
        state, error = 'pending', ''
        rows = []
        try:
            if validate:
                verify_inputs(cfg)
            rows = read_records(result)
            validate_rows(rows, cfg)
            state = 'complete' if len(rows) == len(cfg['test']) else 'partial' if rows else 'pending'
        except (ValueError, OSError, KeyError) as exc:
            state, error = 'invalid', str(exc)
        status_file = result.with_suffix('.status.json')
        if state in ('pending', 'partial') and status_file.exists():
            status = json.loads(status_file.read_text())
            if status.get('status') == 'resource_unavailable':
                state = 'resource_unavailable'
        generated = sum(isinstance(r.get('origin'), dict) and r['origin'].get('type') == 'generated_first_attempt' for r in rows)
        empty = sum(not r.get('translation', {}).get('prediction') for r in rows)
        entry = dict(id=cfg['id'], model=cfg['model'], language=cfg['language'], source=cfg['source'],
                     variant=cfg['variant'], method=cfg['method'], material=cfg['material'], state=state,
                     expected=len(cfg['test']), records=len(rows), generated=generated, imported=len(rows)-generated,
                     empty=empty, validation_error=error, config=item['config'], results=cfg['results'], metrics=cfg['metrics'])
        details.append(entry)
        counts = summary[cfg['model']]
        counts['planned'] += 1
        counts[state] += 1
        for key in ('expected', 'records', 'generated', 'imported', 'empty'):
            counts[key] += entry[key]
        cohorts[(cfg['model'], cfg['language'])].add(len(cfg['test']))
    out = ROOT / 'docs/matched_v1'
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'status.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                validated_hashes=validate, summary=dict(summary), conditions=details))
    with (out / 'experiment_catalog.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(dict(summary), indent=2))
    print('Cohort sizes:')
    for key, sizes in sorted(cohorts.items()):
        print('/'.join(key), sorted(sizes))
    if any(r['state'] == 'invalid' for r in details):
        raise SystemExit('Invalid conditions detected; see docs/matched_v1/status.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    main(parser.parse_args().validate)
