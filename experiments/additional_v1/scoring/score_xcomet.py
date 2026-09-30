"""Separate XL/XXL scoring of additional-v1 outputs; original artifacts read-only."""
import argparse
import fcntl
import importlib.metadata
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BASE, ROOT, load, save, table, destination, require_audit, sha256, read_records, mask
from generate import validate


def reusable(saved, identity):
    segments = saved.get('segments', [])
    mean = saved.get('score')
    return (saved.get('provenance') == identity
        and len(segments) == identity['records']
        and [s.get('idx') for s in segments] == list(range(identity['records']))
        and all(isinstance(s.get('score'), (int, float)) and math.isfinite(s['score']) for s in segments)
        and isinstance(mean, (int, float)) and math.isfinite(mean)
        and abs(mean - sum(s['score'] for s in segments) / len(segments)) < 1e-12)


def targets(plan):
    valid, missing = [], []
    for entry in plan['generation']:
        try:
            parent = load(ROOT / entry['parent'])
            if sha256(ROOT / entry['parent']) != entry['parent_sha256']:
                raise ValueError('Parent configuration changed')
            result = BASE / 'results' / (entry['id'] + '.jsonl')
            before = sha256(result)
            rows = read_records(result)
            validate(rows, parent, entry, complete=True)
            rows = sorted(rows, key=lambda r: r['idx'])
            if any(not isinstance(r['translation']['prediction'], str) for r in rows):
                raise ValueError('Missing prediction field')
            if before != sha256(result):
                raise ValueError('Results changed during validation')
            valid.append((entry, parent, rows, result, before))
        except (OSError, ValueError, KeyError) as exc:
            missing.append(dict(id=entry['id'], reason=str(exc)))
    return valid, missing


def main(size, dry_run=False):
    require_audit()
    settings = load(BASE / 'configs/xcomet_extension.json')
    if settings['scorer_sha256'] != sha256(Path(__file__)):
        raise ValueError('Scorer changed after submission preparation')
    plan = load(BASE / 'configs/plan.json')
    if settings['plan_sha256'] != sha256(BASE / 'configs/plan.json'):
        raise ValueError('Generation plan changed')
    valid, missing = targets(plan)
    print(f'{size}: complete={len(valid)}, incomplete/invalid={len(missing)}', flush=True)
    if dry_run:
        return
    with destination(f'metrics/xcomet_{size}/scoring.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(f'reports/xcomet_{size}_missing.json', missing)
        if not valid:
            raise RuntimeError('No complete outputs; no scorer loaded')
        import torch
        if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Exactly one H100 required')
        checkpoint = Path(settings['models'][size]['checkpoint'])
        identity_base = dict(model=settings['models'][size]['model'], checkpoint_sha256=sha256(checkpoint),
            scorer_sha256=sha256(Path(__file__)), plan_sha256=settings['plan_sha256'],
            comet_version=importlib.metadata.version('unbabel-comet'),
            torch_version=importlib.metadata.version('torch'), precision='float32', batch_size=1, gpus=1,
            failure_policy='all expected rows, including empty translations', scale='native COMET, not percent')
        model = None; summaries = []
        for entry, parent, rows, result, result_hash in valid:
            identity = dict(identity_base, results_sha256=result_hash, fingerprint=entry['fingerprint'], records=len(rows))
            path = destination(f'metrics/xcomet_{size}/{entry["id"]}.json')
            artifact = load(path) if path.exists() else {}
            if not reusable(artifact, identity):
                if model is None:
                    from comet import load_from_checkpoint
                    model = load_from_checkpoint(str(checkpoint)).float().eval()
                start = time.monotonic()
                data = [{'src': r['source'], 'mt': r['translation']['prediction'], 'ref': r['reference']} for r in rows]
                output = model.predict(data, batch_size=1, gpus=1)
                scores = [float(s) for s in output['scores']]
                if len(scores) != len(rows) or not all(math.isfinite(s) for s in scores):
                    raise RuntimeError('Incomplete or nonfinite scores')
                if sha256(result) != result_hash:
                    raise RuntimeError('Results changed during scoring')
                artifact = dict(id=entry['id'], provenance=identity, score=sum(scores)/len(scores),
                    segments=[dict(idx=r['idx'], score=s) for r, s in zip(rows, scores)],
                    runtime_seconds=time.monotonic()-start)
                if not reusable(artifact, identity): raise RuntimeError('Artifact validation failed')
                save(path, artifact)
            if sha256(result) != result_hash: raise RuntimeError('Result changed before summary')
            cohorts = ['valid84', 'sensitivity83'] if parent['language']=='Lezgi' else ['native', 'common99'] if parent['language']=='Tsez' else ['native']
            for cohort in cohorts:
                indices = mask(parent, cohort)
                summaries.append(dict(id=entry['id'], model=entry['model'], language=entry['language'],
                    metric='XCOMET-'+size.upper(), cohort=cohort, n=len(indices),
                    score=sum(artifact['segments'][i]['score'] for i in indices)/len(indices),
                    empty_predictions=sum(not rows[i]['translation']['prediction'] for i in indices),
                    results_sha256=result_hash, score_path=str(path.relative_to(BASE))))
            print(f'[SCORED/VERIFIED {size}] {entry["id"]}', flush=True)
        table(f'reports/xcomet_{size}_scores.tsv', summaries)
        save(f'reports/xcomet_{size}_status.json', dict(status='complete' if not missing else 'partial',
            complete_conditions=len(valid), expected_conditions=len(plan['generation']), missing=missing,
            scale='native COMET; multiply by 100 only when explicitly labelled',
            significance='Not computed here; BLEU/chrF++ p-values do not apply to XCOMET'))
        if missing: raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['xl','xxl'],required=True)
    p.add_argument('--dry-run',action='store_true');args=p.parse_args();main(args.size,args.dry_run)
