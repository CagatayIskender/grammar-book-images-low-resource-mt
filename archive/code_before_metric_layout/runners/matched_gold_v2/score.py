"""Score complete matched results, including genuine model failures in the denominator."""
import argparse
import json
from protocol import ROOT, verify_inputs, validate_rows
from experiment_io import read_records, atomic_json
from run import basic_metrics


def main(xcomet=False):
    catalog=json.loads((ROOT/'configs/matched_gold_v2/catalog.json').read_text())
    targets=[];missing=[]
    for item in catalog:
        cfg=json.loads((ROOT/item['config']).read_text())
        rows=read_records(ROOT/cfg['results'])
        try:
            verify_inputs(cfg);validate_rows(rows,cfg,complete=True)
        except ValueError as exc:
            missing.append(dict(id=cfg['id'],reason=str(exc)));continue
        metric=ROOT/cfg['metrics']
        from experiment_io import sha256
        if not metric.exists() or json.loads(metric.read_text()).get('results_sha256')!=sha256(ROOT/cfg['results']):
            basic_metrics(cfg,rows)
        targets.append(dict(language=cfg['language'],results=cfg['results'],metrics=cfg['metrics'],expected_records=len(rows)))
    atomic_json(ROOT/'docs/matched_gold_v2/scoring_targets.json',targets)
    atomic_json(ROOT/'docs/matched_gold_v2/incomplete.json',missing)
    print(f'[SUMMARY] complete={len(targets)}, incomplete={len(missing)}',flush=True)
    if xcomet:
        from score_verified import score_targets
        count, problems = score_targets(targets,False)
        if problems: raise RuntimeError(f'XCOMET failures: {problems}')
    if missing: raise RuntimeError(f'{len(missing)} conditions incomplete; no final-completion claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--xcomet',action='store_true');main(p.parse_args().xcomet)

