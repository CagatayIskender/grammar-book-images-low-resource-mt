"""User-requested cohort correction before Qwen Tsez jobs start; APIs remain at 99."""
import json
from pathlib import Path
import sys
from matched_walltime import limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runners/matched'))
from protocol import digest,write_rows,verify_inputs
from prepare import reuse
from experiment_io import atomic_json,dataset,read_records
from run_grammamt_openrouter_context import load_glosslm_predictions


def main():
    catalog=json.loads((ROOT/'configs/matched_v1/catalog.json').read_text());changes=[]
    for item in catalog:
        if item['model'] not in ('qwen3','qwen35') or item['language']!='Tsez':continue
        path=ROOT/item['config'];cfg=json.loads(path.read_text());verify_inputs(cfg)
        if len(cfg['test'])==445:continue
        rows=read_records(ROOT/cfg['results'])
        if any(r['origin']['type']=='generated_first_attempt' for r in rows):raise ValueError('A new run already started; do not overwrite')
        before=cfg['fingerprint'];cfg['test']=dataset('Tsez')
        if cfg['method']=='modelgloss':cfg['predicted_glosses']=load_glosslm_predictions('Tsez',ROOT/'runners/openrouter/cache')[:445]
        cfg.pop('fingerprint');cfg['fingerprint']=digest(cfg)
        atomic_json(path,cfg)
        imported=reuse(cfg);write_rows(ROOT/cfg['results'],imported)
        item['expected']=445;item['records']=len(imported)
        changes.append(dict(id=cfg['id'],old_fingerprint=before,new_fingerprint=cfg['fingerprint'],reason='User clarified: only API Tsez is 99; local Qwen uses full 445'))
    atomic_json(ROOT/'configs/matched_v1/catalog.json',catalog)
    if changes:atomic_json(ROOT/'docs/matched_v1/tsez_scope_correction.json',changes)
    jobs=json.loads((ROOT/'configs/matched_v1/jobs.json').read_text())
    jobs=[j for j in jobs if not (j['model'] in ('qwen3','qwen35') and j['language']=='tsez')]
    for model in ('qwen3','qwen35'):
        for method in ('shot','chain_gloss','modelgloss'):
            paths=json.loads((ROOT/f'configs/matched_v1/groups/{model}_tsez_{method}.json').read_text())
            for i in range(0,len(paths),3):
                name=f'{model}_tsez_{method}_batch{i//3+1}'
                group=f'configs/matched_v1/groups/{name}.json';atomic_json(ROOT/group,paths[i:i+3])
                job=f'scripts/jobs/matched_v1/{name}.sh'
                walltime, deadline = limits(paths[i:i+3])
                setup='source scripts/shared/qwen35_job_env.sh' if model=='qwen35' else 'source scripts/shared/cache_env.sh\nsource venv/bin/activate\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
                (ROOT/job).write_text(f'''#!/bin/bash
#SBATCH --job-name=matched_{name}
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lrz-hgx-h100-015,lrz-hgx-h100-026
#SBATCH --mem=80G
#SBATCH --time={walltime}
#SBATCH --output={ROOT}/slurm_outputs/%j.out
set -eo pipefail
cd "{ROOT}"
{setup}
export PYTHONUNBUFFERED=1
python runners/matched/run.py --group {group} --deadline-seconds {deadline}
''')
                jobs.append(dict(name=name,model=model,language='tsez',method=method,job=job,group=group))
    atomic_json(ROOT/'configs/matched_v1/jobs.json',jobs)
    print('Corrected',len(changes),'Qwen Tsez configs. API configs unchanged.')


if __name__=='__main__':main()
