"""Create CPU/GPU jobs; submission is separate and opt-in."""
import json
from pathlib import Path
import sys
from matched_walltime import limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runners'))
from experiment_io import atomic_json


def main():
    groups=ROOT/'configs/matched_v1/groups'
    index=[]
    for group in sorted(groups.glob('*.json')):
        if '_batch' in group.stem:continue
        configs=json.loads(group.read_text())
        first=json.loads((ROOT/configs[0]).read_text())
        model,lang=first['model'],first['language'].lower()
        if model=='gpt56luna':continue
        full_tsez = model in ('qwen3','qwen35') and lang=='tsez' and len(first['test'])>99
        chunks=[configs[i:i+3] for i in range(0,len(configs),3)] if full_tsez else [configs]
        if full_tsez:
            (ROOT/'scripts/jobs/matched_v1'/(group.stem+'.sh')).unlink(missing_ok=True)
        for i,chunk in enumerate(chunks):
            name=group.stem+(f'_batch{i+1}' if len(chunks)>1 else '')
            group_path=groups/(name+'.json');atomic_json(group_path,chunk)
            cpu=model not in ('qwen3','qwen35')
            walltime, deadline = limits(chunk)
            resources='#SBATCH --partition=lrz-cpu\n#SBATCH --qos=cpu\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=8G' if cpu else '#SBATCH --partition=lrz-hgx-h100-94x4\n#SBATCH --gres=gpu:1\n#SBATCH --exclude=lrz-hgx-h100-015,lrz-hgx-h100-026\n#SBATCH --mem=80G'
            setup='source scripts/shared/openrouter_job_env.sh' if cpu else 'source scripts/shared/qwen35_job_env.sh' if model=='qwen35' else 'source scripts/shared/cache_env.sh\nsource venv/bin/activate\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
            body=f'''#!/bin/bash
#SBATCH --job-name=matched_{name}
{resources}
#SBATCH --time={walltime}
#SBATCH --output={ROOT}/slurm_outputs/%j.out
set -eo pipefail
cd "{ROOT}"
{setup}
export PYTHONUNBUFFERED=1
python runners/matched/run.py --group "{group_path.relative_to(ROOT)}" --deadline-seconds {deadline}
'''
            path=ROOT/'scripts/jobs/matched_v1'/(name+'.sh');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body)
            index.append(dict(name=name,model=model,language=lang,method=first['method'],job=str(path.relative_to(ROOT)),group=str(group_path.relative_to(ROOT))))
    luna=[x for x in json.loads((ROOT/'configs/matched_v1/catalog.json').read_text()) if x['model']=='gpt56luna']
    luna.sort(key=lambda x:(x['material']!='baseline',x['language']!='Gitksan',x['language'],x['source'],x['variant'],x['material'],x['method']))
    group=ROOT/'configs/matched_v1/groups/gpt56luna_budget.json'
    atomic_json(group,[x['config'] for x in luna])
    job=ROOT/'scripts/jobs/matched_v1/gpt56luna_budget.sh'
    job.write_text(f'''#!/bin/bash
#SBATCH --job-name=matched_gpt56luna_budget
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output={ROOT}/slurm_outputs/%j.out
set -eo pipefail
cd "{ROOT}"
source scripts/shared/openrouter_job_env.sh
export MATCHED_API_BUDGET_USD=4
export PYTHONUNBUFFERED=1
python runners/matched/run.py --group configs/matched_v1/groups/gpt56luna_budget.json --deadline-seconds 34200
''')
    index.append(dict(name='gpt56luna_budget',model='gpt56luna',language='all',method='all',job=str(job.relative_to(ROOT)),group=str(group.relative_to(ROOT))))
    atomic_json(ROOT/'configs/matched_v1/jobs.json',index)
    for model in sorted({x['model'] for x in index}):
        path=ROOT/'scripts/submit/matched_v1'/f'submit_{model}.sh';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(f'#!/bin/bash\nset -eo pipefail\ncd "{ROOT}"\nvenv/bin/python scripts/submit_matched_jobs.py --models {model} --submit "$@"\n')
    print('Prepared',len(index),'job groups. No jobs submitted.')


if __name__=='__main__':main()
