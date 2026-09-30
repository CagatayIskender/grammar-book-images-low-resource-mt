"""Add scorer jobs without modifying the running, frozen generation suite."""
from pathlib import Path
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *


def main():
    verify_suite()
    if (BASE/'configs/xcomet_extension.json').exists():
        raise ValueError('Scoring extension already prepared')
    cache=Path('/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/cache/huggingface/hub')
    revisions={'xl':'6a123c5e8e6dccab25e5fcffa3c8b417abadb462','xxl':'873bac1b1c461e410c4a6e379f6790d3d1c7c214'}
    models={};jobs={}
    for size,rev in revisions.items():
        checkpoint=cache/f'models--Unbabel--XCOMET-{size.upper()}'/'snapshots'/rev/'checkpoints/model.ckpt'
        if not checkpoint.is_file():raise ValueError('Cached checkpoint missing')
        models[size]=dict(model=f'Unbabel/XCOMET-{size.upper()}',revision=rev,checkpoint=str(checkpoint))
        script=destination(f'jobs/score_xcomet_{size}.sh')
        script.write_text(f'''#!/bin/bash
#SBATCH --job-name=addv1_xcomet_{size}
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lrz-hgx-h100-015,lrz-hgx-h100-026
#SBATCH --mem={'144G' if size=='xxl' else '80G'}
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --output={BASE}/logs/%j.out
set -eo pipefail
cd "{ROOT}"
source scripts/shared/cache_env.sh
source venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u "{BASE}/scoring/score_xcomet.py" --size {size}
''')
        subprocess.run(['bash','-n',str(script)],check=True)
        jobs[size]=dict(path=str(script.relative_to(ROOT)),hours=5,gpu_count=1,script_sha256=sha256(script))
    save('configs/xcomet_extension.json',dict(models=models,jobs=jobs,plan_sha256=sha256(BASE/'configs/plan.json'),
        scorer_sha256=sha256(BASE/'scoring/score_xcomet.py'),dependency_policy='afterany all generation jobs; score only complete validated conditions',
        original_suite_unchanged=True,significance='No XCOMET significance tests in this extension'))


if __name__=='__main__':main()
