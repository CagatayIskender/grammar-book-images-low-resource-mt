"""Submit independent one-GPU scorers, each waiting for generation only."""
import fcntl
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *


def main():
    require_audit()
    cfg=load(BASE/'configs/xcomet_extension.json')
    if sha256(BASE/'scoring/score_xcomet.py')!=cfg['scorer_sha256']:raise ValueError('Scorer changed')
    receipt=load(BASE/'reports/submitted_jobs.json')
    groups=load(BASE/'configs/plan.json')['groups']
    ids=[str(receipt[group]['job_id']) for group in groups]
    dep='afterany:'+':'.join(ids)
    path=BASE/'reports/submitted_xcomet_jobs.json'
    with destination('reports/xcomet_submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sent=load(path) if path.exists() else {}
        for size,job in cfg['jobs'].items():
            if size in sent:continue
            if sha256(ROOT/job['path'])!=job['script_sha256']:raise ValueError('Job changed')
            answer=subprocess.run(['sbatch','--parsable','--dependency='+dep,str(ROOT/job['path'])],capture_output=True,text=True,check=True)
            ident=answer.stdout.strip().split(';')[0]
            if not ident.isdigit():raise ValueError('Invalid sbatch response')
            sent[size]=dict(job,job_id=int(ident),dependency=dep)
            save(path,sent);print(size,ident,flush=True)


if __name__=='__main__':main()
