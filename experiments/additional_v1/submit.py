"""Submit only this frozen offline suite, recording every accepted Slurm ID."""
import argparse
import fcntl
import subprocess
from common import *


def main(execute):
    verify_suite();jobs=load(BASE/'configs/jobs.json')
    if not execute:
        for name,job in jobs.items():print(name,job['gpu_count'],'GPU',job['hours'],'hours')
        return
    with destination('reports/submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt_path=BASE/'reports/submitted_jobs.json'
        receipt=load(receipt_path) if receipt_path.exists() else {}
        # Reinvocation completes a partial submission, never duplicates accepted jobs.
        for name,job in jobs.items():
            if name in receipt:continue
            dependency=None
            if name=='report':dependency='afterany:'+':'.join(str(v['job_id']) for v in receipt.values())
            elif name!='audit':dependency='afterok:'+str(receipt['audit']['job_id'])
            args=['sbatch','--parsable']
            if dependency:args+=['--dependency='+dependency]
            args+=[str(ROOT/job['path'])]
            result=subprocess.run(args,capture_output=True,text=True)
            if result.returncode:
                save('reports/submission_remaining.json',[n for n in jobs if n not in receipt])
                raise RuntimeError(result.stderr.strip())
            ident=result.stdout.strip().split(';')[0]
            if not ident.isdigit():raise ValueError('Unexpected sbatch response: '+result.stdout)
            receipt[name]=dict(job,job_id=int(ident),dependency=dependency,script_sha256=sha256(ROOT/job['path']))
            save(receipt_path,receipt)
            print(name,ident,flush=True)
        save('reports/submission_remaining.json',[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');main(p.parse_args().submit)
