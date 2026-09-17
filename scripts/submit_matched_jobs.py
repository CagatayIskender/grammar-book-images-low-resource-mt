"""Queue-aware submission, respecting available user slots and complete/imported outputs."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runners/matched'))
from protocol import validate_rows,verify_inputs
from experiment_io import atomic_json,read_records


def main(args):
    if 'gpt56luna' in args.models and args.submit and not args.allow_luna:
        raise ValueError('Luna submission requires --allow-luna and an explicit MATCHED_API_BUDGET_USD')
    import os
    if 'gpt56luna' in args.models and args.submit and float(os.environ.get('MATCHED_API_BUDGET_USD','0'))<=0:
        raise ValueError('Luna budget missing')
    receipt=ROOT/'docs/matched_v1/submissions.json';receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=json.loads((ROOT/'configs/matched_v1/jobs.json').read_text())
        queue=subprocess.check_output(['squeue','-u',os.environ.get('USER','ge92kun2'),'-h','-o','%i|%j'],text=True).splitlines() if args.submit else []
        names={x.split('|',1)[1] for x in queue};slots=max(0,48-len(queue))
        # Serialize API workers per model: shared-file locks on compute nodes
        # did not prevent collisions in metadata/budget writes on DSS.
        api_tails={model:[line.split('|',1)[0] for line in queue
                          if line.split('|',1)[1].startswith('matched_'+model+'_')]
                   for model in ('gemini25flashlite','gpt56luna')}
        saved=json.loads(receipt.read_text()) if receipt.exists() else []
        submitted=[];remaining=[]
        for job in jobs:
            if job['model'] not in args.models:continue
            missing=[]
            for name in json.loads((ROOT/job['group']).read_text()):
                cfg=json.loads((ROOT/name).read_text());verify_inputs(cfg)
                try:validate_rows(read_records(ROOT/cfg['results']),cfg,complete=True)
                except ValueError:missing.append(cfg['id'])
            if not missing:continue
            if 'matched_'+job['name'] in names:continue
            print('[PLAN]',job['job'],len(missing),'incomplete conditions',flush=True)
            if not args.submit or slots<=0:
                remaining.append(job);continue
            dependencies=api_tails.get(job['model'],[])
            command=['sbatch','--parsable']
            if dependencies:command.append('--dependency=afterany:'+':'.join(dependencies))
            result=subprocess.run(command+[job['job']],cwd=ROOT,text=True,capture_output=True)
            if result.returncode:
                print(result.stderr,flush=True);remaining.append(job);slots=0;continue
            jid=result.stdout.strip().split(';')[0]
            saved.append(dict(job_id=jid,dependency=dependencies,**job));submitted.append(jid);slots-=1
            if job['model'] in api_tails:api_tails[job['model']]=[jid]
            atomic_json(receipt,saved);print('[SUBMITTED]',jid,flush=True)
        atomic_json(ROOT/'docs/matched_v1/remaining_jobs.json',remaining)
        print('[SUMMARY] submitted=',submitted,'remaining=',len(remaining),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['qwen3','qwen35','gemini25flashlite'])
    p.add_argument('--submit',action='store_true');p.add_argument('--allow-luna',action='store_true');main(p.parse_args())
