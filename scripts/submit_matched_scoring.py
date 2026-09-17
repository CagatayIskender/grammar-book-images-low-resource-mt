"""Submit evaluation after currently queued matched generation jobs, without duplicates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runners'))
from experiment_io import atomic_json


def main(submit=False):
    queue=subprocess.check_output(['squeue','-u',os.environ.get('USER','ge92kun2'),'-h','-o','%i|%j'],text=True).splitlines()
    active={name:jid for jid,name in (line.split('|',1) for line in queue)}
    jobs=[('matched_xcomet','scripts/scoring/run_matched_xcomet.sh'),
          ('matched_significance','scripts/scoring/run_matched_significance.sh'),
          ('matched_significance_common99','scripts/scoring/run_matched_significance_common99.sh')]
    generation=[jid for name,jid in active.items() if name.startswith('matched_') and name not in {name for name,_ in jobs}]
    path=ROOT/'docs/matched_v1/scoring_submissions.json'
    receipt=json.loads(path.read_text()) if path.exists() else []
    for name,script in jobs:
        if name in active:
            print('[SKIP queued]',name,active[name]);continue
        command=['sbatch','--parsable']
        if generation:command+=['--dependency=afterany:'+':'.join(generation)]
        command+=[script]
        print('[PLAN]',command,flush=True)
        if not submit:continue
        completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
        jid=completed.stdout.strip().split(';',1)[0]
        receipt.append(dict(job_id=jid,name=name,script=script,depends_on=generation))
        atomic_json(path,receipt);print('[SUBMITTED]',jid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true')
    main(parser.parse_args().submit)
