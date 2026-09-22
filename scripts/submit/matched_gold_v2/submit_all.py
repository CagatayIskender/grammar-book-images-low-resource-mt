"""Submit the full corrected matrix and separate scoring; no smoke jobs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'runners/matched_gold_v2'))
from protocol import verify_inputs
from prepare import historical_hashes
from experiment_io import atomic_json


def main(submit, retry_api=False):
    docs = ROOT / 'docs/matched_gold_v2'
    if historical_hashes() != json.loads((docs / 'historical_hashes.json').read_text()):
        raise ValueError('Historical artifacts changed')
    catalog = json.loads((ROOT / 'configs/matched_gold_v2/catalog.json').read_text())
    for item in catalog:
        verify_inputs(json.loads((ROOT / item['config']).read_text()))
    jobs = json.loads((docs / 'generation_jobs.json').read_text())
    for name in jobs:
        text = (ROOT / name).read_text()
        if not ('gemini25flashlite' in name) and '#SBATCH --gres=gpu:1\n' not in text:
            raise ValueError('Single-GPU job requirement violated')
        subprocess.run(['bash', '-n', str(ROOT / name)], check=True)
    print(f'{len(catalog)} conditions; {len(jobs)} full generation jobs; separate XCOMET and CPU statistics jobs.', flush=True)
    if not submit:
        return
    receipt = docs / 'submissions.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'generation': {}, 'scoring': None, 'analysis': None}
    if retry_api:
        prior = {k: v for k, v in state['generation'].items() if 'gemini25flashlite' in k}
        if prior:
            active = subprocess.check_output(['squeue', '-h', '-j', ','.join(prior.values()), '-o', '%i'], text=True).strip()
            if active: raise RuntimeError('Stop active API jobs before retrying; saved rows must not have concurrent writers')
            if state['scoring']:
                status = subprocess.check_output(['squeue', '-h', '-j', state['scoring'], '-o', '%T'], text=True).strip()
                if status != 'PENDING': raise RuntimeError('Cannot change dependencies of a non-pending scorer')
            state.setdefault('prior_api_submissions', []).append(prior)
            for key in prior: del state['generation'][key]
            atomic_json(receipt, state)
    queued = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i %j'], text=True)
    queued_names = {line.split(maxsplit=1)[1] for line in queued.splitlines() if len(line.split(maxsplit=1)) == 2}
    previous_api = None
    for job in jobs:
        if job in state['generation']:
            if 'gemini25flashlite' in job: previous_api = state['generation'][job]
            continue
        name = re.search(r'#SBATCH --job-name=(\S+)', (ROOT / job).read_text()).group(1)
        if name in queued_names:
            raise RuntimeError(f'Queued job without receipt: {name}; refusing duplicate')
        try:
            command = ['sbatch', '--parsable']
            if 'gemini25flashlite' in job and previous_api:
                command.append('--dependency=afterany:' + previous_api)
            result = subprocess.check_output(command + [job], cwd=ROOT, text=True).strip().split(';')[0]
        except subprocess.CalledProcessError:
            atomic_json(receipt, state)
            print('Submission stopped; already-submitted jobs preserved. Rerun to submit only missing jobs.', flush=True)
            raise
        state['generation'][job] = result
        if 'gemini25flashlite' in job: previous_api = result
        atomic_json(receipt, state)
        print(f'SUBMITTED {result} {job}', flush=True)
    dependencies = ':'.join(state['generation'].values())
    if state['scoring']:
        subprocess.run(['scontrol', 'update', 'JobId=' + state['scoring'], 'Dependency=afterany:' + dependencies], check=True)
    for key, file, dep in [('scoring', 'scripts/scoring/run_matched_gold_v2_xcomet.sh', 'afterany:' + dependencies),
                           ('analysis', 'scripts/scoring/run_matched_gold_v2_analysis.sh', 'afterok:')]:
        if state[key]: continue
        if key == 'analysis': dep += state['scoring']
        job_id = subprocess.check_output(['sbatch', '--parsable', '--dependency=' + dep, file], cwd=ROOT, text=True).strip().split(';')[0]
        state[key] = job_id
        atomic_json(receipt, state)
        print(f'SUBMITTED {job_id} {key}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true', help='Without this flag, validate and list only')
    parser.add_argument('--retry-api', action='store_true', help='Resume terminal API jobs serially, retaining old job IDs and saved rows')
    args = parser.parse_args()
    if args.retry_api and not args.submit: parser.error('--retry-api requires --submit')
    main(args.submit, args.retry_api)
