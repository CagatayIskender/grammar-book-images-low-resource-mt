"""Plan or replace pending Qwen jobs, preserving evaluation dependencies."""
import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from matched_walltime import limits

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = {'matched_xcomet', 'matched_significance', 'matched_significance_common99'}


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def queue():
    return [line.split('|') for line in command('squeue', '-u', os.environ['USER'],
            '-h', '-o', '%i|%j|%T').splitlines()]


def main(submit=False):
    jobs = {'matched_' + j['name']: j for j in json.loads((ROOT / 'configs/matched_v1/jobs.json').read_text())}
    initial = queue()
    plan = []
    for jid, name, state in initial:
        if name not in jobs or jobs[name]['model'] not in ('qwen3', 'qwen35'):
            continue
        job = jobs[name]
        text = (ROOT / job['job']).read_text()
        walltime, deadline = limits(json.loads((ROOT / job['group']).read_text()))
        old_time = re.search(r'^#SBATCH --time=(.*)$', text, re.M).group(1)
        def seconds(value):
            h, m, s = map(int, value.split(':'))
            return 3600*h + 60*m + s
        if seconds(walltime) >= seconds(old_time) or state != 'PENDING':
            continue
        assert '#SBATCH --gres=gpu:1' in text
        plan.append(dict(old_job_id=jid, name=name, script=job['job'],
                         old_time=old_time, new_time=walltime, deadline_seconds=deadline))
    print(json.dumps(plan, indent=2), flush=True)
    if not submit or not plan:
        return
    # Hold dependents before cancelling parents: afterany treats cancellation as completion.
    dependents = [jid for jid, name, state in initial if name in EVALUATION]
    if any(state != 'PENDING' for jid, name, state in initial if name in EVALUATION):
        raise RuntimeError('Evaluation already started; refusing replacement')
    receipt = ROOT / 'docs/matched_v1/walltime_resubmissions.json'
    if receipt.exists():
        raise RuntimeError('Receipt exists; inspect it before attempting another replacement')
    journal = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), plan=plan,
                   evaluation_job_ids=dependents, status='started')
    def save():
        receipt.write_text(json.dumps(journal, indent=2) + '\n')
    save()
    try:
        for jid in dependents:
            command('scontrol', 'hold', jid)
        for item in plan:
            current = {jid: state for jid, name, state in queue()}
            if current.get(item['old_job_id']) != 'PENDING':
                raise RuntimeError('Generation state changed; refusing to cancel running work')
            command('scontrol', 'hold', item['old_job_id'])
            path = ROOT / item['script']
            text = path.read_text()
            text = re.sub(r'(?m)^#SBATCH --time=.*$', '#SBATCH --time=' + item['new_time'], text)
            text, count = re.subn(r'--deadline-seconds \d+', '--deadline-seconds ' + str(item['deadline_seconds']), text)
            assert count == 1
            path.write_text(text)
            command('bash', '-n', str(path))
            command('scancel', item['old_job_id'])
            item['cancelled'] = True
            save()
            item['new_job_id'] = command('sbatch', '--parsable', '--hold', str(path)).split(';')[0]
            save()
        parents = [jid for jid, name, state in queue()
                   if name.startswith('matched_') and name not in EVALUATION]
        journal['new_dependencies'] = parents
        save()
        for jid in dependents:
            command('scontrol', 'update', 'JobId=' + jid, 'Dependency=afterany:' + ':'.join(parents))
        for item in plan:
            command('scontrol', 'release', item['new_job_id'])
        for jid in dependents:
            command('scontrol', 'release', jid)
        journal['status'] = 'complete'
        save()
    except Exception as exc:
        journal['status'] = 'needs_recovery'
        journal['error'] = str(exc)
        save()
        raise RuntimeError('Inspect receipt; evaluation remains held until replacement is complete') from exc
    print('Replaced', len(plan), 'jobs; evaluation dependencies updated.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    main(parser.parse_args().submit)
