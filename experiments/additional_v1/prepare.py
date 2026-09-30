"""Freeze contrasts and generate single-H100 jobs; never submit implicitly."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from common import *

SEEDS = (20260910, 20260925, 20261001)


def main():
    if (BASE / 'configs/lock.json').exists():
        raise RuntimeError('Already frozen; use a new study version rather than overwrite')
    items = catalog()
    contrasts = []
    for model in MODELS:
        for language in LANGUAGES:
            cohorts = ('valid84', 'sensitivity83') if language == 'Lezgi' else ('native', 'common99') if language == 'Tsez' else ('native',)
            for cohort in cohorts:
                for left, right in (('modelgloss', 'shot'), ('chain_gloss', 'shot'), ('modelgloss', 'chain_gloss')):
                    contrasts.append(dict(package='A', family='A_methods', model=model, language=language,
                        cohort=cohort, left=baseline(model, language, left)['config'],
                        right=baseline(model, language, right)['config']))
    for item in items:
        if item['language'] != 'Lezgi' or item['material'] == 'baseline':
            continue
        for cohort in ('valid84', 'sensitivity83'):
            contrasts.append(dict(package='B', family='B_context', model=item['model'], language='Lezgi', cohort=cohort,
                left=item['config'], right=baseline(item['model'], 'Lezgi', item['method'])['config']))
            if item['material'] in ('cheatsheet_jpg', 'summary_tables_jpg'):
                txt = next(x for x in items if all(x[k] == item[k] for k in ('model', 'language', 'source', 'variant', 'method'))
                           and x['material'] == item['material'].replace('_jpg', '_txt'))
                contrasts.append(dict(package='B', family='B_format', model=item['model'], language='Lezgi',
                    cohort=cohort, left=item['config'], right=txt['config']))
    for i, row in enumerate(contrasts):
        row['id'] = f"{row['family']}_{i:03d}"
    revisions = {}
    cache = Path('/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/cache/huggingface/hub')
    for model in MODELS:
        cfg = config(baseline(model, 'Gitksan', 'shot'))
        directory = cache / ('models--' + cfg['model_id'].replace('/', '--'))
        revision = (directory / 'refs/main').read_text().strip()
        snapshot = directory / 'snapshots' / revision
        if not snapshot.is_dir():
            raise ValueError('Pinned offline model unavailable')
        revisions[model] = dict(model_id=cfg['model_id'], revision=revision, snapshot=str(snapshot))
    generation, groups, identities = [], {}, {}

    def add(group, parent, seed, budget=512, oracle=False):
        identity = dict(parent=parent['config'], parent_sha256=sha256(ROOT / parent['config']),
                        model=parent['model'], language=parent['language'], seed=seed,
                        budget=budget, oracle=oracle, runtime=revisions[parent['model']])
        key = digest(identity)
        if key not in identities:
            name = f"{parent['id']}_s{seed}_t{budget}" + ('_oracle' if oracle else '')
            entry = dict(identity, id=name, fingerprint=key, owner_group=group)
            generation.append(entry); identities[key] = entry
            groups.setdefault(group, []).append(name)
        return identities[key]['id']

    comparisons = []
    for model in MODELS:
        for lang in LANGUAGES:
            parent = baseline(model, lang, 'modelgloss')
            group = f'G_{model}_{lang.lower()}'
            control = add(group, parent, SEEDS[0])
            oracle = add(group, parent, SEEDS[0], oracle=True)
            comparisons.append(dict(package='G', model=model, language=lang, left=oracle, right=control))
            group = f'F_{model}_{lang.lower()}'
            parent = baseline(model, lang, 'chain_gloss')
            control = add(group, parent, SEEDS[0])
            high = add(group, parent, SEEDS[0], budget=1024)
            comparisons.append(dict(package='F', model=model, language=lang, left=high, right=control))
    choices = {'Gitksan': ('shot', 'cheatsheet_txt', 'pdf1_brown'),
               'Lezgi': ('chain_gloss', 'cheatsheet_txt', 'grammar'),
               'Natugu': ('modelgloss', 'cheatsheet_jpg', 'grammar'),
               'Tsez': ('chain_gloss', 'summary_text_txt', 'grammar')}
    for model in MODELS:
        for lang, (method, material, source) in choices.items():
            parent = next(x for x in items if (x['model'], x['language'], x['method'], x['material'], x['source'], x['variant']) ==
                          (model, lang, method, material, source, 'original'))
            for seed in SEEDS:
                group = f'I_{model}_{lang.lower()}_{seed}'
                control = add(group, baseline(model, lang, method), seed)
                context = add(group, parent, seed)
                comparisons.append(dict(package='I', model=model, language=lang, seed=seed, left=context, right=control))
    plan = dict(version='additional_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        post_hoc=True, models=list(MODELS), excluded_models=['gemini25flashlite', 'gpt56luna'],
        contrasts=contrasts, generation=generation, groups=groups, generation_comparisons=comparisons,
        metrics=['chrF++', 'BLEU'], primary_metric='chrF++', samples=100000, seed=20260925,
        confidence=.95, resampling='source-text cluster bootstrap; all rows within a sampled source cluster retained',
        test='two-sided null-centered paired bootstrap with plus-one p-values',
        holm='A_methods, B_context, B_format: each pools both metrics, both Qwen models and all listed cohorts/masks',
        ci='pointwise percentile intervals, not multiplicity adjusted',
        no_duplicate_cohorts='common99 is separately tested only for Tsez; other-language shared views alias native or Lezgi masks',
        output_budget_scope='Gemini pilot excluded by user; preselect all eight Qwen/language no-book pairs at 512 and 1024, no result-dependent expansion',
        generation_inference='F/G: separate exploratory Holm families per package across both metrics and masks/cohorts; I: descriptive across-seed stability',
        seeds=list(SEEDS), human_gates={'D':'formal gloss scoring skipped: no extraction validator; automatic coverage only',
        'E':'human faithfulness review skipped: no qualified evaluator; inventory only',
        'H':'skipped: no validated matched and irrelevant-control materials',
        'J':'skipped: no manually verified transcription targets',
        'K':'skipped by user: no qualified evaluator',
        'L':'runtime comparability and a verified Natugu contamination mask required before any MTOB baseline'},
        pending_scopes={'H':'Qwen3/Qwen3.5 x Gitksan Brown/Natugu x shot, five conditions each',
                        'J':'two sampled regions per source/material, both Qwen models',
                        'K':'40 Natugu items, three no-book methods for both models, 240 judgments per evaluator',
                        'L':'eight no-book Qwen/language baselines; contexts require matched reruns if compatibility cannot be certified'})
    save('configs/plan.json', plan)
    for group, ids in groups.items(): save(f'configs/groups/{group}.json', ids)
    for folder in ('logs', 'results', 'metrics', 'reviews', 'reports', 'cache', 'jobs'):
        (BASE / folder).mkdir(exist_ok=True)
    jobs = {}

    def job(name, command, model=None, hours=5):
        gpu = model is not None
        resources = ('#SBATCH --partition=lrz-hgx-h100-94x4\n#SBATCH --gres=gpu:1\n'
                     '#SBATCH --exclude=lrz-hgx-h100-015,lrz-hgx-h100-026\n#SBATCH --mem=80G\n') if gpu else '#SBATCH --partition=lrz-cpu\n#SBATCH --qos=cpu\n#SBATCH --mem=24G\n'
        env = 'source scripts/shared/qwen35_job_env.sh' if model == 'qwen35' else 'source scripts/shared/cache_env.sh\nsource venv/bin/activate'
        text = (f'#!/bin/bash\n#SBATCH --job-name=addv1_{name}\n{resources}#SBATCH --cpus-per-task=1\n'
                f'#SBATCH --time={hours:02}:00:00\n#SBATCH --output={BASE}/logs/%j.out\nset -eo pipefail\n'
                f'cd "{ROOT}"\n{env}\nexport PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
                'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
                'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n'
                f'python -u "{BASE / command.split()[0]}" ' + ' '.join(command.split()[1:]) + '\n')
        path = destination(f'jobs/{name}.sh'); path.write_text(text)
        subprocess.run(['bash', '-n', str(path)], check=True)
        jobs[name] = dict(path=str(path.relative_to(ROOT)), gpu_count=int(gpu), hours=hours)
    job('audit', 'audit.py', hours=2)
    job('analysis_A', 'analyze.py --package A', hours=5)
    job('analysis_B', 'analyze.py --package B', hours=5)
    for model in MODELS:
        # Processor loading uses a model-specific environment, but no GPU resources.
        job(f'inputs_{model}', f'input_sizes.py --model {model}', hours=3)
        if model == 'qwen35':
            path = destination(f'jobs/inputs_{model}.sh')
            path.write_text(path.read_text().replace('source scripts/shared/cache_env.sh\nsource venv/bin/activate', 'source scripts/shared/qwen35_job_env.sh'))
    for group in groups:
        model = next(m for m in MODELS if group.startswith(('F_'+m, 'G_'+m, 'I_'+m)))
        lang = next(lang for lang in LANGUAGES if '_'+lang.lower() in group)
        hours = 10 if lang == 'Tsez' else 4 if lang == 'Gitksan' else 6
        job(group, f'generate.py --group {group} --deadline {hours*3600-300}', model, hours)
    job('report', 'summarize.py', hours=5)
    save('configs/jobs.json', jobs)
    code = {str(p.relative_to(BASE)): sha256(p) for p in BASE.glob('*.py')}
    save('configs/lock.json', dict(plan_sha256=sha256(BASE/'configs/plan.json'), code=code,
        dependencies={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT/'runners').rglob('*.py')},
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print(f'Frozen: {len(contrasts)*2} A/B metric tests, {len(generation)} generation conditions, {len(groups)} GPU jobs; no API')


if __name__ == '__main__': main()
