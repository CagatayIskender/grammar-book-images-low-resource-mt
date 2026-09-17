"""Read-only experiment audit; write evidence only under docs/comparability_audit."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runners'))
from experiment_io import dataset, sha256, validate_records
import sacrebleu


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    output = ROOT / 'docs/comparability_audit'
    output.mkdir(parents=True, exist_ok=True)
    audit, fingerprints, provenance_map, metric_mismatches = [], {}, {}, []
    models = ('qwen3', 'qwen35', 'gemini25flashlite')
    for family in ('baseline', 'legacy', 'curated_v1', 'chain_gloss_v2'):
        for model in models:
            for path in sorted((ROOT / 'results' / family / model).rglob('*.jsonl')):
                if 'preflight' in path.name:
                    continue
                rel = path.relative_to(ROOT)
                lang = rel.parts[3].title()
                expected = dataset(lang)
                if model == 'gemini25flashlite' and lang == 'Tsez':
                    expected = expected[:99]
                initial_hash = sha256(path)
                item = dict(path=str(rel), family=family, model=model, language=lang,
                            deprecated_md=path.stem.endswith('_md'), expected=len(expected))
                try:
                    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                    item['records'] = len(rows)
                    validate_records(rows, expected, complete=True)
                    item['alignment'] = 'pass'
                except (ValueError, KeyError) as exc:
                    item['alignment'] = str(exc)
                    item['input_sha256'] = initial_hash
                    item['stable_during_read'] = sha256(path) == initial_hash
                    audit.append(item)
                    continue
                blocks = [key for key, value in rows[0].items()
                          if isinstance(value, dict) and 'prediction' in value]
                quality = {}
                for key in blocks:
                    quality[key] = dict(empty=sum(not row.get(key, {}).get('prediction', '').strip() for row in rows),
                                        errors=sum(bool(row.get(key, {}).get('error')) for row in rows))
                item['quality'] = quality
                provenance_path = path.with_suffix('.provenance.json')
                if provenance_path.exists():
                    prov = json.loads(provenance_path.read_text())
                    provenance_map[str(rel)] = prov
                    item['provenance_fingerprint'] = all(row.get('fingerprint') == digest(prov) for row in rows)
                    item['support_matches_current'] = prov.get('support') == dataset(lang, 'train')[:21]
                    item['test_matches_current'] = prov.get('test') == dataset(lang)
                    item['context_hash_mismatches'] = [p for p, h in prov.get('context_hashes', {}).items()
                                                       if not (ROOT / p).exists() or sha256(ROOT / p) != h]
                    item['changed_code_since_run'] = [p for p, h in prov.get('code_hashes', {}).items()
                                                      if not (ROOT / p).exists() or sha256(ROOT / p) != h]
                else:
                    item['provenance_fingerprint'] = 'not recorded'
                metric = ROOT / 'metrics' / path.relative_to(ROOT / 'results').with_suffix('.json')
                if not metric.exists():
                    metric = metric.with_name(metric.name.replace('results_', 'metrics_', 1))
                if metric.exists():
                    values = json.loads(metric.read_text())
                    item['metric_path'] = str(metric.relative_to(ROOT))
                    item['scorer'] = values.get('comet_model_used')
                    item['metric_hash_matches'] = values.get('results_sha256') == initial_hash if 'results_sha256' in values else 'not recorded'
                    item['metric_checks'] = {}
                    for key in blocks:
                        hyps = [r[key]['prediction'] for r in rows]
                        refs = [[r['reference'] for r in rows]]
                        scores = {'bleu':sacrebleu.corpus_bleu(hyps, refs).score,
                                  'chrf':sacrebleu.corpus_chrf(hyps, refs, word_order=2).score}
                        for name, score in scores.items():
                            old = values.get(key, {}).get(name)
                            match = isinstance(old, (float, int)) and abs(old-score) < 1e-5
                            item['metric_checks'][key + ':' + name] = match
                            if not match:
                                metric_mismatches.append(dict(path=str(rel), key=key, metric=name, stored=old, recalculated=score))
                    if model == 'gemini25flashlite':
                        item['api_settings'] = {k:values.get(k) for k in ('model', 'temperature', 'seed', 'support_n', 'reasoning_effort', 'test_n')}
                        item['api_providers'] = sorted({str(r[key].get('api', {}).get('provider')) for r in rows for key in blocks})
                        item['positive_reasoning_tokens'] = sum(bool((r[key].get('api', {}).get('usage', {}).get('completion_tokens_details') or {}).get('reasoning_tokens')) for r in rows for key in blocks)
                        payload = {k:values.get(k) for k in ('model', 'support_n', 'test_n', 'temperature', 'seed', 'reasoning_effort')}
                        payload.update(language=lang, grammar_text_file=rows[0].get('grammar_text_file',''),
                                       grammar_image_dir=str(Path(rows[0]['grammar_image_paths'][0]).parent) if rows[0].get('grammar_image_paths') else '',
                                       grammar_image_paths=rows[0].get('grammar_image_paths', []),
                                       model_gloss='modelgloss' in blocks[0], max_tokens=512)
                        if family == 'baseline':
                            payload['baseline'] = True
                        fp = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                        item['api_fingerprint_reconstructed'] = all(r.get('experiment_fingerprint') == fp for r in rows)
                        item['image_count'] = len(rows[0].get('grammar_image_paths', []))
                        fingerprints[str(rel)] = {r['idx']:r[blocks[0]].get('glosslm_pred_gloss') for r in rows}
                else:
                    item['metric_path'] = 'missing'
                item['input_sha256'] = initial_hash
                item['stable_during_read'] = sha256(path) == initial_hash
                audit.append(item)

    pairs = []
    for path, prov in provenance_map.items():
        if not path.startswith('results/chain_gloss_v2/qwen35/') or not path.endswith('_sampled_v1.jsonl'):
            continue
        lang = prov['config']['language'].lower()
        base = f'results/baseline/qwen35/{lang}/grammar/original/chain_gloss_sampled_v1.jsonl'
        bp = provenance_map.get(base)
        if bp is None:
            pairs.append(dict(context=path, baseline=base, status='no complete baseline'))
            continue
        common = set(prov['code_hashes']) & set(bp['code_hashes'])
        mismatch = [k for k in common if prov['code_hashes'][k] != bp['code_hashes'][k]]
        checks = {k:prov.get(k)==bp.get(k) for k in ('support','test','dtype','gpu_count','attention','enable_thinking','seed','max_attempts')}
        checks['shared_code'] = not mismatch
        checks['sampling'] = all(prov['policy'].get(k)==v for k,v in bp['sampling'].items())
        checks['max_new_tokens'] = prov['policy']['max_new_tokens']==bp['max_new_tokens']
        pairs.append(dict(context=path, baseline=base, checks=checks, differing_code=mismatch))
    api_pairs = []
    for path, gloss in fingerprints.items():
        if not path.startswith('results/curated_v1/'):
            continue
        lang = Path(path).parts[3]
        mode = 'modelgloss_baseline' if 'modelgloss' in Path(path).name else 'baseline'
        base = f'results/baseline/gemini25flashlite/{lang}/grammar/original/results_gemini25flashlite_{lang}_{mode}.jsonl'
        api_pairs.append(dict(context=path, baseline=base, gloss_values_match=gloss==fingerprints.get(base)))
    summary = collections.Counter()
    for row in audit:
        if row['deprecated_md']:
            continue
        status = 'aligned_complete' if row['alignment']=='pass' else 'incomplete_or_misaligned'
        summary[(row['family'],row['model'],status)] += 1
    report = dict(scope='Existing Qwen3, Qwen3.5 and Gemini baselines, legacy, curated_v1, chain_gloss_v2; excludes smoke, MTOB, preflight. No generation or scoring outputs edited.',
                  sacrebleu_version=sacrebleu.__version__, files=audit, metric_mismatches=metric_mismatches,
                  qwen35_sampled_pairs=pairs, gemini_pairs=api_pairs,
                  summary=[dict(family=k[0],model=k[1],status=k[2],count=v) for k,v in summary.items()])
    (output/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','gemini_pairs','qwen35_sampled_pairs','metric_mismatches')}, indent=2))
    print('Metric discrepancies:',len(metric_mismatches))
    print('Qwen35 complete matched pairs:',sum(all(p.get('checks',{'missing':False}).values()) for p in pairs),'/',len(pairs))
    print('Gemini gloss pairs:',sum(p['gloss_values_match'] for p in api_pairs),'/',len(api_pairs))


if __name__ == '__main__':
    main()
