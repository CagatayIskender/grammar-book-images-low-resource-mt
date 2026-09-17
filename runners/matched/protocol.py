"""Frozen within-model grammar ablation protocol; no response-conditioned prompts."""
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runners'))
sys.path.insert(0, str(ROOT / 'runners/openrouter'))
from experiment_io import build_chain_prompt, sha256
from run_grammamt_openrouter_context import build_prompt as api_prompt

REVISION = 'matched_v1'
METHODS = ('shot', 'chain_gloss', 'modelgloss')
MODELS = {'qwen3':'Qwen/Qwen3-VL-8B-Instruct', 'qwen35':'Qwen/Qwen3.5-9B',
          'gemini25flashlite':'google/gemini-2.5-flash-lite', 'gpt56luna':'openai/gpt-5.6-luna'}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def policy(model):
    if model in ('qwen3', 'qwen35'):
        return dict(name='development_validated_sampled_v1', do_sample=True, temperature=0.7,
                    top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
                    max_new_tokens=512, seed=20260910, seed_rule='base + 2 * test_index',
                    dtype='float32', gpu_count=1, enable_thinking=False, semantic_attempts=1)
    return dict(name='api_fixed_v1', temperature=None if model=='gpt56luna' else 0.0,
                max_new_tokens=512, seed=42, reasoning_effort='none', semantic_attempts=1)


def prompt(cfg, source):
    text = cfg.get('grammar_text', '')
    method, lang = cfg['method'], cfg['language']
    if method == 'chain_gloss':
        system, user = build_chain_prompt(cfg['support'], lang, source, text, cfg['context_kind']=='image')
        if cfg['context_kind']=='none':
            user = user.replace(f'Here is a grammar reference summary for {lang}:\n\n', '', 1)
            user = user.replace('Use the grammar reference to produce the gloss first', 'Produce the gloss first', 1)
        return system, user
    gloss = cfg.get('current_gloss') if method=='modelgloss' else None
    return api_prompt([SimpleNamespace(**r) for r in cfg['support']], lang, source,
                      cfg['context_kind'], text, gloss)


def parse(raw, method, truncated=False, reasoning=False):
    """Extract only produced text. A malformed gloss does not erase a final translation."""
    flags, translation, gloss = [], '', ''
    if reasoning or re.search(r'<\s*/?think\b|^\s*(?:analysis|reasoning|thinking)\s*:', raw, re.I|re.M):
        return dict(prediction='', generated_gloss='', error='reasoning_violation', flags=['reasoning_violation'])
    matches = list(re.finditer(r'^\s*FINAL_TRANSLATION:\s*([^\n]+)', raw, re.M|re.I))
    if matches:
        translation = matches[-1][1].strip()
        before = raw[:matches[-1].start()]
        g = re.search(r'^\s*Gloss:\s*(.*)', before, re.M|re.S|re.I)
        if g:
            gloss = g[1].strip()
    elif method != 'chain_gloss':
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines)==1 and not lines[0].lower().startswith(('gloss:', 'note:', 'sorry', 'i cannot', "i can't")):
            translation = re.sub(r'^(?:English translation|Translation|Answer):\s*', '', lines[0], flags=re.I).strip()
            flags.append('unmarked_single_line')
    if method=='chain_gloss' and not gloss:
        flags.append('missing_gloss')
    if '\n' in gloss:
        flags.append('multiline_gloss')
    if truncated:
        flags.append('truncated')
        translation = ''
    if not translation:
        flags.append('empty_or_unextractable_translation')
    return dict(prediction=translation, generated_gloss=gloss,
                error=';'.join(x for x in flags if x not in ('multiline_gloss','unmarked_single_line','missing_gloss')) or None,
                flags=flags)


def validate_rows(rows, cfg, complete=False):
    seen = set()
    for row in rows:
        i = row['idx']
        if not isinstance(i, int) or i < 0 or i >= len(cfg['test']) or i in seen:
            raise ValueError('Invalid/duplicate sentence index')
        seen.add(i)
        ex = cfg['test'][i]
        if row['source']!=ex['source'] or row['reference']!=ex['reference'] or row.get('fingerprint')!=cfg['fingerprint']:
            raise ValueError('Result source/reference/protocol mismatch')
    if complete and seen!=set(range(len(cfg['test']))):
        raise ValueError('Incomplete experiment')


def verify_inputs(cfg):
    for name, checksum in {**cfg['code_hashes'], **cfg['context_hashes']}.items():
        if sha256(ROOT / name)!=checksum:
            raise ValueError(f'Frozen input changed: {name}')
    core = {k:v for k,v in cfg.items() if k!='fingerprint'}
    if digest(core)!=cfg['fingerprint']:
        raise ValueError('Config fingerprint mismatch')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    import os
    with temp.open('w', encoding='utf-8') as handle:
        for row in sorted(rows, key=lambda x:x['idx']):
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def result_row(cfg, index, attempt, origin):
    ex = cfg['test'][index]
    reasoning = attempt.get('reasoning_violation',False) or bool(re.search(r'<\s*/?think\b', attempt.get('raw_with_special_tokens',''), re.I))
    block = parse(attempt.get('raw',''), cfg['method'], attempt.get('truncated',False), reasoning)
    if attempt.get('repetition_warning'):
        block['flags'].append('repetition_warning')
    block['raw'] = attempt.get('raw','')
    block['attempt'] = attempt
    return dict(idx=index, source=ex['source'], reference=ex['reference'], fingerprint=cfg['fingerprint'],
                origin=origin, translation=block)
