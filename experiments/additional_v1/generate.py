"""Isolated Qwen interventions; unchanged prompts except the declared gold oracle."""
import argparse
import fcntl
import gc
import importlib.metadata
import os
import time
from common import *


def active_prompt(parent, entry, index):
    gloss = parent['test'][index]['gloss'] if entry['oracle'] else parent.get('predicted_glosses', [''] * len(parent['test']))[index]
    return original.prompt(dict(parent, current_gloss=gloss), parent['test'][index]['source'])


def validate(rows, parent, entry, complete=False):
    seen = set()
    for row in rows:
        i = row['idx']
        if i in seen or not isinstance(i, int) or not 0 <= i < len(parent['test']):
            raise ValueError('Duplicate or invalid index')
        seen.add(i)
        ex = parent['test'][i]
        if (row['source'], row['reference'], row['fingerprint']) != (ex['source'], ex['reference'], entry['fingerprint']):
            raise ValueError('Checkpoint identity mismatch')
        system, user = active_prompt(parent, entry, i)
        if row['translation']['attempt']['prompt_evidence'] != {'system': system, 'user': user}:
            raise ValueError('Checkpoint prompt mismatch')
    if complete and seen != set(range(len(parent['test']))):
        raise ValueError('Incomplete generation')


def generate(backend, system, user, images, seed, budget):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import LogitsProcessorList
    from fp32_attention import expanded_sdpa_context
    from diagnose_tsez_chain import PresencePenalty, generation_policy, repetition_warning
    torch = backend.torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                {'role': 'user', 'content': [{'type': 'image', 'image': x} for x in images]
                 + [{'type': 'text', 'text': user}]}]
    prompt = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if backend.model_kind == 'qwen35' and not prompt.endswith('<think>\n\n</think>\n\n'):
        raise RuntimeError('Disabled-thinking template missing')
    kwargs = dict(text=[prompt], return_tensors='pt')
    if images: kwargs['images'] = images
    inputs = backend.processor(**kwargs).to('cuda:0')
    length = inputs.input_ids.shape[1]
    with torch.inference_mode(), expanded_sdpa_context(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = backend.model.generate(**inputs, max_new_tokens=budget, **generation_policy(True),
                                    logits_processor=LogitsProcessorList([PresencePenalty(length, 1.5)]))
    ids = out[0, length:]
    eos = backend.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    raw = backend.processor.decode(ids, skip_special_tokens=True).strip()
    truncated = len(ids) >= budget and int(ids[-1]) not in eos
    return dict(raw=raw, raw_with_special_tokens=backend.processor.decode(ids, skip_special_tokens=False),
        truncated=truncated, generated_tokens=len(ids), input_tokens=length, seed=seed,
        finish_reason='length' if truncated else 'eos_or_stopping_criteria',
        repetition_warning=repetition_warning(raw), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_evidence={'system': system, 'user': user})


def run(group, deadline):
    require_audit()
    plan = load(BASE/'configs/plan.json')
    ids = plan['groups'][group]
    entries = {e['id']: e for e in plan['generation']}
    start = time.monotonic(); backend = None
    with destination(f'results/{group}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for ident in ids:
            entry = entries[ident]; parent = load(ROOT/entry['parent'])
            if sha256(ROOT/entry['parent']) != entry['parent_sha256']: raise ValueError('Parent changed')
            original.verify_inputs(parent)
            path = destination(f'results/{ident}.jsonl')
            rows = read_records(path); validate(rows, parent, entry)
            if len(rows) == len(parent['test']): continue
            if backend is None:
                from run_audited_context import Backend
                from fp32_attention import ATTENTION_TAG
                backend = Backend(entry['runtime']['snapshot'], entry['model'], ATTENTION_TAG)
            runtime = dict(entry['runtime'], versions={k: importlib.metadata.version(k) for k in ('torch','transformers')},
                dtype='float32', gpu_count=1, enable_thinking=False, retries=0, support_gold=True,
                test_gold_gloss=entry['oracle'], seed_rule='base + 2 * original test index',
                lock_sha256=sha256(BASE/'configs/lock.json'))
            provenance_path = destination(f'results/{ident}.runtime.json')
            if provenance_path.exists() and load(provenance_path) != runtime:
                raise ValueError('Resume runtime changed')
            save(provenance_path, runtime)
            images=[]
            if parent['context_kind']=='image':
                from PIL import Image
                images=[Image.open(ROOT/p).convert('RGB') for p in parent['context_files']]
            done={r['idx'] for r in rows}
            with path.open('a', encoding='utf-8') as handle:
                for i in range(len(parent['test'])):
                    if i in done: continue
                    if time.monotonic()-start > deadline:
                        save(f'results/{ident}.status.json', dict(status='partial_deadline', records=len(rows), expected=len(parent['test'])))
                        raise SystemExit(75)
                    system,user=active_prompt(parent,entry,i)
                    try:
                        attempt=generate(backend,system,user,images,entry['seed']+2*i,entry['budget'])
                    except Exception as exc:
                        save(f'results/{ident}.failure.json', dict(idx=i,error=repr(exc),records=len(rows),
                            expected=len(parent['test']),status='infrastructure_failure', prompt_evidence={'system':system,'user':user}))
                        raise
                    # The existing extraction policy is preserved; no prompt repair or semantic retry.
                    row=original.result_row(parent,i,attempt,'additional_v1')
                    row['fingerprint']=entry['fingerprint'];row['parent_fingerprint']=parent['fingerprint']
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
                    rows.append(row)
                    print(f'{ident}: {len(rows)}/{len(parent["test"])}',flush=True)
            validate(rows,parent,entry,True)
            save(f'results/{ident}.status.json',dict(status='complete',records=len(rows),expected=len(parent['test']),
                failures=sum(not r['translation']['prediction'] for r in rows),sha256=sha256(path)))
            for image in images: image.close()
            gc.collect();backend.torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--deadline',type=int,required=True)
    args=p.parse_args();run(args.group,args.deadline)
