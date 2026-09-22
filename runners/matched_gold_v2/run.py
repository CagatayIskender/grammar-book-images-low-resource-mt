"""Matched generation; never repair prompts or discard failed sentence indices."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time

from protocol import ROOT, prompt, result_row, validate_rows, verify_inputs, write_rows, digest
from experiment_io import atomic_json, read_records, sha256


def basic_metrics(cfg, rows):
    import sacrebleu
    validate_rows(rows,cfg,complete=True)
    rows=sorted(rows,key=lambda x:x['idx'])
    refs=[[r['reference'] for r in rows]];hyps=[r['translation']['prediction'] for r in rows]
    atomic_json(ROOT/cfg['metrics'],dict(translation=dict(bleu=sacrebleu.corpus_bleu(hyps,refs).score,
        chrf=sacrebleu.corpus_chrf(hyps,refs,word_order=2).score,xcomet=None),records=len(rows),
        errors=sum(bool(r['translation']['error']) for r in rows),empty=sum(not s for s in hyps),
        missing_gloss=sum('missing_gloss' in r['translation']['flags'] for r in rows),
        fingerprint=cfg['fingerprint'],results_sha256=sha256(ROOT/cfg['results'])))


class API:
    def __init__(self,cfg):
        import requests
        self.requests=requests;self.model=cfg['model_id'];self.cfg=cfg
        self.key=os.environ.get('OPENROUTER_API_KEY')
        if not self.key:raise RuntimeError('OPENROUTER_API_KEY missing')
        self.cap=float(os.environ.get('MATCHED_API_BUDGET_USD','10' if cfg['model']=='gemini25flashlite' else '0'))
        if self.cap<=0:raise RuntimeError('A positive explicit API budget is required')
        response=requests.get('https://openrouter.ai/api/v1/models',timeout=60)
        response.raise_for_status()
        model=next((m for m in response.json()['data'] if m['id']==self.model),None)
        if model is None:raise RuntimeError('Configured API model unavailable; no substitution')
        reasoning=model.get('reasoning') or {}
        if reasoning.get('mandatory') or (reasoning.get('supported_efforts') and 'none' not in reasoning['supported_efforts']):
            raise RuntimeError('Model does not allow reasoning off')
        self.pricing=model['pricing']
        self.ledger=ROOT/'docs/matched_v1'/f'{cfg["model"]}_budget.json'
        self.ledger.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(self.ledger.with_suffix('.model.json'),model)

    def reserve(self,amount):
        with self.ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=json.loads(self.ledger.read_text()) if self.ledger.exists() else {'reserved_upper_bound_usd':0,'requests':0}
            if ledger['reserved_upper_bound_usd']+amount>self.cap:
                raise RuntimeError('API budget reached; no further requests sent')
            ledger['reserved_upper_bound_usd']+=amount;ledger['requests']+=1
            ledger['cap_usd']=self.cap
            atomic_json(self.ledger,ledger)

    def settle(self,reserved,actual):
        import math
        if not isinstance(actual,(int,float)) or not math.isfinite(actual) or actual<0:
            return  # Unknown billing retains the conservative reservation.
        with self.ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=json.loads(self.ledger.read_text())
            ledger['reserved_upper_bound_usd']+=actual-reserved
            ledger['reported_cost_usd']=ledger.get('reported_cost_usd',0)+actual
            atomic_json(self.ledger,ledger)
        if actual>reserved:
            raise RuntimeError('Reported API cost exceeded conservative reservation; stop for budget review')

    def generate(self,system,user,images,cfg):
        import base64
        content=[{'type':'text','text':user}]
        for p in images:
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(Path(p).read_bytes()).decode()}})
        payload=dict(model=self.model,messages=[{'role':'system','content':system},{'role':'user','content':content}],
                     max_tokens=512,seed=42,reasoning={'effort':'none'},
                     provider={'data_collection':'deny','require_parameters':True})
        if cfg['model']=='gemini25flashlite':payload['provider']['only']=['Google']
        if cfg['policy']['temperature'] is not None:payload['temperature']=cfg['policy']['temperature']
        # Conservative local ceiling; request accounting is not a provider billing guarantee.
        prices=[self.pricing]+self.pricing.get('overrides',[])
        prompt_price=max(float(p.get('prompt',0)) for p in prices)
        output_price=max(float(p.get('completion',0)) for p in prices)
        reserve=(len((system+user).encode())+2048+20000*len(images))*prompt_price+512*output_price+float(self.pricing.get('request','0') or 0)+len(images)*float(self.pricing.get('image','0') or 0)
        reserve=max(reserve,0.000001)
        for retry in range(5):
            self.reserve(reserve)
            try:
                response=self.requests.post('https://openrouter.ai/api/v1/chat/completions',
                    headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','X-Title':'GrammarMT matched ablation'},
                    json=payload,timeout=180)
            except self.requests.RequestException:
                if retry==4:raise RuntimeError('API transport failed; prompt unchanged, resume required')
                time.sleep(min(60,2**retry));continue
            if response.status_code in (408,429,500,502,503,504):
                if retry==4:raise RuntimeError(f'API transient HTTP {response.status_code}; resume required')
                time.sleep(min(60,2**retry));continue
            if not response.ok:
                if response.status_code in (400,403) and any(marker in response.text.lower() for marker in ('prohibited_content','content_filter','content filter','safety filter')):
                    return dict(raw='',api_error='content_filter',finish_reason='content_filter',request_sha256=digest(payload),transport_attempts=retry+1)
                raise RuntimeError(f'API HTTP {response.status_code}; no model/prompt fallback')
            data=response.json();choice=data['choices'][0];message=choice['message'];usage=data.get('usage',{})
            self.settle(reserve,usage.get('cost'))
            reasoning=any(message.get(k) for k in ('reasoning','reasoning_content','reasoning_details')) or bool((usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
            if data.get('model',self.model)!=self.model:raise RuntimeError('Unexpected API model')
            return dict(raw=message.get('content') or '',truncated=choice.get('finish_reason')=='length',
                        reasoning_violation=reasoning,finish_reason=choice.get('finish_reason'),
                        api={k:data.get(k) for k in ('id','model','provider','usage')},
                        request_sha256=digest(payload),transport_attempts=retry+1)
        raise AssertionError('Transport retry loop')


def run_group(group,deadline_seconds):
    from run_audited_context import Backend
    from run_sampled_context import generate
    from fp32_attention import ATTENTION_TAG
    configs=json.loads((ROOT/group).read_text())
    backend=None;started=time.monotonic();unavailable=[]
    for cfg_path in configs:
        cfg=json.loads((ROOT/cfg_path).read_text());verify_inputs(cfg)
        result=ROOT/cfg['results'];result.parent.mkdir(parents=True,exist_ok=True)
        with result.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            rows=read_records(result);validate_rows(rows,cfg)
            if len(rows)==len(cfg['test']):
                metric=ROOT/cfg['metrics']
                if not metric.exists() or json.loads(metric.read_text()).get('results_sha256')!=sha256(result):basic_metrics(cfg,rows)
                print('[SKIP complete]',cfg['id'],flush=True);continue
            status_path=result.with_suffix('.status.json')
            if status_path.exists() and json.loads(status_path.read_text()).get('status')=='resource_unavailable':
                unavailable.append(cfg['id']);continue
            atomic_json(result.with_suffix('.provenance.json'),cfg)
            images=[str(ROOT/p) for p in cfg['context_files']] if cfg['context_kind']=='image' else []
            loaded=[]
            if cfg['model'] in ('qwen3','qwen35'):
                from PIL import Image
                loaded=[Image.open(p).convert('RGB') for p in images]
                if backend is None:backend=Backend(cfg['model_id'],cfg['model'],ATTENTION_TAG)
            elif backend is None:backend=API(cfg)
            done={r['idx'] for r in rows}
            # Longest inputs are a resource check only. Their first outputs are retained.
            longest=sorted(range(len(cfg['test'])),key=lambda i:len(cfg['test'][i]['source']),reverse=True)[:3]
            order=longest+[i for i in range(len(cfg['test'])) if i not in longest]
            condition_start=time.monotonic();new_count=0;failed_resource=False
            for index in order:
                if index in done:continue
                if time.monotonic()-started>deadline_seconds:
                    print('[CHECKPOINT] Time budget reached; rerun same group to resume',flush=True)
                    return 75
                active=dict(cfg,current_gloss=cfg.get('predicted_glosses',['']*len(cfg['test']))[index])
                system,user=prompt(active,cfg['test'][index]['source'])
                try:
                    if cfg['model'] in ('qwen3','qwen35'):
                        attempt=generate(backend,system,user,loaded,cfg['policy']['seed']+2*index)
                    else:attempt=backend.generate(system,user,images,cfg)
                except Exception as exc:
                    if cfg['model'] in ('qwen3','qwen35') and isinstance(exc,backend.torch.cuda.OutOfMemoryError):
                        atomic_json(status_path,dict(status='resource_unavailable',reason='single_H100_FP32_OOM',
                            records=len(rows),expected=len(cfg['test']),failed_index=index,fingerprint=cfg['fingerprint']))
                        unavailable.append(cfg['id']);failed_resource=True
                        gc.collect();backend.torch.cuda.empty_cache();break
                    raise
                attempt['user_prompt_sha256']=digest([system,user])
                attempt['prompt_evidence']={'system':system,'user':user,'support_count':21,
                    'nonempty_gold_glosses':21,'support_sha256':cfg['gold_support_sha256']}
                row=result_row(cfg,index,attempt,dict(type='generated_first_attempt',config=cfg_path))
                rows.append(row);done.add(index);new_count+=1
                write_rows(result,rows)
                elapsed=time.monotonic()-condition_start
                eta=elapsed/new_count*(len(cfg['test'])-len(rows))
                print(f'[PROGRESS] {cfg["id"]} {len(rows)}/{len(cfg["test"])} ETA={eta/60:.1f}min error={row["translation"]["error"]}',flush=True)
                if attempt.get('reasoning_violation'):
                    raise RuntimeError('Reasoning violation recorded; backend aborted')
            for image in loaded:image.close()
            if failed_resource:continue
            basic_metrics(cfg,rows)
            atomic_json(status_path,dict(status='complete_with_errors' if any(r['translation']['error'] for r in rows) else 'complete',
                records=len(rows),fingerprint=cfg['fingerprint'],results_sha256=sha256(result)))
    print('[SUMMARY] Resource-unavailable conditions:',unavailable,flush=True)
    return 2 if unavailable else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--deadline-seconds',type=int,default=34200)
    a=p.parse_args();sys.exit(run_group(a.group,a.deadline_seconds))

