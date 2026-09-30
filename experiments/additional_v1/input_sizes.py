"""Recorded input lengths and explicitly labelled offline reconstruction, no generation."""
import argparse
import statistics
from PIL import Image, ImageDraw
from common import *


def main(model):
    require_audit();plan=load(BASE/'configs/plan.json')
    runtime=next(e['runtime'] for e in plan['generation'] if e['model']==model)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(runtime['snapshot'],trust_remote_code=True,local_files_only=True)
    tokenizer=processor.tokenizer
    output=[]
    for item in catalog():
        if item['model']!=model:continue
        cfg=config(item);rows=original_rows(cfg)
        actual=[r['translation']['attempt'].get('input_tokens') for r in rows]
        observed=[x for x in actual if isinstance(x,int)]
        images=[];sizes=[]
        if cfg['context_kind']=='image':
            for name in cfg['context_files']:
                image=Image.open(ROOT/name).convert('RGB');images.append(image);sizes.append(image.size)
        visual=None
        if images:
            processed=processor.image_processor(images=images,return_tensors='pt')
            if 'image_grid_thw' in processed:
                merge=processor.image_processor.merge_size
                visual=int(processed['image_grid_thw'].prod(dim=1).sum().item()//(merge*merge))
        first=rows[0]['translation']['attempt']['prompt_evidence']
        output.append(dict(id=cfg['id'],model=model,language=cfg['language'],method=cfg['method'],material=cfg['material'],
            recorded_count=len(observed),expected_count=len(rows),recorded_input_mean=statistics.mean(observed) if observed else '',
            recorded_input_min=min(observed) if observed else '',recorded_input_max=max(observed) if observed else '',
            reconstructed_first_prompt_text_tokens=len(tokenizer.encode(first['system']+'\n'+first['user'])),
            reconstructed_grammar_text_tokens=len(tokenizer.encode(cfg.get('grammar_text',''),add_special_tokens=False)),
            reconstructed_visual_tokens=visual if visual is not None else '',image_count=len(images),
            image_dimensions=json.dumps(sizes),reconstruction_revision=runtime['revision'],
            limitation='Reconstructed components are not original logged counts and are not additive to chat-template length',
            result_sha256=sha256(ROOT/cfg['results'])))
        for image in images:image.close()
        print(cfg['id'],flush=True)
    table(f'reports/C_input_sizes_{model}.tsv',output)
    # Small stratified diagnostic plots; input size alone does not identify a causal effect.
    for language in LANGUAGES:
        selected=[r for r in output if r['language']==language and r['recorded_count']]
        if not selected:continue
        canvas=Image.new('RGB',(1100,700),'white');draw=ImageDraw.Draw(canvas)
        draw.text((40,20),f'{model} / {language}: recorded mean input tokens by condition index',fill='black')
        draw.line((70,70,70,630,1050,630),fill='black',width=2)
        maximum=max(r['recorded_input_mean'] for r in selected) or 1
        for i,row in enumerate(selected):
            x=80+i*950/max(1,len(selected)-1);y=620-row['recorded_input_mean']/maximum*520
            draw.ellipse((x-4,y-4,x+4,y+4),fill='black')
        draw.text((75,645),f'Index follows C_input_sizes table. Maximum: {maximum:.0f} tokens. Descriptive only.',fill='black')
        canvas.save(destination(f'reports/C_{model}_{language.lower()}.jpg'),quality=92)
    save(f'reports/C_{model}.json',dict(status='complete',conditions=len(output),model_revision=runtime,
        limitation='Historical preprocessing package versions may be unavailable; reconstructions cannot replace actual logs.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);main(p.parse_args().model)
