"""Paginated, readable thesis-appendix version of the verified experiment matrix."""
import json
import csv
from pathlib import Path

from finalize_gold_v2 import OUT, MODELS, NAMES, METHODS, Image, ImageDraw, text, tsv


def reporting_scores():
    with (OUT/'scores.tsv').open() as handle:
        rows=list(csv.DictReader(handle,delimiter='\t'))
    with (OUT/'lezgi_sensitivity_scores.tsv').open() as handle:
        filtered={r['id']:r for r in csv.DictReader(handle,delimiter='\t') if r['cohort']=='valid_reference84'}
    final=[]
    for row in rows:
        item={k:row[k] for k in ('id','model','language','source','variant','method','material','cohort')}
        item['generated_records']=row['records']
        item['evaluated_records']=row['evaluated_records']
        for metric in ('bleu','chrf_plus_plus','xcomet_xl','xcomet_xxl'):
            item[metric]=row[metric]
        item['reference_policy']='full_configured_cohort' if row['cohort']=='native' else 'common_cohort'
        item['inference_status']='see_original_analysis_with_declared_correction_family'
        if row['language']=='Lezgi':
            f=filtered[row['id']]
            for metric in ('bleu','chrf_plus_plus','xcomet_xl','xcomet_xxl'):
                item[metric]=f[metric]
            item['evaluated_records']=f['records']
            item['reference_policy']='valid_reference84_excludes_indices_37_62_81'
            item['inference_status']='descriptive_only_no_filtered_cohort_p_values'
        final.append(item)
    if len(final)!=702 or len({(r['id'],r['cohort']) for r in final})!=702:
        raise ValueError('Incomplete reporting table')
    tsv(OUT/'thesis_scores.tsv',final)


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    conditions=audit['conditions_detail']
    if len(conditions)!=351 or not all(c['xl_verified'] and c['xxl_verified'] for c in conditions):
        raise ValueError('Complete verified audit required')
    reporting_scores()
    pages=[]
    lookup={(c['model'],c['method'],c['language'],c['source'],c['variant'],c['material']):c for c in conditions}
    keys=sorted({(c['language'],c['source'],c['variant'],c['material']) for c in conditions},
                key=lambda k:(k[3]!='baseline',k[0],k[1],k[2],k[3]))
    labels={'cheatsheet_txt':'Cheat sheet TXT','cheatsheet_jpg':'Cheat sheet JPG',
            'summary_tables_txt':'Summary tables TXT','summary_tables_jpg':'Summary tables JPG',
            'summary_text_txt':'Summary text TXT','pdfpages_impactful_jpg':'Selected pages JPG','baseline':'No-book baseline'}
    for model,name in zip(MODELS,NAMES):
        for start in (0,20):
            im=Image.new('RGB',(1500,2050),'white');d=ImageDraw.Draw(im)
            text(d,(35,25,1420,55),name+' | final experiment matrix',35,bold=True)
            text(d,(35,95,1420,45),f'matched_gold_v2 | Part {1 if start==0 else 2}/2 | Counts, not accuracy',25)
            for j,label in enumerate(('Gloss-shot','Chain-gloss','ModelGloss')):
                text(d,(660+260*j,165,250,40),label,25,bold=True)
            for i,key in enumerate(keys[start:start+20]):
                y=235+i*78;lang,source,variant,material=key
                source={'pdf1_brown':'Brown PDF1','pdf2_rigsby':'Rigsby PDF2'}.get(source,variant)
                text(d,(35,y,590,32),f'{lang} / {source}',24,bold=True)
                text(d,(35,y+35,590,32),labels[material],23)
                for j,method in enumerate(METHODS):
                    c=lookup[(model,method)+key];x=660+260*j
                    d.rectangle((x,y,x+246,y+66),fill='#14756b')
                    text(d,(x+9,y+3,229,33),f'{c["records"]}/{c["expected"]}',26,'white',True)
                    text(d,(x+9,y+37,229,28),f'Empty: {c["empty"]}',20,'white')
            notes=['All cells have verified XL and XXL scores. Empty outputs remain in evaluation.',
                   'Lezgi generation: 87 rows; report 84/83-row descriptive reference sensitivities.',
                   'Tsez: Qwen 445 rows, Gemini first 99. Gitksan sources share one baseline.',
                   'Source: thesis/audit.json. No Luna or MTOB conditions in this matrix.']
            for i,line in enumerate(notes):text(d,(35,1830+i*44,1430,38),line,24)
            im.save(OUT/'figures'/f'appendix_matrix_{model}_part{1 if start==0 else 2}.jpg',quality=95,subsampling=0)
            pages.append(im)
    pages[0].save(OUT/'figures/experiment_matrix_appendix.pdf',save_all=True,append_images=pages[1:],resolution=180)
    print('Saved six appendix pages and experiment_matrix_appendix.pdf')


if __name__=='__main__':main()
