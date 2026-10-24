"""Matched Ling generated-query control, reusing the durable API budget."""
import argparse
import json
import re
import time
from pathlib import Path
import run_budget_enrichment as engine

ROOT=engine.ROOT
SOURCE=ROOT/'outputs/budget_enrichment/test/products_enriched.jsonl'
PROMPT=ROOT/'prompts/doc2query_ling_english_v1.txt'

def products(scope):
    with SOURCE.open() as stream:
        for line in stream:
            row=json.loads(line)
            yield {'product_id':row['product_id'],**row['input']}

def pack(rows):
    inputs=[{k:row[k] for k in ('product_title','product_brand','product_color',
                              'product_bullet_point','product_description')} for row in rows]
    messages=[{'role':'system','content':PROMPT.read_text()},
              {'role':'user','content':json.dumps([{'id':i,'text':v} for i,v in enumerate(inputs)],ensure_ascii=False,separators=(',',':'))}]
    count=len(engine.TOKENIZER.encode(engine.TEMPLATE.render(messages=messages,tools=None,
                enable_thinking=False,add_generation_prompt=True)).ids)
    if count>engine.INPUT_RESERVE*len(rows):
        raise ValueError('Matched source exceeds input reservation; do not truncate')
    return messages,inputs,count

def parse(raw,inputs):
    raw=re.sub(r'^```(?:json)?\s*','',raw.strip());raw=re.sub(r'\s*```$','',raw)
    values=json.loads(raw)
    if isinstance(values,dict) and len(inputs)==1:values=[values]
    if not isinstance(values,list) or len(values)!=len(inputs):raise ValueError('Incorrect batch length')
    cleaned=[]
    for i,item in enumerate(values):
        if isinstance(item,dict) and set(item)=={'id','query'}:
            item={'id':item['id'],'queries':item['query']}
        if not isinstance(item,dict) or set(item)!={'id','queries'} or item['id']!=i:raise ValueError('Invalid fields')
        qs=item['queries']
        if not isinstance(qs,list) or len(qs)!=3 or any(not isinstance(q,str) or not q.strip() or len(q)>160 for q in qs):
            raise ValueError('Expected three short query strings')
        cleaned.append({'queries':[q.strip() for q in qs]})
    return cleaned,[[] for _ in inputs]

def main():
    p=argparse.ArgumentParser();p.add_argument('--credentials-file',type=Path);a=p.parse_args()
    engine.PROMPT_PATH=PROMPT;engine.PROMPT=PROMPT.read_text()
    engine.pack=pack;engine.parse=parse;engine.products=products
    out=ROOT/'outputs/budget_enrichment/ling_doc2query';out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    protocol={'model':engine.MODEL,'input':'Exact saved metadata supplied to structured Ling; no further truncation',
              'source_sha256':engine.digest(SOURCE),'prompt_sha256':engine.digest(PROMPT),
              'products':164900,'queries_per_product':3,'batch_size':8,
              'output_token_limit_per_product':220,'temperature':0.2,'reasoning_enabled':False,
              'shared_budget_usd':4.4,'scope':'Matched model/input/output allowance; natural-language queries replace structured fields',
              'no_training':True,'no_labels_or_test_queries_used':True}
    mp=out/'control_protocol.json'
    if mp.exists():
        old=json.loads(mp.read_text());old.pop('workers',None)
        if old!=protocol:raise ValueError('Protocol changed')
    mp.write_text(json.dumps(protocol,indent=2)+'\n')
    for phase in range(3):
        engine.STOP.clear()
        args=argparse.Namespace(scope='ling_doc2query',budget=4.4,batch_size=8 if phase==0 else 1,
               workers=16,credentials_file=a.credentials_file,plan_only=False,retry_failed=phase>0)
        engine.main(args)
        rows=[json.loads(l) for l in (out/'products_enriched.jsonl').open()]
        (out/'products_queries.jsonl').write_text(''.join(json.dumps({'product_id':r['product_id'],'input':r['input'],
           'queries':r['enrichment']['queries']},ensure_ascii=False)+'\n' for r in rows))
        summary=json.loads((out/'summary.json').read_text())
        summary.update(complete=len(rows)==164900,session_wall_seconds=time.perf_counter()-started,protocol=protocol)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if summary['complete']:return
        if summary['stopped_for_safety']:raise SystemExit('Stopped at shared budget, pricing or request guard; preserve checkpoints')
    raise SystemExit('Incomplete after bounded retries; preserve outputs')

if __name__=='__main__':main()
