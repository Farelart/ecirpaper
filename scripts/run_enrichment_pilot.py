"""Prepare and run a reproducible metadata-only 200-product expansion pilot."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/esci_us_small'
OUT = ROOT / 'outputs/enrichment_pilot'
PROMPT = ROOT / 'prompts/enrichment_english_v1.txt'
FIELDS = ('product_title','product_description','product_bullet_point','product_brand','product_color')
CAPS = {'product_description':4000,'product_bullet_point':2000}
MODEL = 'google/gemini-3-flash'
ENDPOINT = 'https://ai-gateway.vercel.sh/v1/chat/completions'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'products.jsonl'
    if path.exists(): return [json.loads(s) for s in path.read_text().split('\n') if s]
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name in ('products.parquet','train.parquet','test.parquet'):
        expected=next(r['sha256'] for r in manifest['files'] if r['name']==name)
        if digest(DATA/name)!=expected: raise ValueError('Dataset checksum mismatch')
    c=duckdb.connect()
    # Labels and query text are never selected. Exclude any product appearing in
    # official test candidates; balance four metadata-availability strata.
    sql=f"""WITH eligible AS (
      SELECT p.*, (CASE WHEN length(trim(coalesce(product_description,'')))>0 THEN 'description' ELSE 'no_description' END)
       || '+' || (CASE WHEN length(trim(coalesce(product_bullet_point,'')))>0 THEN 'bullets' ELSE 'no_bullets' END) AS stratum
      FROM read_parquet('{DATA}/products.parquet') p
      SEMI JOIN (SELECT DISTINCT product_id FROM read_parquet('{DATA}/train.parquet')) t USING(product_id)
      ANTI JOIN (SELECT DISTINCT product_id FROM read_parquet('{DATA}/test.parquet')) e USING(product_id)
    ), chosen AS (
      SELECT *,row_number() OVER(PARTITION BY stratum ORDER BY hash(product_id || ':42'),product_id) AS position
      FROM eligible
    ) SELECT product_id,{','.join(FIELDS)},stratum FROM chosen WHERE position<=50 ORDER BY stratum,product_id"""
    cur=c.execute(sql)
    names=[r[0] for r in cur.description]
    rows=[dict(zip(names,row)) for row in cur.fetchall()]
    c.close()
    if len(rows)!=200 or len({r['product_id'] for r in rows})!=200: raise ValueError('Incomplete pilot strata')
    for r in rows:
        r['input']={f:(r[f] or '')[:CAPS.get(f,len(r[f] or ''))] for f in FIELDS}
        r['truncated_fields']=[f for f in CAPS if len(r[f] or '')>CAPS[f]]
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    record={'products':len(rows),'selection':'50 per description/bullets availability stratum, official train-only products',
            'seed':42,'duckdb_version':duckdb.__version__,'source_revision':manifest['source_revision'],
            'sample_sha256':digest(path),'prompt_sha256':digest(PROMPT),'input_caps_characters':CAPS,
            'strata':dict(Counter(r['stratum'] for r in rows)),'model':MODEL,'temperature':0.4,'max_tokens':1500,
            'endpoint':ENDPOINT,'test_products_excluded':True,'queries_and_labels_used_for_generation':False,
            'created_at_utc':datetime.now(timezone.utc).isoformat()}
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return rows


def credentials(path):
    key=os.getenv('AI_GATEWAY_API_KEY2') or os.getenv('AI_GATEWAY_API_KEY')
    if key: return key
    if path:
        values={}
        for line in path.read_text().splitlines():
            k,sep,v=line.partition('=')
            if sep and k.strip() in ('AI_GATEWAY_API_KEY2','AI_GATEWAY_API_KEY'):
                values[k.strip()]=v.strip().strip('\"\'')
        key=values.get('AI_GATEWAY_API_KEY2') or values.get('AI_GATEWAY_API_KEY')
    if not key: raise ValueError('No AI gateway credential configured')
    return key


def validate(raw):
    text=raw.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text)
        text=re.sub(r'\s*```$','',text)
    value=json.loads(text)
    if not isinstance(value,dict) or set(value)!= {'keywords','situations','synonyms'}:
        raise ValueError('Expected exactly three enrichment fields')
    for field,limit in [('keywords',15),('situations',8),('synonyms',8)]:
        items=value[field]
        if not isinstance(items,list) or len(items)>limit or any(not isinstance(x,str) or not x.strip() for x in items):
            raise ValueError('Invalid enrichment array')
        if len({x.strip().casefold() for x in items})!=len(items): raise ValueError('Duplicate entries')
    return value


def generate(row,key,prompt):
    started=time.perf_counter()
    payload={'model':MODEL,'temperature':0.4,'max_tokens':1500,
             'messages':[{'role':'system','content':prompt},
                         {'role':'user','content':'Product metadata:\n'+json.dumps(row['input'],ensure_ascii=False)}]}
    request=urllib.request.Request(ENDPOINT,data=json.dumps(payload).encode(),
                                  headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=90) as response: result=json.load(response)
        choice=result['choices'][0]
        raw=choice['message'].get('content') or ''
        record={'product_id':row['product_id'],'stratum':row['stratum'],'raw_output':raw,
                'usage':result.get('usage',{}),'response_id':result.get('id'),
                'response_model':result.get('model'),'finish_reason':choice.get('finish_reason'),
                'elapsed_seconds':time.perf_counter()-started,'created_at_utc':datetime.now(timezone.utc).isoformat()}
        try:
            if choice.get('finish_reason')!='stop': raise ValueError('Incomplete model response')
            record['enrichment']=validate(raw)
            record['status']='ok'
        except (ValueError,TypeError): record.update(status='invalid_output')
        return record
    except urllib.error.HTTPError as e:
        # Never print response bodies, headers or credentials.
        return {'product_id':row['product_id'],'stratum':row['stratum'],'status':'http_error','http_status':e.code}
    except Exception as e:
        return {'product_id':row['product_id'],'stratum':row['stratum'],'status':'request_error','error_type':type(e).__name__}


def main(args):
    rows=prepare()
    print('Prepared 200 metadata-only products in four equal strata',flush=True)
    if args.prepare_only: return
    manifest=json.loads((OUT/'manifest.json').read_text())
    if digest(PROMPT)!=manifest['prompt_sha256'] or digest(OUT/'products.jsonl')!=manifest['sample_sha256']:
        raise ValueError('Pilot inputs changed; preserve this run and start a new one')
    log=OUT/'generations.jsonl'
    previous=[json.loads(s) for s in log.read_text().split('\n') if s] if log.exists() else []
    done={r['product_id'] for r in previous}
    pending=[r for r in rows if r['product_id'] not in done]
    key=credentials(args.credentials_file)
    prompt=PROMPT.read_text()
    # Verify one API call before launching the remaining bounded pilot.
    records=list(previous)
    if pending:
        first=generate(pending.pop(0),key,prompt)
        with log.open('a') as f: f.write(json.dumps(first,ensure_ascii=False)+'\n')
        records.append(first)
        print('Initial request:',first['status'],flush=True)
        if first['status']!='ok':
            print('Stopped before bulk requests; inspect saved status.',flush=True)
            return
    with ThreadPoolExecutor(max_workers=4) as pool, log.open('a') as f:
        tasks=[pool.submit(generate,r,key,prompt) for r in pending]
        for future in as_completed(tasks):
            r=future.result()
            records.append(r)
            f.write(json.dumps(r,ensure_ascii=False)+'\n'); f.flush()
            if len(records)%20==0: print(f'Completed {len(records)}/200: {dict(Counter(r["status"] for r in records))}',flush=True)
    status=Counter(r['status'] for r in records)
    good=[r for r in records if r['status']=='ok']
    usage={k:sum(r.get('usage',{}).get(k,0) or 0 for r in records) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    summary={'products':len(rows),'attempted':len(records),'status':dict(status),'usage':usage,
             'mean_entries':{k:statistics.mean(len(r['enrichment'][k]) for r in good) if good else None for k in ('keywords','situations','synonyms')},
             'mean_request_seconds':statistics.mean(r['elapsed_seconds'] for r in records if 'elapsed_seconds' in r),
             'quality_review':'Schema checks only; semantic grounding requires inspecting source/output pairs.',
             'implementation_sha256':digest(Path(__file__))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'reports/enrichment_pilot.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--credentials-file',type=Path)
    main(parser.parse_args())
