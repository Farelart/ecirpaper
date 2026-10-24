"""Token-capped Ling enrichment with a shared durable dollar budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib, html, json, os, re, sqlite3, threading, time
from pathlib import Path
import urllib.request, urllib.error
import duckdb
from jinja2 import Environment
from tokenizers import Tokenizer
from run_enrichment_pilot import credentials, digest
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/esci_us_small'
MODEL='inclusionai/ling-3.0-flash'
ENDPOINT='https://ai-gateway.vercel.sh/v1/chat/completions'
TOKEN_DIR=ROOT/'data/models/ling_tokenizer'
PROMPT_PATH=ROOT/'prompts/enrichment_english_v2.txt'
TOKENIZER=Tokenizer.from_file(str(TOKEN_DIR/'tokenizer.json'))
TEMPLATE=Environment().from_string((TOKEN_DIR/'chat_template.jinja').read_text())
PROMPT=PROMPT_PATH.read_text()
INPUT_LIMIT=450
INPUT_RESERVE=500
OUTPUT_LIMIT=220
PRICE_IN=Decimal('0.000000021')
PRICE_OUT=Decimal('0.000000063')
STOP=threading.Event()
LOCK=threading.Lock()


def pack(rows):
    # Individual context budgets, not variable truncation driven by relevance.
    result=[]
    base={'product_title':140,'product_brand':40,'product_color':30,'product_bullet_point':350,'product_description':350}
    for row in rows:
        value={f:re.sub(r'\s+',' ',re.sub(r'<[^>]*>',' ',html.unescape(row.get(f) or ''))).strip()[:cap] for f,cap in base.items()}
        while len(TOKENIZER.encode(json.dumps(value,ensure_ascii=False,separators=(',',':'))).ids)>INPUT_LIMIT-120:
            key=max(('product_description','product_bullet_point','product_title'),key=lambda k:len(value[k]))
            value[key]=value[key][:max(0,len(value[key])-20)]
        result.append(value)
    messages=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps([{'id':i,'text':v} for i,v in enumerate(result)],ensure_ascii=False,separators=(',',':'))}]
    rendered=TEMPLATE.render(messages=messages,tools=None,enable_thinking=False,add_generation_prompt=True)
    estimate=len(TOKENIZER.encode(rendered).ids)
    # 450 tokens per product, including prompt and provider-template estimate.
    while estimate>INPUT_LIMIT*len(rows):
        found=False
        for value in result:
            for field in ('product_description','product_bullet_point','product_title'):
                if value[field]: value[field]=value[field][:-20];found=True
        if not found:raise ValueError('Prompt alone exceeds token budget')
        messages[1]['content']=json.dumps([{'id':i,'text':v} for i,v in enumerate(result)],ensure_ascii=False,separators=(',',':'))
        estimate=len(TOKENIZER.encode(TEMPLATE.render(messages=messages,tools=None,enable_thinking=False,add_generation_prompt=True)).ids)
    return messages,result,estimate


def normalized(s):return ' '.join(re.findall(r'[^\W_]+',s.casefold()))


def parse(raw,inputs):
    raw=re.sub(r'^```(?:json)?\s*','',raw.strip());raw=re.sub(r'\s*```$','',raw)
    # Repair only an unambiguous missing quote/colon in a known array key.
    raw=re.sub(r'(,\s*|{\s*)"(keywords|situations|synonyms)\[\s*"',r'\1"\2":["',raw)
    values=json.loads(raw)
    if isinstance(values,dict) and len(inputs)==1:values=[values]
    if not isinstance(values,list) or len(values)!=len(inputs):raise ValueError('Incorrect batch length')
    cleaned=[];removed=[]
    for index,(item,source) in enumerate(zip(values,inputs)):
        if not isinstance(item,dict) or set(item)!={'id','keywords','situations','synonyms'} or item.get('id')!=index:raise ValueError('Incorrect fields')
        for f,n in [('keywords',8),('situations',3),('synonyms',3)]:
            if not isinstance(item[f],list) or any(not isinstance(v,str) or not v.strip() for v in item[f]):raise ValueError('Invalid strings')
            # Deterministic cleanup only: preserve order, deduplicate and cap.
            seen=set();kept=[]
            for v in item[f]:
                v=v.strip()
                if len(v)<=160 and v.casefold() not in seen:
                    kept.append(v);seen.add(v.casefold())
            item[f]=kept[:n]
        text=' '+normalized(' '.join(source.values()))+' '
        dropped=[v for v in item['keywords'] if not normalized(v) or ' '+normalized(v)+' ' not in text]
        clean={f:list(item[f]) for f in ('keywords','situations','synonyms')}
        clean['keywords']=[v for v in item['keywords'] if v not in dropped]
        cleaned.append(clean);removed.append(dropped)
    return cleaned,removed


class Budget:
    def __init__(self,path,limit):
        self.path=path;self.limit=Decimal(str(limit))
        with sqlite3.connect(path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS charges(id INTEGER PRIMARY KEY, run TEXT, state TEXT, reserved TEXT, charged TEXT, created TEXT)')
    def used(self):
        with sqlite3.connect(self.path) as c:
            return sum((Decimal(r[0]) for r in c.execute("SELECT coalesce(charged,reserved) FROM charges")),Decimal(0))
    def reserve(self,count,run):
        amount=(PRICE_IN*INPUT_RESERVE+PRICE_OUT*OUTPUT_LIMIT)*count*Decimal('1.03')
        with LOCK,sqlite3.connect(self.path) as c:
            c.execute('BEGIN IMMEDIATE')
            used=sum((Decimal(r[0]) for r in c.execute('SELECT coalesce(charged,reserved) FROM charges')),Decimal(0))
            if STOP.is_set() or used+amount>self.limit:raise RuntimeError('Budget limit reached')
            cur=c.execute('INSERT INTO charges(run,state,reserved,created) VALUES (?,?,?,?)',(run,'reserved',str(amount),datetime.now(timezone.utc).isoformat()))
            return cur.lastrowid,amount
    def settle(self,identifier,actual):
        with LOCK,sqlite3.connect(self.path) as c:
            c.execute('UPDATE charges SET state=?,charged=? WHERE id=?',('settled',str(actual),identifier))


def price_check():
    models=json.load(urllib.request.urlopen('https://ai-gateway.vercel.sh/v1/models',timeout=30))['data']
    entry=next(r for r in models if r['id']==MODEL)
    p=entry['pricing']
    if Decimal(p['input'])>PRICE_IN or Decimal(p['output'])>PRICE_OUT:raise RuntimeError('Advertised model price exceeds configured ceiling')
    return entry


def call(rows,key,budget,run):
    messages,inputs,estimate=pack(rows)
    identifier,reserved=budget.reserve(len(rows),run)
    record={'product_ids':[r['product_id'] for r in rows],'inputs':inputs,'estimated_prompt_tokens':estimate,
            'batch_size':len(rows),'reservation_id':identifier,'reserved_usd':str(reserved),'created_at_utc':datetime.now(timezone.utc).isoformat()}
    payload={'model':MODEL,'messages':messages,'temperature':0.2,'max_tokens':OUTPUT_LIMIT*len(rows),'reasoning':{'enabled':False}}
    started=time.perf_counter()
    request=urllib.request.Request(ENDPOINT,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=120) as f:response=json.load(f)
        usage=response.get('usage',{})
        actual=usage.get('cost')
        if actual is not None:
            amount=Decimal(str(actual));budget.settle(identifier,amount)
            if amount>reserved:STOP.set()
        else:STOP.set() # Keep worst-case reservation; do not guess missing charges.
        record.update(usage=usage,response_id=response.get('id'),response_model=response.get('model'),elapsed_seconds=time.perf_counter()-started)
        choice=response['choices'][0];raw=choice['message'].get('content') or ''
        record.update(raw_output=raw,finish_reason=choice.get('finish_reason'))
        if usage.get('prompt_tokens',0)>INPUT_RESERVE*len(rows) or usage.get('completion_tokens',0)>OUTPUT_LIMIT*len(rows):STOP.set()
        if choice.get('finish_reason')!='stop':raise ValueError('Incomplete response')
        result,dropped=parse(raw,inputs)
        record.update(status='ok',enrichments=result,dropped_keywords=dropped)
    except urllib.error.HTTPError as e:
        record.update(status='http_error',http_status=e.code)
        STOP.set() # Unknown billing/rate limits: retain reserve and stop dispatch.
    except (ValueError,KeyError,TypeError) as e:record.update(status='invalid_output',error_type=type(e).__name__)
    except Exception as e:
        record.update(status='request_error',error_type=type(e).__name__);STOP.set()
    return record


def load_rows(scope):
    if scope=='pilot':return [json.loads(s) for s in (ROOT/'outputs/enrichment_pilot/products.jsonl').read_text().split('\n') if s]
    c=duckdb.connect()
    q=f"SELECT p.* FROM read_parquet('{DATA}/products.parquet') p SEMI JOIN (SELECT DISTINCT product_id FROM read_parquet('{DATA}/test.parquet')) t USING(product_id) ORDER BY p.product_id"
    cur=c.execute(q);cols=[r[0] for r in cur.description]
    while rows:=cur.fetchmany(256):
        for row in rows:yield dict(zip(cols,row))
    c.close()


def products(scope):
    # load_rows is a generator due to test branch; handle pilot explicitly.
    if scope=='pilot':return [json.loads(s) for s in (ROOT/'outputs/enrichment_pilot/products.jsonl').read_text().split('\n') if s]
    return load_rows(scope)


def export_records(records,out):
    saved={}
    for record in records:
        if record['status']=='ok':
            for pid,source,enriched,dropped in zip(record['product_ids'],record['inputs'],record['enrichments'],record['dropped_keywords']):
                saved[pid]={'product_id':pid,'input':source,'enrichment':enriched,'dropped_keywords':dropped,'response_id':record.get('response_id')}
    (out/'products_enriched.jsonl').write_text(''.join(json.dumps(saved[pid],ensure_ascii=False)+'\n' for pid in sorted(saved)))
    return len(saved)


def main(args):
    root=ROOT/'outputs/budget_enrichment'
    root.mkdir(parents=True,exist_ok=True)
    out=root/args.scope;out.mkdir(exist_ok=True)
    budget=Budget(root/'budget.sqlite',args.budget)
    entry=price_check()
    manifest={'model':MODEL,'prompt_sha256':digest(PROMPT_PATH),'tokenizer_sha256':digest(TOKEN_DIR/'tokenizer.json'),
              'tokenizer_revision':(TOKEN_DIR/'revision.txt').read_text().strip(),'input_token_limit_per_product':INPUT_LIMIT,
              'reserved_input_tokens_per_product':INPUT_RESERVE,'max_output_tokens_per_product':OUTPUT_LIMIT,'batch_size':args.batch_size,
              'budget_usd':args.budget,'implementation_sha256':digest(Path(__file__)),'pricing':entry['pricing'],'reasoning_enabled':False,
              'keyword_filter':'Keep only normalized exact source spans, independent of queries/labels.',
              'cleanup':'Deduplicate and retain first 8/3/3 entries; single-object wrapping allowed for one-product retry.',
              'source_products_sha256':digest(DATA/'products.parquet'),'source_test_sha256':digest(DATA/'test.parquet')}
    mp=out/'manifest.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        for k in ('model','prompt_sha256','tokenizer_sha256','batch_size','source_products_sha256','source_test_sha256'):
            if old[k]!=manifest[k] and not (k=='batch_size' and args.retry_failed):raise RuntimeError('Configuration changed; preserve the existing run')
    else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
    if args.plan_only:
        print(json.dumps({'products':164900,'maximum_reserved_cost_usd':str((PRICE_IN*INPUT_RESERVE+PRICE_OUT*OUTPUT_LIMIT)*164900*Decimal('1.03')),'budget_already_used_usd':str(budget.used()),'remaining_budget_usd':str(budget.limit-budget.used())},indent=2));return
    log=out/'batches.jsonl'
    previous=[json.loads(s) for s in log.read_text().split('\n') if s] if log.exists() else []
    done=set()
    for r in previous:
        if r.get('finish_reason')=='stop' and r.get('raw_output'):
            try:
                r['enrichments'],r['dropped_keywords']=parse(r['raw_output'],r['inputs']);r['status']='ok'
            except (ValueError,KeyError,TypeError):pass
        if r['status']=='ok' or not args.retry_failed:done.update(r['product_ids'])
    iterator=iter(r for r in products(args.scope) if r['product_id'] not in done)
    key=credentials(args.credentials_file)
    count=sum(len(r['product_ids']) for r in previous);records=list(previous)
    def batch():
        b=[]
        for _ in range(args.batch_size):
            try:b.append(next(iterator))
            except StopIteration:break
        return b
    with log.open('a') as stream,ThreadPoolExecutor(max_workers=args.workers) as pool:
        active=set()
        first=batch()
        if first:
            r=call(first,key,budget,args.scope);records.append(r)
            stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();count+=len(first)
            print('Initial batch:',r['status'],'cost',r.get('usage',{}).get('cost'),flush=True)
            if r['status'] in ('http_error','request_error'):STOP.set()
        last_price_check=time.monotonic()
        while not STOP.is_set() or active:
            if time.monotonic()-last_price_check>60 and not STOP.is_set():
                try:price_check()
                except Exception:STOP.set();print('Price verification failed; stopping dispatch',flush=True)
                last_price_check=time.monotonic()
            while len(active)<args.workers and not STOP.is_set():
                b=batch()
                if not b:break
                active.add(pool.submit(call,b,key,budget,args.scope))
            if not active:break
            ready,active=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            if not ready:
                print('Waiting for responses; completed',count,flush=True);continue
            for f in ready:
                try:r=f.result()
                except RuntimeError as e:STOP.set();print(str(e),flush=True);continue
                records.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();count+=len(r['product_ids'])
                if count%80==0:print('Completed',count,'budget used',str(budget.used()),flush=True)
    export_records(records,out)
    summary={'scope':args.scope,'attempted_products':count,'successful_products':len({pid for r in records if r['status']=='ok' for pid in r['product_ids']}),
             'batch_status':dict(Counter(r['status'] for r in records)),'reported_cost_usd':sum(float(r.get('usage',{}).get('cost',0)) for r in records),
             'budget_used_including_unknown_charges_usd':str(budget.used()),'stopped_for_safety':STOP.is_set()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['pilot','test'],default='pilot')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--budget',type=float,default=4.4)
    p.add_argument('--credentials-file',type=Path)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if not 1<=a.batch_size<=16 or not 1<=a.workers<=64 or not 0<a.budget<=4.4:raise ValueError('Settings exceed tested safety limits')
    main(a)
