"""Export frozen lexical token streams and source text for common-host timing."""
import gzip, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecirpaper.retrieval import tokenize

def main():
    out=ROOT/'outputs/modal_common_t4';out.mkdir(exist_ok=True)
    structured={}
    for line in (ROOT/'outputs/budget_enrichment/test/products_enriched.jsonl').open():
        r=json.loads(line);structured[r['product_id']]=' '.join(s for field in r['enrichment'].values() for s in field)
    expansions={}
    for name,path in [('doc2query_t5','outputs/doc2query_t5/full_b16/products.jsonl'),
                      ('ling_doc2query','outputs/budget_enrichment/ling_doc2query/products_queries.jsonl')]:
        expansions[name]={}
        for line in (ROOT/path).open():
            r=json.loads(line);expansions[name][r['product_id']]=' '.join(r['queries'])
    raw={json.loads(l)['product_id']:json.loads(l)['text'] for l in
         (ROOT/'outputs/final_benchmark_inputs/products_bm25.jsonl').open()}
    t5raw={json.loads(l)['product_id']:json.loads(l)['text'] for l in
           (ROOT/'outputs/neural_inputs/products.jsonl').open()}
    n=0;pids=[]
    with gzip.open(out/'lexical_inputs.jsonl.gz','wt',compresslevel=3) as dest:
        for line in (ROOT/'outputs/final_benchmark_inputs/canonical_tokens.jsonl').open():
            r=json.loads(line);pid=r['product_id'];pids.append(pid)
            r['doc2query_t5']=tokenize(t5raw[pid]+' '+expansions['doc2query_t5'][pid])
            r['raw_base']=raw[pid];r['raw_t5']=t5raw[pid]+' '+expansions['doc2query_t5'][pid]
            r['raw_structured_expansion']=structured[pid]
            r['raw_ling_expansion']=expansions['ling_doc2query'][pid]
            dest.write(json.dumps(r,ensure_ascii=False)+'\n');n+=1
            if n%25000==0:print('Prepared',n,flush=True)
    assert n==164900 and pids==sorted(set(pids))
    (out/'products_ids.json').write_text(json.dumps(pids)+'\n')
    for src,dest in [('outputs/neural_inputs/queries.jsonl','queries.jsonl'),
                     ('outputs/final_benchmark_inputs/query_tokens_reference.json','query_tokens_reference.json')]:
        (out/dest).write_bytes((ROOT/src).read_bytes())
    manifest={'products':n,'queries':8956,'t5_tokenizer_runtime':'Python3.9 frozen scorer',
              'files':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                       ('lexical_inputs.jsonl.gz','queries.jsonl','products_ids.json','query_tokens_reference.json')}}
    (out/'inputs_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared common-host inputs',flush=True)

if __name__=='__main__':main()
