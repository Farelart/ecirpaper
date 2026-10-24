"""Evaluate fixed field ablations on all ESCI English test candidates."""
import argparse,csv,json,math,random,statistics,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecirpaper.retrieval import FIELDS,ANALYZER,tokenize,document_text,bm25_score
from ecirpaper.metrics import GAINS,evaluate_ranking
from run_bm25 import file_hash,percentile
DATA=ROOT/'data/processed/esci_us_small'
ENRICH=ROOT/'outputs/budget_enrichment/test/products_enriched.jsonl'
VARIANTS={'plain':(),'keywords':('keywords',),'synonyms':('synonyms',),'situations':('situations',),'all':('keywords','synonyms','situations')}


def extra(enrichment,fields):return tokenize(' '.join(s for f in fields for s in enrichment[f]))


def paired_interval(differences,repetitions,seed):
    rng=random.Random(seed);n=len(differences)
    samples=[sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(repetitions)]
    samples.sort()
    return [percentile(samples,.025),percentile(samples,.975)]


def load_enrichment(path):
    result={}
    with path.open() as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);pid=row['product_id']
            if pid in result:raise ValueError('Duplicate enriched product')
            value=row['enrichment']
            if set(value)!= {'keywords','synonyms','situations'} or any(not isinstance(v,list) or any(not isinstance(x,str) for x in v) for v in value.values()):raise ValueError('Invalid enrichment')
            result[pid]=value
    return result


def main(args):
    out=ROOT/'outputs/enrichment_evaluation'
    if out.exists():raise FileExistsError('Preserve previous evaluation; output already exists')
    enrichment=load_enrichment(ENRICH)
    c=duckdb.connect();c.execute("SET threads=1");c.execute("SET memory_limit='2GB'")
    c.execute(f"CREATE VIEW test AS SELECT * FROM read_parquet('{DATA}/test.parquet')")
    expected={r[0] for r in c.execute('SELECT DISTINCT product_id FROM test').fetchall()}
    if set(enrichment)!=expected:raise ValueError(f'Incomplete enrichment: {len(expected-set(enrichment))} missing, {len(set(enrichment)-expected)} extra')
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name in ('products.parquet','test.parquet'):
        if file_hash(DATA/name)!=next(r['sha256'] for r in manifest['files'] if r['name']==name):raise ValueError('Input checksum mismatch')
    out.mkdir(parents=True)
    df={v:Counter() for v in VARIANTS};lengths={v:0 for v in VARIANTS};n=0
    cursor=c.execute(f"SELECT p.product_id,{','.join('p.'+f for f in FIELDS)} FROM read_parquet('{DATA}/products.parquet') p SEMI JOIN (SELECT DISTINCT product_id FROM test) q USING(product_id)")
    while batch:=cursor.fetchmany(256):
        for row in batch:
            base=tokenize(document_text(row[1:]));base_set=set(base)
            for variant,fields in VARIANTS.items():
                addition=extra(enrichment[row[0]],fields)
                df[variant].update(base_set|set(addition));lengths[variant]+=len(base)+len(addition)
            n+=1
        if n%25600==0:print('Corpus statistics:',n,'products',flush=True)
    if n!=164900:raise ValueError('Unexpected corpus count')
    stats={v:{'documents':n,'average_length':lengths[v]/n,'document_frequency':df[v]} for v in VARIANTS}
    for v in VARIANTS:(out/f'{v}_corpus.json').write_text(json.dumps({'documents':n,'average_length':lengths[v]/n,'vocabulary':len(df[v])},indent=2)+'\n')
    cursor=c.execute(f"SELECT query_id,query,source,product_id,esci_label,{','.join(FIELDS)} FROM test ORDER BY query_id,product_id")
    cols=[r[0] for r in cursor.description]
    rows={v:[] for v in VARIANTS};streams={v:(out/f'{v}_rankings.csv').open('w',newline='') for v in VARIANTS}
    writers={v:csv.writer(f) for v,f in streams.items()}
    for w in writers.values():w.writerow(['query_id','product_id','rank','bm25_score'])
    count=0
    def score(group):
        nonlocal count
        qid=group[0]['query_id'];judgments={r['product_id']:r['esci_label'] for r in group}
        if len(judgments)!=len(group):raise ValueError('Duplicate candidates')
        terms=tokenize(group[0]['query'])
        bases={r['product_id']:tokenize(document_text(r[f] for f in FIELDS)) for r in group}
        for variant,fields in VARIANTS.items():
            scored=[]
            for pid,tokens in bases.items():
                value=bm25_score(terms,tokens+extra(enrichment[pid],fields),stats[variant],1.2,.75)
                scored.append((pid,value))
            scored.sort(key=lambda r:(-r[1],r[0]))
            metrics=evaluate_ranking([r[0] for r in scored],judgments)
            rows[variant].append({'query_id':qid,'source':group[0]['source'],'query_tokens':len(terms),'candidates':len(group),**metrics})
            for rank,(pid,value) in enumerate(scored,1):writers[variant].writerow([qid,pid,rank,format(value,'.17g')])
        count+=len(group)
        if len(rows['plain'])%1000==0:print('Evaluated',len(rows['plain']),'queries across five variants',flush=True)
    current=None;group=[]
    while batch:=cursor.fetchmany(256):
        for raw in batch:
            row=dict(zip(cols,raw))
            if current is not None and row['query_id']!=current:score(group);group=[]
            current=row['query_id'];group.append(row)
    if group:score(group)
    for f in streams.values():f.close()
    c.close()
    if count!=181701 or len(rows['plain'])!=8956:raise ValueError('Incomplete evaluation coverage')
    summary={'queries':8956,'candidate_pairs':count,'products':n,'k1':1.2,'b':.75,'gains':GAINS,'analyzer':ANALYZER,
             'original_fields':FIELDS,'corpus_scope':'All distinct English official test-candidate products, identical across methods.',
             'note':'This corpus differs from the initial 482,105-product plain baseline; this experiment reruns plain BM25 for paired comparisons.',
             'bootstrap_repetitions':args.bootstrap,'seed':42,'results':{},'paired_comparisons':{},
             'provenance':{'source_revision':manifest['source_revision'],'enrichment_sha256':file_hash(ENRICH),'test_sha256':file_hash(DATA/'test.parquet'),'implementation_sha256':file_hash(Path(__file__))},
             'built_at_utc':datetime.now(timezone.utc).isoformat()}
    for variant,values in rows.items():
        with (out/f'{variant}_per_query.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
        summary['results'][variant]={}
        for metric in ('ndcg_at_10','ndcg_full','mrr_exact','exact_at_1','recall_exact_at_10'):
            valid=[r[metric] for r in values if r[metric] is not None]
            summary['results'][variant][metric]={'mean':statistics.mean(valid),'queries':len(valid)}
        if variant!='plain':
            paired={}
            for metric in ('ndcg_at_10','ndcg_full'):
                differences=[r[metric]-base[metric] for r,base in zip(values,rows['plain'])]
                paired[metric]={'mean_difference':statistics.mean(differences),'bootstrap_95_percent_ci':paired_interval(differences,args.bootstrap,42),
                                'improved_queries':sum(d>1e-12 for d in differences),'worsened_queries':sum(d< -1e-12 for d in differences)}
            summary['paired_comparisons'][variant]=paired
    # Descriptive strata are fixed by source metadata and query token length.
    subgroups={}
    for group_type in ('source','query_length'):
        membership={}
        for i,row in enumerate(rows['plain']):
            label=str(row['source']) if group_type=='source' else ('1' if row['query_tokens']==1 else '2' if row['query_tokens']==2 else '3-5' if row['query_tokens']<=5 else '6+')
            membership.setdefault(label,[]).append(i)
        subgroups[group_type]={}
        for label,indices in membership.items():
            subgroups[group_type][label]={'queries':len(indices),'ndcg_at_10':{v:statistics.mean(rows[v][i]['ndcg_at_10'] for i in indices) for v in VARIANTS}}
    summary['descriptive_subgroups']=subgroups
    changes=[{'query_id':r['query_id'],'plain_ndcg_at_10':base['ndcg_at_10'],'all_ndcg_at_10':r['ndcg_at_10'],'difference':r['ndcg_at_10']-base['ndcg_at_10']} for r,base in zip(rows['all'],rows['plain'])]
    changes.sort(key=lambda r:(r['difference'],r['query_id']))
    (out/'largest_changes.json').write_text(json.dumps({'selection':'20 largest decreases and 20 largest increases; extremes, not representative quality samples.','decreases':changes[:20],'increases':changes[-20:]},indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'reports/enrichment_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Full ESCI enrichment comparison','', '8,956 queries, 181,701 candidate pairs, 164,900 products.','',
           '| Variant | NDCG@10 | Full NDCG | Exact at rank 1 | Change in NDCG@10 | Paired 95% interval |',
           '| --- | --- | --- | --- | --- | --- |']
    for v,res in summary['results'].items():
        comp=summary['paired_comparisons'].get(v,{}).get('ndcg_at_10')
        delta=f"{comp['mean_difference']:+.4f}" if comp else '—'
        ci=f"[{comp['bootstrap_95_percent_ci'][0]:+.4f}, {comp['bootstrap_95_percent_ci'][1]:+.4f}]" if comp else '—'
        lines.append(f"| {v} | {res['ndcg_at_10']['mean']:.4f} | {res['ndcg_full']['mean']:.4f} | {res['exact_at_1']['mean']:.4f} | {delta} | {ci} |")
    lines+=['',summary['note'],'','All variants use original full metadata plus their selected expansion fields. BM25 corpus statistics are recomputed separately for each variant. Relevance judgments are used only for evaluation. No parameter or prompt tuning was performed on these test results.',
            '','Paired bootstrap intervals use 1,000 query resamples with seed 42. Intervals are exploratory and not adjusted for multiple comparisons. This is candidate ranking, not whole-catalog recall, latency, or multi-turn evaluation.',
            '','Enrichment includes source-span keyword filtering and deterministic cleanup. See generation manifests and pilot audits for input truncation, batching, retries and semantic limitations.','']
    (ROOT/'reports/enrichment_evaluation.md').write_text('\n'.join(lines))
    print(json.dumps(summary['results'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args()
    if args.bootstrap<1:raise ValueError('Positive bootstrap count required')
    main(args)
