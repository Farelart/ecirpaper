"""Evaluate full dense/hybrid candidate rankings against frozen BM25 results."""
import csv
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecirpaper.metrics import evaluate_ranking, GAINS
from ecirpaper.fusion import reciprocal_rank_fusion

def load_rankings(path):
    result={}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            result.setdefault(int(row['query_id']),[]).append((int(row['rank']),row['product_id']))
    ordered={}
    for qid,items in result.items():
        items.sort()
        if [rank for rank,pid in items]!=list(range(1,len(items)+1)):raise ValueError('Invalid ranks')
        ordered[qid]=[pid for rank,pid in items]
    return ordered

def interval(differences):
    rng=np.random.default_rng(42); values=np.asarray(differences,dtype=np.float64); samples=[]
    for _ in range(100):
        positions=rng.integers(0,len(values),size=(10,len(values)))
        samples.extend(values[positions].mean(axis=1).tolist())
    return np.quantile(samples,[.025,.975]).tolist()

def summarize(rows):
    return {metric:{'mean':statistics.mean(r[metric] for r in rows if r[metric] is not None),
                    'queries':sum(r[metric] is not None for r in rows)}
            for metric in ('ndcg_at_10','ndcg_full','mrr_exact','exact_at_1','recall_exact_at_10')}

def main():
    enc=ROOT/'outputs/e5_small'; out=ROOT/'outputs/neural_evaluation'
    if out.exists():raise FileExistsError('Preserve previous evaluation')
    run=json.loads((enc/'summary.json').read_text())
    if not run.get('complete'):raise ValueError('Incomplete encoding')
    qrows=[json.loads(line) for line in (ROOT/'outputs/neural_inputs/queries.jsonl').open()]
    pids=json.loads((enc/'products_ids.json').read_text());qids=json.loads((enc/'queries_ids.json').read_text())
    products=np.load(enc/'products_vectors.npy',mmap_mode='r');queries=np.load(enc/'queries_vectors.npy',mmap_mode='r')
    if products.shape!=(164900,384) or queries.shape!=(8956,384):raise ValueError('Vector coverage mismatch')
    if not np.load(enc/'products_done.npy').all() or not np.load(enc/'queries_done.npy').all():raise ValueError('Incomplete checkpoint')
    if len(set(pids))!=164900 or len(set(qids))!=8956 or [r['query_id'] for r in qrows]!=qids:raise ValueError('ID alignment mismatch')
    for begin in range(0,len(products),1024):
        chunk=products[begin:begin+1024]
        if not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk,axis=1),1,atol=1e-5):raise ValueError('Invalid product vectors')
    if not np.isfinite(queries).all() or not np.allclose(np.linalg.norm(queries,axis=1),1,atol=1e-5):raise ValueError('Invalid query vectors')
    mapping={pid:i for i,pid in enumerate(pids)}
    if set(mapping)!={pid for r in qrows for pid in r['judgments']}:raise ValueError('Unexpected product IDs')
    baselines={v:load_rankings(ROOT/f'outputs/enrichment_evaluation/{f}_rankings.csv') for v,f in [('plain','plain'),('structured','all')]}
    if any(set(b)!=set(qids) for b in baselines.values()):raise ValueError('Baseline coverage mismatch')
    out.mkdir(parents=True)
    variants=('dense','hybrid_plain','hybrid_structured')
    results={v:[] for v in ('plain','structured')+variants};pairs=0
    streams={v:(out/f'{v}_rankings.csv').open('w',newline='') for v in variants}
    writers={v:csv.writer(s) for v,s in streams.items()}
    for w in writers.values():w.writerow(['query_id','product_id','rank','score'])
    compute_seconds=0
    for i,row in enumerate(qrows):
        qid=row['query_id'];truth=row['judgments'];ids=sorted(truth);pairs+=len(ids)
        if any(set(b[qid])!=set(ids) or len(b[qid])!=len(ids) for b in baselines.values()):raise ValueError('Candidate mismatch')
        start=time.perf_counter()
        scores=products[[mapping[pid] for pid in ids]]@queries[i]
        dense=sorted(zip(ids,scores.tolist()),key=lambda r:(-r[1],r[0]))
        compute_seconds+=time.perf_counter()-start
        dense_ids=[pid for pid,score in dense]
        rankings={'dense':dense,
                  'hybrid_plain':reciprocal_rank_fusion(baselines['plain'][qid],dense_ids),
                  'hybrid_structured':reciprocal_rank_fusion(baselines['structured'][qid],dense_ids)}
        for v,ranking in rankings.items():
            results[v].append({'query_id':qid,**evaluate_ranking([pid for pid,score in ranking],truth)})
            for rank,(pid,score) in enumerate(ranking,1):writers[v].writerow([qid,pid,rank,format(score,'.17g')])
        for v,baseline in baselines.items():
            results[v].append({'query_id':qid,**evaluate_ranking(baseline[qid],truth)})
        if (i+1)%1000==0:print('Evaluated',i+1,'queries across dense and two hybrids',flush=True)
    for s in streams.values():s.close()
    if pairs!=181701 or len(qrows)!=8956:raise ValueError('Incomplete evaluation')
    frozen=json.loads((ROOT/'reports/enrichment_evaluation.json').read_text())
    summary={'products':164900,'queries':8956,'candidate_pairs':pairs,'gains':GAINS,'rrf_k':60,
             'results':{v:summarize(rows) for v,rows in results.items()},'paired_comparisons':{},
             'encoding':run,'bootstrap_repetitions':1000,'bootstrap_seed':42,
             'dense_candidate_dot_and_sort_seconds':compute_seconds,
             'note':'Candidate ranking only. Query vectors were encoded offline in batches for effectiveness evaluation. These timings do not establish online per-search latency, index cost, or whole-catalog recall.'}
    for v,f in [('plain','plain'),('structured','all')]:
        for metric in summary['results'][v]:
            if abs(summary['results'][v][metric]['mean']-frozen['results'][f][metric]['mean'])>1e-12:raise ValueError('Frozen baseline metrics changed')
    for v in variants:
        for baseline in ('plain','structured'):
            comp={}
            for metric in ('ndcg_at_10','ndcg_full'):
                changes=[r[metric]-b[metric] for r,b in zip(results[v],results[baseline])]
                comp[metric]={'mean_difference':statistics.mean(changes),'bootstrap_95_percent_ci':interval(changes),
                              'improved_queries':sum(d>1e-12 for d in changes),'worsened_queries':sum(d< -1e-12 for d in changes)}
            summary['paired_comparisons'][v+'_vs_'+baseline]=comp
    for v,rows in results.items():
        with (out/f'{v}_per_query.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'reports/neural_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Full embedding and hybrid comparison','','All 8,956 queries, 181,701 pairs and 164,900 products.','',
           '| Method | NDCG@10 | Full NDCG | Exact at 1 |','| --- | --- | --- | --- |']
    for v,res in summary['results'].items():lines.append(f"| {v} | {res['ndcg_at_10']['mean']:.4f} | {res['ndcg_full']['mean']:.4f} | {res['exact_at_1']['mean']:.4f} |")
    lines+=['',summary['note'],'','E5-small uses a pinned checkpoint, full original metadata capped at 512 model tokens, required query/passage prefixes, masked mean pooling and L2 normalization. Hybrid uses equal-weight reciprocal rank fusion with k=60. No ESCI training or test-result tuning.',
            '','Paired intervals are exploratory, with 1,000 query bootstrap resamples and no multiple-comparison correction. Local API spending is zero; compute and energy are not free. See the encoding summary for measured runtime and representation bytes.','']
    (ROOT/'reports/neural_evaluation.md').write_text('\n'.join(lines))
    print(json.dumps(summary['results'],indent=2),flush=True)

if __name__=='__main__':main()
