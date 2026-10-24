"""Independent standard-library validation of saved reranker scores, quality and timing."""
import csv,json,math,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/modal_rerank/results'
def read(name):return json.loads((OUT/name).read_text())
manifest=read('manifest.json');summary=read('summary.json');assert summary['complete']
assert manifest['model']=='cross-encoder/ms-marco-MiniLM-L-6-v2' and manifest['revision']=='233902d25c440f23af6f7d6e94d2946bac0bee0a'
assert manifest['candidate_sizes']==[10,20,50,100] and manifest['timing_queries']==100 and manifest['timing_repeats']==3
for name,sha in manifest['input_hashes'].items():
    assert hashlib.sha256((ROOT/'outputs/splade'/f'{name}.jsonl').read_bytes()).hexdigest()==sha
rows=list(map(json.loads,(ROOT/'outputs/neural_inputs/queries.jsonl').open()));qmap={r['query_id']:r for r in rows}
cloud_queries={r['id']:r['text'] for r in map(json.loads,(ROOT/'outputs/splade/queries.jsonl').open())}
assert cloud_queries=={r['query_id']:r['query'] for r in rows}
cloud_products={r['id']:r['text'] for r in map(json.loads,(ROOT/'outputs/splade/products.jsonl').open())}
original_products={r['product_id']:r['text'] for r in map(json.loads,(ROOT/'outputs/neural_inputs/products.jsonl').open())}
assert cloud_products==original_products and len(cloud_products)==164900
expected={(r['query_id'],p) for r in rows for p in r['judgments']};scores={}
for row in csv.DictReader((OUT/'pair_scores.csv').open()):
    key=(int(row['query_id']),row['product_id']);assert key not in scores
    scores[key]=float(row['score']);assert math.isfinite(scores[key])
assert set(scores)==expected and len(scores)==181701
G={'E':1,'S':.1,'C':.01,'I':0}
def metric(order,j):
    den=sum(g/math.log2(i+2) for i,g in enumerate(sorted((G[v] for v in j.values()),reverse=True)[:10]))
    return sum(G[j[p]]/math.log2(i+2) for i,p in enumerate(order[:10]))/den if den else 0
observed={}
for r in csv.DictReader((OUT/'quality.csv').open()):
    k=(r['method'],int(r['candidate_size']),int(r['query_id']));assert k not in observed;observed[k]=r
results=[];maxdiff=0
sample=read('timing_query_ids.json');assert len(sample)==len(set(sample))==100 and set(sample)<=set(qmap)
def quantile(values,p):
    values=sorted(values);pos=(len(values)-1)*p;lo=int(pos);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)
for method in manifest['methods']:
    base={}
    for r in csv.DictReader((ROOT/'outputs/modal_common_t4/checkpoints/results'/method/'rankings.csv').open()):
        base.setdefault(int(r['query_id']),[]).append((int(r['rank']),r['product_id']))
    base={q:[p for _,p in sorted(v)] for q,v in base.items()}
    times=list(csv.DictReader((OUT/f'{method}_latencies.csv').open()));assert len(times)==1200
    keys={(int(r['candidate_size']),int(r['repeat']),int(r['query_id'])) for r in times}
    assert keys=={(n,repeat,q) for n in [10,20,50,100] for repeat in range(3) for q in sample}
    for r in times:
        assert r['method']==method
        a,b,c=(float(r[k]) for k in ('retrieval_seconds','reranking_seconds','total_seconds'))
        assert all(math.isfinite(v) and v>0 for v in (a,b,c)) and abs(a+b-c)<1e-10
    for n in [10,20,50,100]:
        metrics=[];baselines=[];recalls=[]
        for q,r in qmap.items():
            original=base[q];assert set(original)==set(r['judgments'])
            short=original[:n];order=sorted(short,key=lambda p:(-scores[q,p],p))+original[n:]
            val=metric(order,r['judgments']);baseline=metric(original,r['judgments'])
            relevant=sum(G[v]>0 for v in r['judgments'].values())
            recall=sum(G[r['judgments'][p]]>0 for p in short)/relevant if relevant else 0
            observed_row=observed[method,n,q]
            diffs=[abs(val-float(observed_row['ndcg_at_10'])),abs(baseline-float(observed_row['baseline_ndcg_at_10'])),abs(recall-float(observed_row['judged_relevant_candidate_recall']))]
            maxdiff=max(maxdiff,*diffs);assert max(diffs)<1e-12 and int(observed_row['actual_candidates'])==len(short)
            metrics.append(val);baselines.append(baseline);recalls.append(recall)
        ts=[float(r['total_seconds'])*1000 for r in times if int(r['candidate_size'])==n]
        rt=[float(r['reranking_seconds'])*1000 for r in times if int(r['candidate_size'])==n]
        results.append({'method':method,'candidates':n,'ndcg_at_10':statistics.mean(metrics),'baseline_ndcg_at_10':statistics.mean(baselines),'judged_relevant_candidate_recall':statistics.mean(recalls),'total_median_ms':statistics.median(ts),'total_p95_ms':quantile(ts,.95),'reranker_median_ms':statistics.median(rt),'timed_requests':len(ts)})
assert len(observed)==8956*20
report={'passed':True,'queries':8956,'pairs':181701,'settings':20,'timing_sample_queries':100,'timed_requests':6000,'maximum_metric_discrepancy':maxdiff,'manifest':manifest,'results':results,'cloud_summary':summary}
(ROOT/'reports/rerank_comparison_verified.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Shared reranker comparison','', 'Accuracy uses all 8,956 official judged-candidate sets. Latency uses exact full-corpus retrieval on a fixed random sample of100 queries, three shuffled repeats. These are different evaluation scopes; no whole-catalog recall claim.','', '| Method | Reranked candidates | NDCG@10 | Baseline | Total median / p95(ms) | Reranker median(ms) |','|---|---:|---:|---:|---:|---:|']
for r in results:lines.append(f"| {r['method']} | {r['candidates']} | {r['ndcg_at_10']:.4f} | {r['baseline_ndcg_at_10']:.4f} | {r['total_median_ms']:.2f} / {r['total_p95_ms']:.2f} | {r['reranker_median_ms']:.2f} |")
lines+=['','Pinned MiniLM reranker, revision '+manifest['revision']+', float16, pair max256 tokens, batch32, one T4, one CPU thread. Same original metadata for all methods; no generated expansion given to the reranker; no ESCI training. Neural retrievers retain their previous float32 settings. Each latency request includes fresh retrieval (including neural query encoding), pair tokenization/transfers/inference and deterministic top10. Network, model loading and cold starts excluded.','', 'N50/N100 may rerank the complete official candidate set for many queries. This cannot establish that full-catalog first-stage shortlists have equal quality. The shared quality score cache saves experiment compute; serving timings perform fresh scoring and do not use the cache.','', 'Modal metered usage is retrieved separately in outputs/modal_rerank/billing_report.json. Do not treat this whole experimental job bill as per-query serving cost or assign it to one method. Preparation costs remain in the previous verified report.','',f"Independent validation passed: {len(scores):,} unique pairs, all20 configurations on8,956 queries,6,000 timed requests. Maximum NDCG discrepancy {maxdiff:.3g}."]
(ROOT/'reports/rerank_comparison_verified.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'passed':True,'rows':len(results),'best_ndcg':max(results,key=lambda r:r['ndcg_at_10']),'fastest':min(results,key=lambda r:r['total_median_ms'])}))
