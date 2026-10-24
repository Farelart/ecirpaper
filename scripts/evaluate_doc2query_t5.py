"""Evaluate completed pinned T5 expansion on every official test candidate."""
import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecirpaper.retrieval import tokenize,bm25_score,ANALYZER
from ecirpaper.metrics import evaluate_ranking,GAINS
from evaluate_neural_methods import load_rankings,interval,summarize

def main(args):
    evaluation_started=time.perf_counter()
    generation=ROOT/args.generation; out=ROOT/'outputs/doc2query_evaluation'
    if out.exists():raise FileExistsError('Preserve prior evaluation')
    generated=json.loads((generation/'summary.json').read_text())
    if not generated.get('complete') or generated['products']!=164900:raise ValueError('Incomplete T5 generation')
    originals={r['product_id']:r['text'] for r in (json.loads(line) for line in (ROOT/'outputs/neural_inputs/products.jsonl').open())}
    expansions={}
    with (generation/'products.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line);pid=r['product_id'];queries=r['queries']
            if pid in expansions or not isinstance(queries,list) or len(queries)!=generated['configuration']['num_return_sequences'] or any(not isinstance(q,str) for q in queries):raise ValueError('Invalid T5 row')
            expansions[pid]=tokenize(' '.join(queries))
    if len(originals)!=164900 or set(expansions)!=set(originals):raise ValueError('Product coverage mismatch')
    corpus_started=time.perf_counter()
    frequencies=Counter();total=0
    for i,(pid,text) in enumerate(originals.items(),1):
        tokens=tokenize(text)+expansions[pid];frequencies.update(set(tokens));total+=len(tokens)
        if i%25000==0:print('T5 corpus statistics:',i,'products',flush=True)
    stats={'documents':len(originals),'average_length':total/len(originals),'document_frequency':frequencies}
    corpus_statistics_seconds=time.perf_counter()-corpus_started
    qrows=[json.loads(line) for line in (ROOT/'outputs/neural_inputs/queries.jsonl').open()]
    baselines={v:load_rankings(ROOT/f'outputs/enrichment_evaluation/{f}_rankings.csv') for v,f in [('plain','plain'),('structured','all')]}
    if any(set(b)!={r['query_id'] for r in qrows} for b in baselines.values()):raise ValueError('Baseline coverage mismatch')
    out.mkdir(parents=True); results={v:[] for v in ('plain','structured','doc2query_t5')};pairs=0
    candidate_scoring_and_sort_seconds=0.0
    ranking_started=time.perf_counter()
    with (out/'doc2query_t5_rankings.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['query_id','product_id','rank','bm25_score'])
        for i,row in enumerate(qrows,1):
            qid=row['query_id'];truth=row['judgments'];terms=tokenize(row['query']);pairs+=len(truth)
            scoring_started=time.perf_counter()
            scored=[(pid,bm25_score(terms,tokenize(originals[pid])+expansions[pid],stats,1.2,.75)) for pid in sorted(truth)]
            scored.sort(key=lambda r:(-r[1],r[0]))
            candidate_scoring_and_sort_seconds+=time.perf_counter()-scoring_started
            results['doc2query_t5'].append({'query_id':qid,**evaluate_ranking([pid for pid,s in scored],truth)})
            for rank,(pid,score) in enumerate(scored,1):writer.writerow([qid,pid,rank,format(score,'.17g')])
            for v,b in baselines.items():results[v].append({'query_id':qid,**evaluate_ranking(b[qid],truth)})
            if i%1000==0:print('T5 evaluated',i,'queries',flush=True)
    if pairs!=181701 or len(qrows)!=8956:raise ValueError('Candidate coverage mismatch')
    ranking_loop_seconds=time.perf_counter()-ranking_started
    summary={'products':164900,'queries':8956,'candidate_pairs':pairs,'gains':GAINS,'k1':1.2,'b':.75,'analyzer':ANALYZER,
             'results':{v:summarize(rows) for v,rows in results.items()},'paired_comparisons':{},'generation':generated,
             'corpus':{'documents':164900,'average_length':stats['average_length'],'vocabulary':len(frequencies)},
             'note':'Pinned pretrained doc2query-T5 with 3 queries per product: economical configuration, not reproduction of the original 40-query MS MARCO experiment. Same saved metadata inputs as structured LLM; full original metadata remains in BM25.'}
    frozen=json.loads((ROOT/'reports/enrichment_evaluation.json').read_text())
    for v,f in [('plain','plain'),('structured','all')]:
        for metric,res in summary['results'][v].items():
            if abs(res['mean']-frozen['results'][f][metric]['mean'])>1e-12:raise ValueError('Frozen baseline changed')
        comp={}
        for metric in ('ndcg_at_10','ndcg_full'):
            changes=[r[metric]-b[metric] for r,b in zip(results['doc2query_t5'],results[v])]
            comp[metric]={'mean_difference':statistics.mean(changes),'bootstrap_95_percent_ci':interval(changes),
                          'improved_queries':sum(d>1e-12 for d in changes),'worsened_queries':sum(d< -1e-12 for d in changes)}
        summary['paired_comparisons']['doc2query_t5_vs_'+v]=comp
    with (out/'doc2query_t5_per_query.csv').open('w',newline='') as stream:
        rows=results['doc2query_t5'];writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary['timing']={'corpus_statistics_seconds':corpus_statistics_seconds,
                       'candidate_scoring_and_sort_seconds':candidate_scoring_and_sort_seconds,
                       'ranking_metrics_and_csv_seconds':ranking_loop_seconds,
                       'evaluation_wall_seconds_before_report_write':time.perf_counter()-evaluation_started,
                       'note':'Offline evaluation phases; scoring includes candidate document tokenization. Not indexed whole-catalog search or online serving latency. Timed while the experiment pipeline is active; not a controlled cross-method latency benchmark.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'reports/doc2query_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Full doc2query-T5 comparison','','| Method | NDCG@10 | Full NDCG | Exact at 1 |','| --- | --- | --- | --- |']
    for v,res in summary['results'].items():lines.append(f"| {v} | {res['ndcg_at_10']['mean']:.4f} | {res['ndcg_full']['mean']:.4f} | {res['exact_at_1']['mean']:.4f} |")
    lines+=['',summary['note'],'','All 8,956 queries and 181,701 candidates evaluated. Candidate ranking does not establish whole-catalog recall or online latency. Local inference adds no API charges; energy and hardware costs remain unmeasured. Paired bootstrap intervals use 1,000 query resamples and are exploratory.','']
    (ROOT/'reports/doc2query_evaluation.md').write_text('\n'.join(lines))
    print(json.dumps(summary['results'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generation',default='outputs/doc2query_t5/full_b16');main(p.parse_args())
