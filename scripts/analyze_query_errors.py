"""Post-hoc descriptive analysis of frozen test rankings; no tuning or inference."""
import csv,json,math,random,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports'
METHODS=['plain','structured','ling_doc2query','doc2query_t5','e5','splade','hybrid_plain','hybrid_structured']
G={'E':1.0,'S':.1,'C':.01,'I':0.0}
queries=list(map(json.loads,(ROOT/'outputs/neural_inputs/queries.jsonl').open()));qmap={r['query_id']:r for r in queries};ids=sorted(qmap)
rankings={};metrics={}
for method in METHODS:
    ranks={}
    for r in csv.DictReader((ROOT/'outputs/modal_common_t4/checkpoints/results'/method/'rankings.csv').open()):ranks.setdefault(int(r['query_id']),[]).append((int(r['rank']),r['product_id']))
    assert set(ranks)==set(qmap);rankings[method]={q:[p for _,p in sorted(v)] for q,v in ranks.items()};metrics[method]={}
    for q in ids:
        j=qmap[q]['judgments'];order=rankings[method][q];assert len(order)==len(set(order))==len(j) and set(order)==set(j)
        den=sum(g/math.log2(i+2) for i,g in enumerate(sorted([G[v] for v in j.values()],reverse=True)[:10]))
        metrics[method][q]=sum(G[j[p]]/math.log2(i+2) for i,p in enumerate(order[:10]))/den if den else 0
comparisons=[]
for left,right in [('structured','plain'),('structured','doc2query_t5'),('structured','ling_doc2query'),('e5','structured'),('hybrid_structured','e5'),('hybrid_plain','e5')]:
    d=[metrics[left][q]-metrics[right][q] for q in ids];rng=random.Random(42);b=sorted(statistics.mean(rng.choices(d,k=len(d))) for _ in range(1000))
    comparisons.append({'left':left,'right':right,'mean_difference':statistics.mean(d),'improved_queries':sum(x>1e-12 for x in d),'worsened_queries':sum(x< -1e-12 for x in d),'unchanged_queries':sum(abs(x)<=1e-12 for x in d),'exploratory_95_percent_interval':[b[24],b[974]],'bootstrap_resamples':1000,'seed':42,'multiple_comparison_adjustment':False})
strata=[]
for kind in ['source','query_length']:
    groups={}
    for q in ids:
        r=qmap[q];n=len(re.findall(r'[^\W_]+',r['query']))
        label=r['source'] if kind=='source' else '1' if n==1 else '2' if n==2 else '3-5' if n<=5 else '6+'
        groups.setdefault(label,[]).append(q)
    for label,selected in sorted(groups.items()):
        values={m:statistics.mean(metrics[m][q] for q in selected) for m in METHODS}
        strata.append({'grouping':kind,'group':label,'queries':len(selected),'ndcg_at_10':values,'structured_minus_plain':values['structured']-values['plain'],'structured_minus_ling':values['structured']-values['ling_doc2query'],'hybrid_structured_minus_e5':values['hybrid_structured']-values['e5']})
cases=[]
for left,right in [('structured','plain'),('structured','ling_doc2query'),('hybrid_structured','e5')]:
    sorted_ids=sorted(ids,key=lambda q:(metrics[left][q]-metrics[right][q],q))
    for direction,selected in [('largest_losses',sorted_ids[:5]),('largest_gains',list(reversed(sorted_ids[-5:]))),('seeded_random',random.Random(42).sample(ids,5))]:
        for q in selected:
            r=qmap[q];case={'comparison':[left,right],'selection':direction,'query_id':q,'query':r['query'],'source':r['source'],'delta':metrics[left][q]-metrics[right][q],'ndcg':{m:metrics[m][q] for m in METHODS},'top5':{m:[{'product_id':p,'label':r['judgments'][p]} for p in rankings[m][q][:5]] for m in METHODS}}
            cases.append(case)
selected_pids={p['product_id'] for case in cases for ranks in case['top5'].values() for p in ranks}
# Real product titles from prepared parquet, not inferred from concatenated metadata.
import subprocess
script="import duckdb,json; c=duckdb.connect(); print(json.dumps(dict(c.execute(\"SELECT product_id,product_title FROM read_parquet('data/processed/esci_us_small/products.parquet')\").fetchall())))"
result=subprocess.run([str(ROOT/'.venv/bin/python'),'-c',script],cwd=ROOT,capture_output=True,text=True,check=True);titles=json.loads(result.stdout)
expanded={r['product_id']:r['enrichment'] for r in map(json.loads,(ROOT/'outputs/budget_enrichment/test/products_enriched.jsonl').open()) if r['product_id'] in selected_pids}
for case in cases:
    for ranks in case['top5'].values():
        for p in ranks:p['title']=titles[p['product_id']];p['structured_enrichment']=expanded.get(p['product_id'])
report={'scope':'Descriptive post-hoc analysis of frozen official judged-candidate test rankings; not whole-catalog retrieval, no training/tuning','queries':len(ids),'comparisons':comparisons,'strata':strata,'examples':cases,'example_selection':'Five largest losses, five largest gains and five seeded random queries per comparison; extremes are illustrative, not representative','limitations':['Source is dataset metadata, not verified linguistic intent','No causal attribution from aggregate strata or rank changes alone','Exploratory bootstrap without multiple-comparison adjustment','No first-stage full-catalog recall measured']}
(OUT/'query_error_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
lines=['# Query-level analysis of frozen ESCI results','',report['scope']+'.','', '| Comparison | Mean NDCG difference | Improved / worsened / unchanged queries | Exploratory95% interval |','|---|---:|---:|---:|']
for c in comparisons:lines.append(f"| {c['left']} minus {c['right']} | {c['mean_difference']:+.5f} | {c['improved_queries']} / {c['worsened_queries']} / {c['unchanged_queries']} | [{c['exploratory_95_percent_interval'][0]:+.5f}, {c['exploratory_95_percent_interval'][1]:+.5f}] |")
lines+=['','Intervals use1,000 paired query resamples, seed42; exploratory, no multiple-comparison correction.','', '| Grouping | Group | Queries | Plain | Structured | Ling queries | E5 | Structured+E5 |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in strata:
    v=r['ndcg_at_10'];lines.append(f"| {r['grouping']} | {r['group']} | {r['queries']} | {v['plain']:.4f} | {v['structured']:.4f} | {v['ling_doc2query']:.4f} | {v['e5']:.4f} | {v['hybrid_structured']:.4f} |")
lines+=['','Dataset source names are metadata, not independently verified query types. Small strata should not support general claims.','', '## Examples for manual inspection','', 'These deliberately selected extremes show possible failure modes; the JSON also includes seeded random examples. Product labels: E exact, S substitute, C complement, I irrelevant. Expansion text and all eight top-five rankings are preserved in query_error_analysis.json.']
for c in cases:
    if c['selection']=='seeded_random':continue
    left,right=c['comparison'];lines+=['',f"### {left} minus {right}: {c['selection']}, query{c['query_id']}",f"Query: {c['query']}",f"NDCG difference: {c['delta']:+.4f}"]
    for m in (right,left):
        first=c['top5'][m][0];lines.append(f"- {m} rank1: [{first['label']}] {first['title']} ({first['product_id']})")
(OUT/'query_error_analysis.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'comparisons':comparisons,'strata':strata},indent=2))
