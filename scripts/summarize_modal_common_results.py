"""Produce the verified common-T4 report and paired candidate-quality analysis."""
import datetime,json,math,random,statistics
from decimal import Decimal
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parents[1]

def quantile(a,p):
    a=sorted(a);pos=(len(a)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
    return a[lo]+(a[hi]-a[lo])*(pos-lo)

def main():
    r=ROOT/'reports';out=ROOT/'outputs/modal_common_t4';base=out/'checkpoints/results'
    check=json.loads((r/'modal_common_t4_validation.json').read_text());assert check['all_passed']
    summary=json.loads((base/'summary.json').read_text());assert summary['complete']
    e5=json.loads((r/'e5_t4_encoding_validation.json').read_text());assert e5['passed']
    splade=json.loads((r/'splade_encoding_validation.json').read_text())
    t5=json.loads((r/'modal_t5_full_verified.json').read_text())
    generation=json.loads((r/'generation_timing_recovered.json').read_text())
    c=duckdb.connect();c.execute('SET threads=1')
    c.execute(f"CREATE VIEW truth AS SELECT query_id,product_id,CASE esci_label WHEN 'E' THEN 1.0 WHEN 'S' THEN 0.1 WHEN 'C' THEN 0.01 ELSE 0.0 END gain FROM read_parquet('{ROOT}/data/processed/esci_us_small/test.parquet')")
    c.execute('CREATE TABLE ideal AS SELECT query_id,sum(CASE WHEN pos<=10 THEN gain/log2(pos+1) ELSE 0 END) idcg FROM (SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY gain DESC,product_id) pos FROM truth) GROUP BY query_id')
    metrics={};quality={}
    for name in summary['methods']:
        path=base/name/'rankings.csv'
        values=c.execute(f"SELECT query_id,dcg/idcg FROM (SELECT r.query_id,sum(CASE WHEN rank<=10 THEN gain/log2(rank+1) ELSE 0 END) dcg FROM read_csv_auto('{path}',types={{'product_id':'VARCHAR'}}) r JOIN truth USING(query_id,product_id) GROUP BY r.query_id) JOIN ideal USING(query_id) ORDER BY query_id").fetchall()
        assert len(values)==8956;metrics[name]=dict(values)
        assert abs(statistics.fmean(metrics[name].values())-check['methods'][name]['independent_ndcg_at_10'])<1e-12
        quality[name]=json.loads((base/name/'quality.json').read_text())['results'][name]
    pairs={}
    for a,b in [('structured','plain'),('structured','doc2query_t5'),('structured','ling_doc2query'),('e5','structured'),('splade','e5')]:
        d=[metrics[a][q]-metrics[b][q] for q in sorted(metrics[a])];rng=random.Random(42)
        samples=[statistics.fmean(rng.choices(d,k=len(d))) for _ in range(1000)]
        pairs[a+'_minus_'+b]={'mean_difference':statistics.fmean(d),'paired_95_percent_interval':[quantile(samples,.025),quantile(samples,.975)],'resamples':1000,'seed':42,'note':'Exploratory query bootstrap, standard-library RNG, no multiple-comparison correction.'}
    rows=json.loads((out/'billing_report.json').read_text());app_costs={}
    for row in rows:
        if not row['object_id'].startswith('ap-') or not row['description'].startswith('ecir-'):continue
        item=app_costs.setdefault(row['object_id'],{'description':row['description'],'compute_usd':Decimal(0),'including_egress_usd':Decimal(0)})
        item['including_egress_usd']+=Decimal(row['cost'])
        if row['resource'] in ('T4','L4','CPU','Memory'):item['compute_usd']+=Decimal(row['cost'])
    total=sum((v['including_egress_usd'] for v in app_costs.values()),Decimal(0));assert total<30
    app_costs={k:{a:str(b) if isinstance(b,Decimal) else b for a,b in v.items()} for k,v in app_costs.items()}
    prep={'plain':0,'structured':generation['methods']['structured']['observed_generation_span_seconds'],
          'ling_doc2query':generation['methods']['ling_doc2query']['observed_generation_span_seconds'],
          'doc2query_t5':t5['batch_processing_seconds'],'e5':e5['coverage']['products']['processing_seconds'],
          'splade':splade['processing_seconds']['products']}
    prep['hybrid_plain']=prep['e5'];prep['hybrid_structured']=prep['e5']+prep['structured']
    timing_table={}
    for name,t in summary['methods'].items():
        build=t['index'].get('total_indexing_seconds',t['index']['build_and_write_seconds'])
        timing_table[name]={'preparation_seconds':prep[name],'indexing_seconds':build,
            'combined_recorded_phase_seconds':prep[name]+build,'index_bytes':t['index']['index_bytes'],
            'median_search_ms':t['median_ms'],'p95_search_ms':t['p95_ms'],
            'ndcg_at_10':check['methods'][name]['independent_ndcg_at_10']}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report={'complete':True,'verified_at_utc':now,'products':164900,'queries':8956,'judged_pairs':181701,
        'table':timing_table,'quality':quality,'paired_comparisons':pairs,'cloud_summary':summary,
        'independent_validation':check,'e5_encoding_validation':e5,
        'source_tokenization_audit':json.loads((r/'modal_common_t4_tokenization_audit.json').read_text()),
        'api_generation_timing':generation,
        'billing':{'checked_at_utc':now,'experiment_app_totals':app_costs,'reported_experiment_apps_total_usd':str(total),
            'workspace_monthly_summary':json.loads((out/'billing_summary.json').read_text()),
            'note':'Provider-reported snapshots, before credits, subject to later adjustments; failed-attempt/pilot charges retained. Workspace summary has broader scope; not attributed wholly to this experiment.'},
        'boundaries':['Neural preparation uses the same T4 resource class and CPU/memory allocation across distinct preparation jobs; retrieval methods all ran sequentially in one common T4 invocation.',
            'BM25 uses CPU postings; neural query encoding and exact dot products use CUDA; GPU-to-CPU score transfer and deterministic CPU top10 selection included. No network serving or ANN.',
            'Preparation sums recorded phases: Ling spans include gaps/recovery, neural batch timers exclude checkpoint/loading overhead. Hybrid sums are component sums, not measured concurrent elapsed time.',
            'Index construction includes lexical tokenization separately, or neural representation assembly/serialization/GPU transfer. Index sizes exclude shared metadata/IDs and model weights.',
            'Frozen lexical token counts preserve earlier scoring. Expansion-field order changes do not change BM25; one T5 tokenization runtime discrepancy is frozen rather than applied.',
            'Quality ranks judged candidates. Hybrid quality uses full-candidate RRF, timed hybrids use1000 hits per component with k60.',
            'Ling API generation is a separate deployment measure, not T4-hosted. Model/input/output allowance matched between the two Ling strategies, but observed concurrency differs.',
            'Shortened metadata for expansion versus full512token inputs for E5/SPLADE limits cross-family attribution. No ESCI fine-tuning, no universal cheapest-method or accuracy claim.']}
    (r/'modal_common_t4_verified.json').write_text(json.dumps(report,indent=2)+'\n')
    (r/'modal_common_t4_paired_comparisons.json').write_text(json.dumps(pairs,indent=2)+'\n')
    names={'plain':'Plain BM25','structured':'Structured Ling + BM25','doc2query_t5':'T5 queries + BM25',
           'ling_doc2query':'Ling queries + BM25','e5':'E5-small','splade':'SPLADE',
           'hybrid_plain':'Plain BM25 + E5','hybrid_structured':'Structured BM25 + E5'}
    e5_cost=app_costs['ap-0i1xiZ8m2lOJEYnfutdIdJ']['compute_usd']
    costs={'plain':'No generation; CPU cost unallocated','structured':'API $1.424037','ling_doc2query':'API $0.995010',
           'doc2query_t5':'T4 compute $1.607373','e5':'T4 compute $'+e5_cost,'splade':'T4 compute $1.522311',
           'hybrid_plain':'E5 component $'+e5_cost,'hybrid_structured':'API $1.424037 + E5 $'+e5_cost}
    lines=['# Verified common Modal T4 comparison','',
        'All eight configurations have independently verified complete candidate rankings and three retrieval timing passes over all8,956 queries. Corpus:164,900 products;181,701 judged pairs. No ESCI-specific training or tuning.','',
        'The primary comparison now uses T4 for all neural preparation and a single Modal T4 host for all retrieval methods. Earlier local results remain preserved. Ling generation remains an API measurement.','',
        '| Method | NDCG@10 | Preparation min | Index s | Combined min | Index MiB | Median/p95 ms | Preparation cost |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for n,t in timing_table.items():
        lines.append(f"| {names[n]} | {t['ndcg_at_10']:.4f} | {t['preparation_seconds']/60:.2f} | {t['indexing_seconds']:.2f} | {t['combined_recorded_phase_seconds']/60:.2f} | {t['index_bytes']/1048576:.2f} | {t['median_search_ms']:.2f}/{t['p95_search_ms']:.2f} | {costs[n]} |")
    lines+=['','## What the measurements support','',
        'Structured expansion improves plain BM25 and this three-query T5 configuration. The matched Ling generated-query control ranks slightly better and has lower logged API charges. E5 has higher quality than structured expansion, lower recorded preparation time and lower reported preparation compute cost in these deployments. Structured BM25 has lower retrieval latency. SPLADE has the highest mean quality, but its small advantage over E5 is not reliably separated by the exploratory paired interval.','',
        'The common-host medians narrow the structured/E5 search-time gap to approximately1.65x:4.75ms versus7.82ms. The previous approximately15x local CPU comparison must not be used as the primary deployment-speed claim. Small latency differences between BM25 variants should not be interpreted as a reliable ordering.','',
        '## Timing boundaries and deployment costs','']
    lines+=['- '+b for b in report['boundaries']]
    lines+=['','Model loading, checkpoint I/O, full setup/final export, storage and network serving charges are not included consistently in preparation phase sums. These columns therefore do not measure complete end-to-end pipeline wall time. API generation records do include pauses between first request and last reconstructed response. Do not compare them as intrinsic model throughput.','',
        f"Reported experiment-app usage, including all reported prior pilots and failed attempts: **${total}** including egress, below the$30 ceiling. E5 preparation app: compute${e5_cost}, total${app_costs['ap-0i1xiZ8m2lOJEYnfutdIdJ']['including_egress_usd']}. Common eight-method benchmark app: compute${app_costs['ap-KeEObtyO01vt5qW3ECOdld']['compute_usd']}, total${app_costs['ap-KeEObtyO01vt5qW3ECOdld']['including_egress_usd']}. The common benchmark bill covers all indexes, evaluations and timing passes; it is not a per-method serving-cost estimate.", '',
        'The workspace monthly summary reports$4.29770780 ephemeral-app usage and$0 payable after credits/allowances; it has broader scope than the experiment app list. API shared ledger remains$2.424933 of its unchanged$4.40 ceiling, including uncertain reservations. Earlier outside-ledger Gemini pilot records remain separate.','',
        '## Paired candidate-quality differences','']
    for n,p in pairs.items():
        lo,hi=p['paired_95_percent_interval'];lines.append(f"- {n}: mean ΔNDCG@10 {p['mean_difference']:+.8f};95% interval [{lo:+.8f},{hi:+.8f}].")
    lines+=['','Exploratory bootstrap:1,000 query resamples, seed42, standard-library RNG; no multiple-comparison correction.','',
        '## Validation','',
        'Independent SQL confirms every ranking has181,701 rows,8,956 queries, complete unique query-product coverage, valid ranks and matching NDCG. Each method has26,868 finite timing rows with complete query coverage per repetition; median/p95 independently recomputed.','',
        'All E5 checkpoint vectors independently verified:164,900 products,8,956 queries,384 dimensions,float32, finite values, unit L2 norms, pinned revision and identical input hashes to SPLADE. CUDA dense/sparse scores were independently checked against CPU dot products on21 full-corpus queries before timing. SPLADE checkpoints and pins had already passed complete coverage/numerical validation.','',
        'The downloader took longer than expected. A reserved SQL alias in the verification script was corrected; saved artifacts passed without any inference rerun. All earlier local outputs and cloud preparation records are preserved.','',
        'Sources:modal_common_t4_verified.json;modal_common_t4_validation.json;modal_common_t4_paired_comparisons.json;e5_t4_encoding_validation.json;modal_common_t4_tokenization_audit.json;generation_timing_recovered.json.']
    (r/'modal_common_t4_verified.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':True,'reported_experiment_app_cost_usd':str(total),'paired_comparisons':pairs,'table':timing_table},indent=2))

if __name__=='__main__':main()
