"""Independently verify complete candidate coverage and NDCG from saved rankings."""
import json
import hashlib
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    c=duckdb.connect();c.execute('SET threads=1');c.execute("SET memory_limit='1GB'")
    c.execute(f"CREATE VIEW truth AS SELECT query_id,product_id,esci_label, CASE esci_label WHEN 'E' THEN 1.0 WHEN 'S' THEN 0.1 WHEN 'C' THEN 0.01 ELSE 0.0 END gain FROM read_parquet('{ROOT}/data/processed/esci_us_small/test.parquet')")
    c.execute('CREATE TABLE ideal AS SELECT query_id,sum(CASE WHEN ideal_rank<=10 THEN gain/log2(ideal_rank+1) ELSE 0 END) idcg10,sum(gain/log2(ideal_rank+1)) idcgfull FROM (SELECT *,row_number() OVER (PARTITION BY query_id ORDER BY gain DESC,product_id) AS ideal_rank FROM truth) GROUP BY query_id')
    sources=[('plain',ROOT/'outputs/enrichment_evaluation/plain_rankings.csv',ROOT/'reports/enrichment_evaluation.json','plain'),
             ('structured',ROOT/'outputs/enrichment_evaluation/all_rankings.csv',ROOT/'reports/enrichment_evaluation.json','all')]
    for v in ('dense','hybrid_plain','hybrid_structured'):
        sources.append((v,ROOT/f'outputs/neural_evaluation/{v}_rankings.csv',ROOT/'reports/neural_evaluation.json',v))
    sources.append(('doc2query_t5',ROOT/'outputs/doc2query_evaluation/doc2query_t5_rankings.csv',ROOT/'reports/doc2query_evaluation.json','doc2query_t5'))
    for v in ('ling_doc2query','splade'):
        sources.append((v,ROOT/f'outputs/final_benchmark/{v}/rankings.csv',ROOT/f'outputs/final_benchmark/{v}/quality.json',v))
    results={}
    for name,path,summary_path,key in sources:
        if not summary_path.exists():continue
        if not path.exists():raise ValueError('Completed evaluation has missing rankings')
        c.execute(f"CREATE OR REPLACE VIEW ranking AS SELECT query_id,product_id,rank FROM read_csv_auto('{path}',types={{'product_id':'VARCHAR'}})")
        rows,queries=c.execute('SELECT count(*),count(DISTINCT query_id) FROM ranking').fetchone()
        missing=c.execute('SELECT count(*) FROM (SELECT query_id,product_id FROM truth EXCEPT SELECT query_id,product_id FROM ranking)').fetchone()[0]
        extra=c.execute('SELECT count(*) FROM (SELECT query_id,product_id FROM ranking EXCEPT SELECT query_id,product_id FROM truth)').fetchone()[0]
        duplicates=c.execute('SELECT count(*) FROM (SELECT query_id,product_id,count(*) n FROM ranking GROUP BY ALL HAVING n<>1)').fetchone()[0]
        invalid_ranks=c.execute('SELECT count(*) FROM (SELECT query_id,min(rank) first_rank,max(rank) last_rank,count(*) n,count(DISTINCT rank) unique_ranks FROM ranking GROUP BY query_id HAVING first_rank<>1 OR last_rank<>n OR unique_ranks<>n)').fetchone()[0]
        metrics=c.execute('SELECT avg(dcg10/idcg10),avg(dcgfull/idcgfull) FROM (SELECT r.query_id,sum(CASE WHEN rank<=10 THEN gain/log2(rank+1) ELSE 0 END) dcg10,sum(gain/log2(rank+1)) dcgfull FROM ranking r JOIN truth t USING(query_id,product_id) GROUP BY r.query_id) JOIN ideal USING(query_id)').fetchone()
        expected=json.loads(summary_path.read_text())['results'][key]
        discrepancies={metric:abs(value-expected[metric]['mean']) for metric,value in zip(('ndcg_at_10','ndcg_full'),metrics)}
        record={'rows':rows,'queries':queries,'missing':missing,'extra':extra,'duplicates':duplicates,'invalid_ranks':invalid_ranks,
                'independent_sql_ndcg_at_10':metrics[0],'independent_sql_ndcg_full':metrics[1],'absolute_metric_discrepancies':discrepancies,'rankings_sha256':sha(path)}
        record['passed']=(rows==181701 and queries==8956 and missing==extra==duplicates==invalid_ranks==0 and max(discrepancies.values())<1e-12)
        if not record['passed']:raise ValueError(f'Independent validation failed: {name}: {record}')
        results[name]=record
    report={'validated_methods':results,'method':'Independent DuckDB SQL over saved ranks and original ESCI labels; direct gains and full-candidate ideal ranking.','all_present_methods_passed':True}
    target='ranking_sql_validation_final.json' if 'splade' in results or 'ling_doc2query' in results else 'ranking_sql_validation.json'
    (ROOT/'reports'/target).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
