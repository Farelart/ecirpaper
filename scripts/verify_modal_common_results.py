"""Independent full-coverage SQL metrics and timing checks for new Modal results."""
import csv,hashlib,json,math,statistics
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parents[1]

def quantile(a,p):
    a=sorted(a);pos=(len(a)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
    return a[lo]+(a[hi]-a[lo])*(pos-lo)

def main():
    base=ROOT/'outputs/modal_common_t4/checkpoints/results'
    summary=json.loads((base/'summary.json').read_text());assert summary['complete']
    queries={json.loads(l)['query_id'] for l in (ROOT/'outputs/neural_inputs/queries.jsonl').open()}
    c=duckdb.connect();c.execute('SET threads=1')
    c.execute(f"CREATE VIEW truth AS SELECT query_id,product_id,CASE esci_label WHEN 'E' THEN 1.0 WHEN 'S' THEN 0.1 WHEN 'C' THEN 0.01 ELSE 0.0 END gain FROM read_parquet('{ROOT}/data/processed/esci_us_small/test.parquet')")
    c.execute('CREATE TABLE ideal AS SELECT query_id,sum(CASE WHEN pos<=10 THEN gain/log2(pos+1) ELSE 0 END) idcg,sum(gain/log2(pos+1)) idcg_full FROM (SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY gain DESC,product_id) pos FROM truth) GROUP BY query_id')
    records={}
    for name,timing in summary['methods'].items():
        ranking=base/name/'rankings.csv';latencies=base/name/'latencies.csv'
        c.execute(f"CREATE OR REPLACE VIEW ranking AS SELECT query_id,product_id,rank FROM read_csv_auto('{ranking}',types={{'product_id':'VARCHAR'}})")
        rows,nqueries=c.execute('SELECT count(*),count(DISTINCT query_id) FROM ranking').fetchone()
        missing=c.execute('SELECT count(*) FROM (SELECT query_id,product_id FROM truth EXCEPT SELECT query_id,product_id FROM ranking)').fetchone()[0]
        extra=c.execute('SELECT count(*) FROM (SELECT query_id,product_id FROM ranking EXCEPT SELECT query_id,product_id FROM truth)').fetchone()[0]
        duplicates=c.execute('SELECT count(*) FROM (SELECT query_id,product_id,count(*) n FROM ranking GROUP BY ALL HAVING n<>1)').fetchone()[0]
        bad=c.execute('SELECT count(*) FROM (SELECT query_id,min(rank) first_rank,max(rank) last_rank,count(*) n,count(DISTINCT rank) nr FROM ranking GROUP BY query_id HAVING first_rank<>1 OR last_rank<>n OR nr<>n)').fetchone()[0]
        ndcg,ndcg_full=c.execute('SELECT avg(dcg/idcg),avg(dcg_full/idcg_full) FROM (SELECT query_id,sum(CASE WHEN rank<=10 THEN gain/log2(rank+1) ELSE 0 END) dcg,sum(gain/log2(rank+1)) dcg_full FROM ranking JOIN truth USING(query_id,product_id) GROUP BY query_id) JOIN ideal USING(query_id)').fetchone()
        expected=json.loads((base/name/'quality.json').read_text())['results'][name]
        assert rows==181701 and nqueries==8956 and missing==extra==duplicates==bad==0
        assert abs(ndcg-expected['ndcg_at_10']['mean'])<1e-12
        assert abs(ndcg_full-expected['ndcg_full']['mean'])<1e-12
        ts=list(csv.DictReader(latencies.open()));assert len(ts)==26868
        assert {(int(r['repeat']),int(r['query_id'])) for r in ts}=={(rep,q) for rep in range(3) for q in queries}
        for r in ts:
            a,b,d=[float(r[k]) for k in ('total_seconds','encoding_seconds','search_and_top10_seconds')]
            assert math.isfinite(a) and math.isfinite(b) and math.isfinite(d) and a>0 and b>=0 and d>=0
            assert abs(a-b-d)<1e-12
        ms=[float(r['total_seconds'])*1000 for r in ts]
        assert abs(statistics.median(ms)-timing['median_ms'])<1e-8
        assert abs(quantile(ms,.95)-timing['p95_ms'])<1e-8
        assert timing['hardware']['gpu_class']=='T4'
        records[name]={'passed':True,'rows':rows,'queries':nqueries,'searches':len(ts),
                       'independent_ndcg_at_10':ndcg,'independent_ndcg_full':ndcg_full,'median_ms':statistics.median(ms),'p95_ms':quantile(ms,.95),
                       'rankings_sha256':hashlib.sha256(ranking.read_bytes()).hexdigest(),
                       'timings_sha256':hashlib.sha256(latencies.read_bytes()).hexdigest()}
    assert len(records)==8
    (ROOT/'reports/modal_common_t4_validation.json').write_text(json.dumps({'all_passed':True,'methods':records},indent=2)+'\n')
    print('All8 Modal methods independently verified',flush=True)

if __name__=='__main__':main()
