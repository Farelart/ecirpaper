"""Validate completed timing records and synthesize independently checked results."""
import csv
import datetime
import hashlib
import json
import math
import random
import statistics
from decimal import Decimal
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def quantile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    verified = read('reports/ranking_sql_validation_final.json')
    assert verified['all_present_methods_passed']
    assert len(verified['validated_methods']) == 8
    timings = read('reports/retrieval_benchmark.json')
    query_ids = {json.loads(line)['query_id'] for line in
                 (ROOT / 'outputs/neural_inputs/queries.jsonl').open()}
    validation = {}
    for name, summary in timings.items():
        path = ROOT / f'outputs/final_benchmark/{name}/latencies.csv'
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 26868
        keys = {(int(r['repeat']), int(r['query_id'])) for r in rows}
        assert keys == {(rep, qid) for rep in range(3) for qid in query_ids}
        for r in rows:
            total, encoding, search = [float(r[k]) for k in
                                      ('total_seconds', 'encoding_seconds', 'search_and_top10_seconds')]
            assert all(math.isfinite(v) for v in (total, encoding, search))
            assert total > 0 and encoding >= 0 and search >= 0
            assert abs(total - encoding - search) < 1e-12
        ms = [float(r['total_seconds']) * 1000 for r in rows]
        assert abs(statistics.median(ms) - summary['median_ms']) < 1e-8
        assert abs(quantile(ms, .95) - summary['p95_ms']) < 1e-8
        validation[name] = {'passed': True, 'searches': len(rows),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    c = duckdb.connect()
    c.execute('SET threads=1')
    c.execute("SET memory_limit='1GB'")
    c.execute(f"""CREATE VIEW truth AS SELECT query_id,product_id,
      CASE esci_label WHEN 'E' THEN 1.0 WHEN 'S' THEN 0.1 WHEN 'C' THEN 0.01 ELSE 0.0 END gain
      FROM read_parquet('{ROOT}/data/processed/esci_us_small/test.parquet')""")
    c.execute("""CREATE TABLE ideal AS SELECT query_id,
      sum(CASE WHEN pos<=10 THEN gain/log2(pos+1) ELSE 0 END) idcg
      FROM (SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY gain DESC,product_id) pos
      FROM truth) GROUP BY query_id""")
    paths = {
        'plain': 'outputs/enrichment_evaluation/plain_rankings.csv',
        'structured': 'outputs/enrichment_evaluation/all_rankings.csv',
        'doc2query_t5': 'outputs/doc2query_evaluation/doc2query_t5_rankings.csv',
        'dense': 'outputs/neural_evaluation/dense_rankings.csv',
        'hybrid_plain': 'outputs/neural_evaluation/hybrid_plain_rankings.csv',
        'hybrid_structured': 'outputs/neural_evaluation/hybrid_structured_rankings.csv',
        'ling_doc2query': 'outputs/final_benchmark/ling_doc2query/rankings.csv',
        'splade': 'outputs/final_benchmark/splade/rankings.csv',
    }
    per_query = {}
    for name, path in paths.items():
        data = c.execute(f"""SELECT query_id,dcg/idcg FROM
          (SELECT r.query_id,sum(CASE WHEN rank<=10 THEN gain/log2(rank+1) ELSE 0 END) dcg
          FROM read_csv_auto('{ROOT/path}',types={{'product_id':'VARCHAR'}}) r
          JOIN truth USING(query_id,product_id) GROUP BY r.query_id)
          JOIN ideal USING(query_id) ORDER BY query_id""").fetchall()
        assert {r[0] for r in data} == query_ids
        per_query[name] = dict(data)
        assert abs(statistics.fmean(per_query[name].values()) -
                   verified['validated_methods'][name]['independent_sql_ndcg_at_10']) < 1e-12
    comparisons = {}
    for a, b in [('structured', 'plain'), ('structured', 'doc2query_t5'),
                 ('structured', 'ling_doc2query'), ('dense', 'structured'),
                 ('splade', 'dense')]:
        diff = [per_query[a][qid] - per_query[b][qid] for qid in sorted(query_ids)]
        rng = random.Random(42)
        boot = [statistics.fmean(rng.choices(diff, k=len(diff))) for _ in range(1000)]
        comparisons[f'{a}_minus_{b}'] = {
            'mean_difference': statistics.fmean(diff),
            'paired_95_percent_bootstrap_interval': [quantile(boot, .025), quantile(boot, .975)],
            'resamples': 1000, 'seed': 42,
            'note': 'Exploratory query bootstrap; no multiple-comparison correction. Standard-library RNG.'}
    (ROOT / 'reports/final_paired_comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    (ROOT / 'reports/final_per_query_ndcg.json').write_text(json.dumps(per_query)+'\n')

    billing_rows = read('outputs/splade/billing_report_latest.json')
    app_totals = {}
    for row in billing_rows:
        app = row['object_id']
        if not app.startswith('ap-'):
            continue
        item = app_totals.setdefault(app, {'description': row['description'],
                                         'compute_usd': Decimal(0), 'including_egress_usd': Decimal(0)})
        item['including_egress_usd'] += Decimal(row['cost'])
        if row['resource'] in ('T4', 'L4', 'CPU', 'Memory'):
            item['compute_usd'] += Decimal(row['cost'])
    app_totals = {app: {k: str(v) if isinstance(v, Decimal) else v for k,v in item.items()}
                  for app,item in app_totals.items()}
    experiment_apps = {app: item for app,item in app_totals.items()
                       if item['description'].startswith('ecir-')}
    total = sum((Decimal(item['including_egress_usd']) for item in experiment_apps.values()), Decimal(0))
    assert total < 30
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    final = {
        'verified_at_utc': now, 'complete': True,
        'products': 164900, 'queries': 8956, 'judged_pairs': 181701,
        'quality_validation': verified, 'timing_validation': validation,
        'timings': timings, 'paired_comparisons': comparisons,
        'preparation': {
            'structured_logged_api_usd': 1.4240369976,
            'ling_queries_logged_api_usd': .9950095764,
            'shared_api_ledger_including_reservations_usd': '2.42493314280',
            'api_ceiling_usd': 4.4,
            'structured_total_elapsed_seconds': None,
            'ling_queries_total_elapsed_seconds': None,
            't5_cloud': read('reports/modal_t5_full_verified.json'),
            'splade': read('reports/splade_encoding_validation.json'),
            'e5_product_inference_seconds': 2191.880882013,
            'e5_compute_cost_usd': None,
            'note': 'API charges and metered cloud resources describe different deployment stacks. API full elapsed times and local E5 monetary compute cost were not measured.'},
        'modal_billing': {'checked_at_utc': now, 'experiment_app_totals': experiment_apps,
                          'all_reported_experiment_apps_including_failed_attempts_usd': str(total),
                          'monthly_workspace_summary': read('outputs/splade/billing_summary_latest.json'),
                          'note': 'Provider-reported usage snapshot, subject to later adjustments. App totals include reported pilots/failed attempts; absent charges are not proof of zero. Workspace credits/payable differ from metered costs.'},
        'limitations': [
            'Quality is official judged-candidate ranking; timing is exact full-corpus top10. No whole-catalog recall evaluation.',
            'Timing uses one local CPU thread, warm resident indexes, no network/ANN. Three shuffled passes; no uncertainty analysis across independent hardware runs.',
            'Neural document encoding occurs separately on different hardware; preparation boundaries are not identical.',
            'Hybrids use full-candidate RRF for quality, but top1000 per component for timing; these are different retrieval protocols.',
            'Serialized index sizes exclude common product-ID mapping, metadata, model weights and resident-memory overhead.',
            'Structured/T5/Ling generation uses shortened metadata; E5/SPLADE use full metadata capped at512tokens.',
            'T5 keeps prior local effectiveness outputs; cloud rerun measures preparation, not replacement rankings.',
            'No ESCI training/tuning. Three T5 queries per product is not a reproduction of the original40-query setup.'],
    }
    (ROOT/'reports/final_experiments_verified.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({'timing_methods_verified':len(validation), 'paired_comparisons': comparisons,
                      'modal_experiment_apps_reported_usd':str(total)},indent=2))


if __name__ == '__main__':
    main()
