"""Export fixed test metadata and judgments; no model training or data sampling."""
import json
import hashlib
import html
import sys
from pathlib import Path
import duckdb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ecirpaper.retrieval import FIELDS, TAG, document_text

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()

def main():
    data = ROOT / 'data/processed/esci_us_small'
    out = ROOT / 'outputs/neural_inputs'
    if out.exists():
        if (out / 'manifest.json').exists():
            m = json.loads((out / 'manifest.json').read_text())
            if m['test_sha256'] != digest(data / 'test.parquet'): raise ValueError('Test changed')
            for name, sha in m['files'].items():
                if digest(out / name) != sha: raise ValueError('Export changed')
            print('Validated existing metadata export', flush=True); return
        raise ValueError('Incomplete export directory; preserve and inspect it before rebuilding')
    out.mkdir(parents=True)
    c = duckdb.connect(); c.execute('SET threads=1')
    c.execute(f"CREATE VIEW test AS SELECT * FROM read_parquet('{data}/test.parquet')")
    cursor = c.execute(f"SELECT p.product_id,{','.join('p.' + f for f in FIELDS)} FROM read_parquet('{data}/products.parquet') p SEMI JOIN (SELECT DISTINCT product_id FROM test) t USING(product_id) ORDER BY product_id")
    n = 0
    with (out / 'products.jsonl').open('w') as stream:
        while batch := cursor.fetchmany(256):
            for row in batch:
                text = TAG.sub(' ', html.unescape(document_text(row[1:])))
                stream.write(json.dumps({'product_id': row[0], 'text': text}, ensure_ascii=False) + '\n'); n += 1
    cursor = c.execute('SELECT query_id,query,source,product_id,esci_label FROM test ORDER BY query_id,product_id')
    group = None; queries = 0; pairs = 0
    with (out / 'queries.jsonl').open('w') as stream:
        while batch := cursor.fetchmany(256):
            for qid, query, source, pid, label in batch:
                if group is not None and qid != group['query_id']:
                    stream.write(json.dumps(group, ensure_ascii=False) + '\n'); queries += 1; group = None
                if group is None: group = {'query_id': qid, 'query': query, 'source': source, 'judgments': {}}
                if pid in group['judgments']: raise ValueError('Duplicate judgment')
                group['judgments'][pid] = label; pairs += 1
        if group is not None: stream.write(json.dumps(group, ensure_ascii=False) + '\n'); queries += 1
    if (n, queries, pairs) != (164900, 8956, 181701): raise ValueError('Coverage mismatch')
    manifest = {'products': n, 'queries': queries, 'pairs': pairs, 'fields': FIELDS,
                'text_preparation': 'Full original metadata; same HTML cleanup as BM25, case preserved for encoder.',
                'test_sha256': digest(data / 'test.parquet'), 'products_parquet_sha256': digest(data / 'products.parquet'),
                'files': {name: digest(out / name) for name in ('products.jsonl', 'queries.jsonl')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__': main()
