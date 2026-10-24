"""Build English ESCI small-version query/candidate/product datasets.

This is a reproducible metadata join, not model training. Relevance labels
remain untouched. Parquet row order is unspecified; grouped examples are
ordered by product ID, never by relevance.
"""
import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/esci"
DEFAULT_OUTPUT = ROOT / "data/processed/esci_us_small"
METADATA = ["product_title", "product_description", "product_bullet_point",
            "product_brand", "product_color"]


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose another --output-dir to rebuild.")
    provenance = json.loads((ROOT / "data/esci_manifest.json").read_text())
    for record in provenance["files"]:
        path = RAW / record["name"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Raw source failed checksum verification: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".esci-build-", dir=output.parent) as directory:
        temporary = Path(directory)
        stage = temporary / "dataset"
        stage.mkdir()
        con = duckdb.connect()
        con.execute("SET threads=1")
        con.execute("SET memory_limit='2GB'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET temp_directory=?", [str(temporary / "spill")])
        con.execute(f"""CREATE VIEW examples AS SELECT example_id, query_id, query,
            product_id, product_locale, esci_label, split
            FROM read_parquet({sql_path(RAW / 'shopping_queries_dataset_examples.parquet')})
            WHERE product_locale='us' AND small_version=1""")
        con.execute(f"CREATE VIEW raw_products AS SELECT * FROM read_parquet({sql_path(RAW / 'shopping_queries_dataset_products.parquet')})")
        con.execute(f"CREATE VIEW sources AS SELECT * FROM read_csv_auto({sql_path(RAW / 'shopping_queries_dataset_sources.csv')})")

        def records(sql):
            cursor = con.execute(sql)
            return [dict(zip([x[0] for x in cursor.description], row)) for row in cursor.fetchall()]

        def scalar(sql):
            return con.execute(sql).fetchone()[0]

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        expected = records("SELECT split, count(*) pairs, count(DISTINCT query_id) queries FROM examples GROUP BY split ORDER BY split")
        require({r["split"] for r in expected} == {"train", "test"}, "Unexpected split names")
        require(scalar("SELECT count(*) FROM examples WHERE esci_label IS NULL OR esci_label NOT IN ('E','S','C','I')") == 0, "Invalid relevance labels")
        require(scalar("SELECT count(*) FROM (SELECT query_id FROM examples GROUP BY query_id HAVING count(DISTINCT split)>1 OR count(DISTINCT query)>1)") == 0, "Query text/split inconsistency")
        require(scalar("SELECT count(*) FROM (SELECT query_id, product_id, product_locale FROM examples GROUP BY ALL HAVING count(*)>1)") == 0, "Duplicate query-product pairs")
        require(scalar("SELECT count(*) FROM (SELECT query_id FROM sources GROUP BY query_id HAVING count(*)>1)") == 0, "Duplicate source keys")
        require(scalar("SELECT count(*) FROM examples e ANTI JOIN raw_products p USING(product_id,product_locale)") == 0, "Missing product metadata")
        require(scalar("SELECT count(*) FROM (SELECT product_id,product_locale FROM raw_products GROUP BY ALL HAVING count(*)>1)") == 0, "Duplicate product metadata keys")

        print("Writing unique product metadata", flush=True)
        columns = ", ".join("p." + name for name in ["product_id", "product_locale"] + METADATA)
        con.execute(f"""COPY (
            SELECT {columns} FROM raw_products p
            SEMI JOIN (SELECT DISTINCT product_id, product_locale FROM examples) e
            USING(product_id,product_locale)
            ORDER BY p.product_locale, p.product_id
        ) TO {sql_path(stage / 'products.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        con.execute(f"CREATE VIEW products AS SELECT * FROM read_parquet({sql_path(stage / 'products.parquet')})")
        con.execute("""CREATE VIEW queries AS
            SELECT e.query_id, e.query, e.product_locale, e.split, s.source, count(*) candidate_count
            FROM examples e LEFT JOIN sources s USING(query_id)
            GROUP BY ALL""")
        require(scalar("SELECT count(*) FROM queries WHERE source IS NULL") == 0, "Missing query sources")
        con.execute(f"COPY (SELECT * FROM queries ORDER BY query_id) TO {sql_path(stage / 'queries.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
        metadata = ", ".join("p." + name for name in METADATA)
        con.execute(f"""CREATE VIEW joined AS SELECT e.*, s.source, {metadata}
            FROM examples e JOIN products p USING(product_id,product_locale)
            JOIN sources s USING(query_id)""")
        require(scalar("SELECT count(*) FROM joined") == sum(r["pairs"] for r in expected), "Join changed pair count")

        for split in ["train", "test"]:
            print(f"Writing {split} query-product pairs", flush=True)
            # Sorting full descriptions and bullet points retains too much
            # text in memory. Export the relation in streaming batches; clients
            # group by query_id and explicitly sort candidates as needed.
            con.execute(f"COPY (SELECT * FROM joined WHERE split='{split}') TO {sql_path(stage / (split + '.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 8192)")
            file = sql_path(stage / (split + ".parquet"))
            stats = records(f"SELECT count(*) pairs,count(DISTINCT query_id) queries FROM read_parquet({file})")[0]
            source_stats = next(r for r in expected if r["split"] == split)
            require(stats["pairs"] == source_stats["pairs"] and stats["queries"] == source_stats["queries"], "Saved split count mismatch")
            require(scalar(f"""SELECT count(*) FROM (
                (SELECT example_id,query_id,product_id,product_locale,esci_label,split FROM examples WHERE split='{split}'
                 EXCEPT ALL SELECT example_id,query_id,product_id,product_locale,esci_label,split FROM read_parquet({file}))
                UNION ALL
                (SELECT example_id,query_id,product_id,product_locale,esci_label,split FROM read_parquet({file})
                 EXCEPT ALL SELECT example_id,query_id,product_id,product_locale,esci_label,split FROM examples WHERE split='{split}')
            )""") == 0, "Saved pairs differ from official labels or keys")

        product_count = scalar("SELECT count(*) FROM products")
        query_count = scalar("SELECT count(*) FROM queries")
        schema = records("DESCRIBE joined")
        label_counts = records("SELECT split,esci_label,count(*) pairs FROM joined GROUP BY ALL ORDER BY split,esci_label")
        candidate_stats = records("SELECT split,min(candidate_count) minimum,median(candidate_count) median,avg(candidate_count) mean,max(candidate_count) maximum FROM queries GROUP BY split ORDER BY split")
        samples = []
        sample_ids = records("SELECT query_id FROM queries WHERE split='train' ORDER BY hash(query_id) LIMIT 3")
        for item in sample_ids:
            query_id = item["query_id"]
            query = records(f"SELECT * FROM queries WHERE query_id={query_id}")[0]
            query["candidates"] = records(f"SELECT product_id,product_locale,esci_label,{','.join(METADATA)} FROM joined WHERE query_id={query_id} ORDER BY product_id")
            samples.append(query)
        con.close()
        files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(stage.glob("*.parquet"))]
        manifest = {"dataset": "Amazon ESCI English small-version candidate ranking",
                    "selection": {"product_locale": "us", "small_version": 1},
                    "source_revision": provenance["revision"], "source_files": provenance["files"],
                    "built_at_utc": datetime.now(timezone.utc).isoformat(), "duckdb_version": duckdb.__version__,
                    "splits": expected, "products": product_count, "queries": query_count,
                    "joined_schema": schema, "label_counts": label_counts,
                    "candidates_per_query": candidate_stats, "files": files,
                    "validation": {"raw_checksums_match": True, "query_split_overlap": 0,
                                   "duplicate_candidate_pairs": 0, "missing_product_metadata": 0,
                                   "missing_query_sources": 0, "saved_pairs_and_labels_match_source": True},
                    "notes": ["No model training or enrichment performed.",
                              "Original null metadata values are preserved.",
                              "Parquet row order is unspecified; grouped previews sort by product ID, never relevance.",
                              "No relevance gains or binary relevance threshold have been assigned.",
                              "These candidate judgments are not exhaustive whole-catalog qrels."]}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "esci_us_small_build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (reports / "esci_us_small_training_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "splits": expected, "products": product_count,
                      "queries": query_count, "validation": manifest["validation"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output_dir.resolve())
