"""Profile the official ESCI files and save an auditable JSON/Markdown report.

Only training queries are displayed as examples. Test data is inspected for
structural statistics, not used for prompt design or method selection.
"""
import json
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/esci"
OUTPUT = ROOT / "reports"


def main():
    OUTPUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="esci-profile-") as temporary:
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.execute("SET memory_limit='1GB'")
        con.execute("SET temp_directory=?", [temporary])
        for alias, file in [("examples", "shopping_queries_dataset_examples.parquet"),
                            ("products", "shopping_queries_dataset_products.parquet")]:
            path = str(RAW / file).replace("'", "''")
            if not (RAW / file).exists():
                raise FileNotFoundError(RAW / file)
            con.execute(f"CREATE VIEW {alias} AS SELECT * FROM read_parquet('{path}')")
        path = str(RAW / "shopping_queries_dataset_sources.csv").replace("'", "''")
        con.execute(f"CREATE VIEW sources AS SELECT * FROM read_csv_auto('{path}')")

        def rows(sql):
            cursor = con.execute(sql)
            keys = [x[0] for x in cursor.description]
            return [dict(zip(keys, values)) for values in cursor.fetchall()]

        def section(name, sql):
            print(f"Inspecting {name}", flush=True)
            return rows(sql)

        result = {}
        result["schemas"] = {table: rows(f"DESCRIBE {table}") for table in ["examples", "products", "sources"]}
        result["totals"] = {table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                            for table in ["examples", "products", "sources"]}
        result["example_splits"] = section("example splits", """
            SELECT product_locale, split, count(*) judgments,
                   count(DISTINCT query_id) queries, count(DISTINCT product_id) products
            FROM examples GROUP BY ALL ORDER BY product_locale, split
        """)
        result["version_flags"] = rows("""
            SELECT small_version, large_version, count(*) judgments
            FROM examples GROUP BY ALL ORDER BY small_version, large_version
        """)
        result["small_splits"] = section("small-version splits", """
            SELECT product_locale, split, count(*) judgments,
                   count(DISTINCT query_id) queries, count(DISTINCT product_id) products
            FROM examples WHERE small_version=1
            GROUP BY ALL ORDER BY product_locale, split
        """)
        result["label_counts"] = rows("""
            SELECT product_locale, split, esci_label, count(*) judgments
            FROM examples WHERE small_version=1
            GROUP BY ALL ORDER BY product_locale, split, esci_label
        """)
        result["candidates_per_query"] = section("candidate counts", """
            WITH q AS (
                SELECT product_locale, split, query_id, count(*) n,
                       count(DISTINCT product_id) unique_n
                FROM examples WHERE small_version=1 GROUP BY ALL
            )
            SELECT product_locale, split, min(n) min_candidates, median(n) median_candidates,
                   avg(n) mean_candidates, max(n) max_candidates,
                   count(*) FILTER (WHERE n != unique_n) queries_with_duplicate_candidates
            FROM q GROUP BY ALL ORDER BY product_locale, split
        """)
        fields = ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]
        expressions = ["count(*) products", "count(DISTINCT product_id) unique_ids"]
        for field in fields:
            expressions.extend([
                f"count(*) FILTER (WHERE {field} IS NULL OR trim({field})='') AS {field}_missing",
                f"avg(length(coalesce({field},''))) AS {field}_mean_characters",
            ])
        result["product_fields"] = section("product metadata completeness",
            "SELECT product_locale, " + ", ".join(expressions) + " FROM products GROUP BY product_locale ORDER BY product_locale")
        result["integrity"] = section("keys and split integrity", """
            SELECT
              (SELECT count(*) FROM (
                  SELECT product_id, product_locale FROM products GROUP BY ALL HAVING count(*)>1
              )) duplicate_product_locale_keys,
              (SELECT count(*) FROM (
                  SELECT query_id, product_locale FROM examples GROUP BY ALL HAVING count(DISTINCT split)>1
              )) query_ids_in_multiple_splits,
              (SELECT count(*) FROM examples e ANTI JOIN products p
                  ON e.product_id=p.product_id AND e.product_locale=p.product_locale) unmatched_judgments,
              (SELECT count(*) FROM examples WHERE esci_label IS NULL
                  OR esci_label NOT IN ('E','S','C','I')) invalid_labels
        """)[0]
        result["small_split_integrity"] = rows("""
            SELECT count(*) overlapping_query_ids FROM (
                SELECT query_id, product_locale FROM examples WHERE small_version=1
                GROUP BY ALL HAVING count(DISTINCT split)>1
            )
        """)[0]
        result["cross_split_ids_all_versions"] = rows("""
            SELECT query_id, product_locale, count(DISTINCT split) splits
            FROM examples GROUP BY ALL HAVING count(DISTINCT split)>1
        """)
        result["us_small_product_fields"] = section("US ranking product fields", """
            WITH keys AS (
                SELECT DISTINCT product_id, product_locale FROM examples
                WHERE small_version=1 AND product_locale='us'
            )
            SELECT count(*) products,
                   count(*) FILTER (WHERE product_description IS NULL OR trim(product_description)='') missing_description,
                   count(*) FILTER (WHERE product_bullet_point IS NULL OR trim(product_bullet_point)='') missing_bullet_points,
                   count(*) FILTER (WHERE product_description IS NULL OR trim(product_description)='') * 100.0/count(*) missing_description_percent
            FROM products JOIN keys USING(product_id,product_locale)
        """)[0]
        result["us_small_product_overlap"] = section("US product overlap", """
            WITH keys AS (
                SELECT product_id, bool_or(split='train') in_train, bool_or(split='test') in_test
                FROM examples WHERE small_version=1 AND product_locale='us' GROUP BY product_id
            )
            SELECT count(*) union_products,
                   count(*) FILTER (WHERE in_train) train_products,
                   count(*) FILTER (WHERE in_test) test_products,
                   count(*) FILTER (WHERE in_train AND in_test) shared_products
            FROM keys
        """)[0]
        result["us_small_query_lengths"] = rows("""
            WITH q AS (
                SELECT DISTINCT split, query_id, query FROM examples
                WHERE small_version=1 AND product_locale='us'
            ), n AS (SELECT *, len(regexp_extract_all(query,'\\S+')) words FROM q)
            SELECT split, count(*) queries, median(words) median_words, avg(words) mean_words,
                   count(*) FILTER (WHERE words>=6) queries_at_least_six_words
            FROM n GROUP BY split ORDER BY split
        """)
        result["sources"] = rows("SELECT source, count(*) queries FROM sources GROUP BY source ORDER BY source")
        result["training_samples"] = section("training-only examples", """
            WITH q AS (
                SELECT DISTINCT query_id, query FROM examples
                WHERE small_version=1 AND product_locale='us' AND split='train'
            ), chosen AS (
                (SELECT * FROM q WHERE len(regexp_extract_all(query,'\\S+'))<=2 ORDER BY hash(query_id) LIMIT 2)
                UNION ALL
                (SELECT * FROM q WHERE len(regexp_extract_all(query,'\\S+'))>=6 ORDER BY hash(query_id) LIMIT 2)
            ), candidates AS (
                SELECT c.query_id, c.query, e.esci_label, e.product_id, p.product_title,
                       row_number() OVER (PARTITION BY c.query_id, e.esci_label ORDER BY e.product_id) rn
                FROM chosen c JOIN examples e USING(query_id)
                JOIN products p ON e.product_id=p.product_id AND e.product_locale=p.product_locale
                WHERE e.small_version=1 AND e.product_locale='us' AND e.split='train'
            )
            SELECT query_id, query, esci_label, product_id, product_title
            FROM candidates WHERE rn=1 ORDER BY query_id, esci_label
        """)
        sample_keys = [(r["product_id"],) for r in result["training_samples"] if r["esci_label"]=="E"]
        con.execute("CREATE TEMP TABLE sample_keys(product_id VARCHAR)")
        if sample_keys:
            con.executemany("INSERT INTO sample_keys VALUES (?)", sample_keys)
        result["sample_product_metadata"] = rows("""
            SELECT p.product_id, product_title, product_brand, product_color,
                   left(product_description,300) description_excerpt,
                   left(product_bullet_point,300) bullet_point_excerpt
            FROM products p JOIN sample_keys k USING(product_id)
            WHERE product_locale='us' ORDER BY product_id
        """)
        con.close()

    (OUTPUT / "esci_profile.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Amazon ESCI: initial dataset inspection", "",
             "Computed from the downloaded, checksum-verified official files.", "",
             "Source revision and file checksums: `data/esci_manifest.json`.", "",
             "Only training queries are shown below. Test rows are used only for structural statistics.", ""]

    def table(records):
        if not records:
            return ["No rows.", ""]
        keys = list(records[0])
        def format_value(value):
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value).replace("|", "\\|").replace("\n", " ")
        return ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"]*len(keys)) + " |"] + [
            "| " + " | ".join(format_value(record[key]) for key in keys) + " |" for record in records
        ] + [""]

    lines += ["## Files and schemas", ""]
    for name, schema in result["schemas"].items():
        lines += [f"### {name}: {result['totals'][name]:,} rows", ""]
        lines += table([{k: row[k] for k in ["column_name", "column_type"]} for row in schema])
    for name in ["example_splits", "version_flags", "small_splits", "label_counts", "candidates_per_query",
                 "product_fields", "integrity", "small_split_integrity", "cross_split_ids_all_versions",
                 "us_small_product_fields", "us_small_product_overlap", "us_small_query_lengths", "sources",
                 "training_samples", "sample_product_metadata"]:
        lines += ["## " + name.replace("_", " ").title(), ""]
        value = result[name]
        lines += table(value if isinstance(value, list) else [value])
    lines += ["## Implications for the experiment", "",
              "- Identify products by `(product_id, product_locale)` and preserve official query splits.",
              "- Begin with English (US), `small_version=1`, using the supplied candidate sets.",
              "- Candidate judgments are not exhaustive whole-catalog judgments.",
              "- Generate document expansions from product metadata only, excluding queries and relevance labels.",
              "- Build development data from training query IDs and keep the official test split for final comparisons.",
              "- Fix the same original product fields, tokenizer, BM25 settings, and relevance gains for all variants.",
              "- Use fixed, complete candidate judgments for normalization; never build qrels from retrieved hits.",
              "- Product overlap between train and test does not imply query leakage: document indexing is unsupervised.",
              "- Missing descriptions affect the enrichment hypothesis and should be included in subgroup analysis.",
              "- The full release has one cross-split query ID (79706, US); the small ranking subset has no overlapping query IDs.",
              "- Candidate counts in the files can exceed the README's stated 40; use the actual supplied candidate lists.",
              "- Verify the relevance-gain convention before implementing evaluation: the archived upstream conversion script's substitute/complement mapping deserves reconciliation with the benchmark paper.",
              "- These data do not establish multi-turn conversational search effectiveness.", ""]
    (OUTPUT / "esci_inspection.md").write_text("\n".join(lines))
    print(json.dumps({"totals": result["totals"], "small_splits": result["small_splits"],
                      "integrity": result["integrity"]}, indent=2), flush=True)
    print("Saved reports/esci_profile.json and reports/esci_inspection.md", flush=True)


if __name__ == "__main__":
    main()
