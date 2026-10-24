"""Run the fixed plain-BM25 candidate-ranking reference on ESCI."""
import argparse
import csv
import gzip
import hashlib
import json
import random
import statistics as statistics_module
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecirpaper.metrics import GAINS, evaluate_ranking
from ecirpaper.retrieval import ANALYZER, FIELDS, document_text, rank_candidates, tokenize

DATA = ROOT / "data/processed/esci_us_small"
CACHE = ROOT / "data/processed/bm25_plain_stats.json.gz"


def quoted(path):
    return "'" + str(path).replace("'", "''") + "'"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_statistics(con, manifest):
    fingerprint = next(r["sha256"] for r in manifest["files"] if r["name"] == "products.parquet")
    if file_hash(DATA / "products.parquet") != fingerprint:
        raise ValueError("Product file differs from dataset manifest")
    identity = {"products_sha256": fingerprint, "fields": list(FIELDS), "analyzer": ANALYZER}
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as stream:
            cached = json.load(stream)
        if cached["identity"] == identity:
            print("Using verified corpus-statistics cache", flush=True)
            return cached
    cursor = con.execute(f"SELECT {','.join(FIELDS)} FROM read_parquet({quoted(DATA / 'products.parquet')})")
    df = Counter()
    n, total_length = 0, 0
    started = time.perf_counter()
    while batch := cursor.fetchmany(512):
        for row in batch:
            tokens = tokenize(document_text(row))
            df.update(set(tokens))
            total_length += len(tokens)
            n += 1
        if n % (512 * 100) == 0:
            print(f"Corpus statistics: {n:,}/{manifest['products']:,} products ({time.perf_counter()-started:.1f}s)", flush=True)
    if n != manifest["products"]:
        raise ValueError("Corpus size differs from manifest")
    result = {"identity": identity, "documents": n, "average_length": total_length / n,
              "total_tokens": total_length, "document_frequency": dict(df)}
    temporary = CACHE.with_suffix(CACHE.suffix + ".part")
    with gzip.open(temporary, "wt") as stream:
        json.dump(result, stream)
    temporary.replace(CACHE)
    return result


def percentile(values, fraction):
    position = fraction * (len(values) - 1)
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def summarize(rows, bootstrap_repetitions, seed):
    names = ["ndcg_at_10", "ndcg_full", "mrr_exact", "exact_at_1", "recall_exact_at_10"]
    result = {}
    for metric in names:
        values = [r[metric] for r in rows if r[metric] is not None]
        result[metric] = {"mean": statistics_module.mean(values), "queries": len(values)} if values else {"mean": None, "queries": 0}
    # Query-level intervals on the two graded metrics. Enrichment comparisons
    # should subsequently bootstrap paired per-query score differences.
    random_source = random.Random(seed)
    distributions = {metric: [] for metric in ["ndcg_at_10", "ndcg_full"]}
    n = len(rows)
    for _ in range(bootstrap_repetitions):
        totals = {metric: 0.0 for metric in distributions}
        for _ in range(n):
            row = rows[random_source.randrange(n)]
            for metric in totals:
                totals[metric] += row[metric]
        for metric in totals:
            distributions[metric].append(totals[metric] / n)
    for metric, samples in distributions.items():
        samples.sort()
        result[metric]["bootstrap_95_percent_ci"] = [percentile(samples, .025), percentile(samples, .975)]
    return result


def main(args):
    if args.k1 <= 0 or not 0 <= args.b <= 1 or args.bootstrap < 1:
        raise ValueError("Invalid BM25 or bootstrap settings")
    if args.limit_queries is not None and args.limit_queries <= 0:
        raise ValueError("Query limit must be positive")
    suffix = f"_{args.limit_queries}" if args.limit_queries else ""
    output = ROOT / "outputs" / f"bm25_{args.split}{suffix}"
    if args.output_dir:
        output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Existing run: {output}; choose --output-dir to preserve it")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    input_fingerprint = next(r["sha256"] for r in manifest["files"] if r["name"] == args.split + ".parquet")
    if file_hash(DATA / (args.split + ".parquet")) != input_fingerprint:
        raise ValueError("Evaluation split differs from dataset manifest")
    total_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".bm25-run-", dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        con = duckdb.connect()
        con.execute("SET threads=1")
        con.execute("SET memory_limit='2GB'")
        con.execute("SET temp_directory=?", [str(Path(temporary) / "spill")])
        corpus_start = time.perf_counter()
        stats = corpus_statistics(con, manifest)
        corpus_seconds = time.perf_counter() - corpus_start
        con.execute(f"CREATE VIEW input AS SELECT * FROM read_parquet({quoted(DATA / (args.split + '.parquet'))})")
        selection = f"SELECT DISTINCT query_id FROM input ORDER BY hash(query_id + {args.seed})"
        if args.limit_queries:
            if args.limit_queries <= 0:
                raise ValueError("Query limit must be positive")
            selection += f" LIMIT {args.limit_queries}"
        con.execute(f"CREATE TEMP TABLE selected_queries AS {selection}")
        expected_queries = con.execute("SELECT count(*) FROM selected_queries").fetchone()[0]
        expected_candidates = con.execute("SELECT count(*) FROM input SEMI JOIN selected_queries USING(query_id)").fetchone()[0]
        cursor = con.execute(f"SELECT query_id,query,source,product_id,esci_label,{','.join(FIELDS)} FROM input SEMI JOIN selected_queries USING(query_id) ORDER BY query_id,product_id")
        columns = [r[0] for r in cursor.description]
        query_results = []
        candidate_total = 0
        zero_queries = 0
        ranking_start = time.perf_counter()
        with (stage / "rankings.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_id", "product_id", "rank", "bm25_score"])

            def evaluate_group(group):
                nonlocal candidate_total, zero_queries
                query_id = group[0]["query_id"]
                judgments = {r["product_id"]: r["esci_label"] for r in group}
                if len(judgments) != len(group) or len({r["query"] for r in group}) != 1:
                    raise ValueError("Inconsistent query/candidate group")
                scored = rank_candidates(group[0]["query"], group, stats, args.k1, args.b)
                ranking = [r[0] for r in scored]
                metrics = evaluate_ranking(ranking, judgments)
                zero_candidates = sum(score == 0 for _, score in scored)
                zero_queries += zero_candidates == len(scored)
                query_results.append({"query_id": query_id, "source": group[0]["source"],
                                      "candidates": len(group), "zero_score_candidates": zero_candidates,
                                      "exact_candidates": sum(label == "E" for label in judgments.values()), **metrics})
                for rank, (product, score) in enumerate(scored, 1):
                    writer.writerow([query_id, product, rank, format(score, ".17g")])
                candidate_total += len(group)
                if len(query_results) % 500 == 0:
                    print(f"Ranked {len(query_results):,}/{expected_queries:,} queries ({time.perf_counter()-ranking_start:.1f}s)", flush=True)

            group, current_id = [], None
            while batch := cursor.fetchmany(256):
                for row in batch:
                    item = dict(zip(columns, row))
                    if current_id is not None and item["query_id"] != current_id:
                        evaluate_group(group)
                        group = []
                    current_id = item["query_id"]
                    group.append(item)
            if group:
                evaluate_group(group)
        ranking_seconds = time.perf_counter() - ranking_start
        con.close()
        if len(query_results) != expected_queries or candidate_total != expected_candidates:
            raise ValueError("Output query/candidate coverage mismatch")
        with (stage / "per_query.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(query_results[0]))
            writer.writeheader()
            writer.writerows(query_results)
        print("Computing query-level confidence intervals", flush=True)
        aggregates = summarize(query_results, args.bootstrap, args.seed)
        summary = {"method": "plain_bm25_candidate_ranking", "split": args.split,
                   "queries": expected_queries, "candidate_pairs": candidate_total,
                   "all_zero_score_queries": zero_queries, "metrics": aggregates,
                   "config": {"fields": list(FIELDS), "analyzer": ANALYZER,
                              "stopwords_removed": False, "stemming": False, "text_truncated": False,
                              "k1": args.k1, "b": args.b, "query_term_frequency": "unique_terms",
                              "gain_mapping": GAINS, "tie_break": "product_id ascending",
                              "query_limit": args.limit_queries, "seed": args.seed,
                              "bootstrap_repetitions": args.bootstrap},
                   "corpus": {"documents": stats["documents"], "average_length": stats["average_length"],
                              "total_tokens": stats["total_tokens"], "vocabulary": len(stats["document_frequency"]),
                              "statistics_scope": "union of all English small-version candidate products"},
                   "provenance": {"source_revision": manifest["source_revision"],
                                  "products_sha256": stats["identity"]["products_sha256"],
                                  "evaluation_split_sha256": input_fingerprint,
                                  "implementation_sha256": {name: file_hash(ROOT / name) for name in [
                                      "scripts/run_bm25.py", "src/ecirpaper/retrieval.py", "src/ecirpaper/metrics.py"]},
                                  "duckdb_version": duckdb.__version__},
                   "timings_seconds": {"corpus_statistics": corpus_seconds,
                                       "scoring_and_evaluation": ranking_seconds,
                                       "total_pipeline": time.perf_counter() - total_start},
                   "built_at_utc": datetime.now(timezone.utc).isoformat(),
                   "notes": ["Initial untuned reference, not an optimized BM25 claim.",
                             "Corpus statistics use metadata only, without query text or relevance labels.",
                             "Every supplied candidate is ranked; zero scores are retained.",
                             "NDCG ideal rankings use all official candidate judgments.",
                             "Recall-exact excludes queries with no Exact candidates; MRR-exact includes them as zero.",
                             "Elapsed processing times are not serving or whole-catalog retrieval latency.",
                             "Confidence intervals describe this benchmark query set, not all shopping traffic."]}
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(output)
    reports = ROOT / "reports"
    report_suffix = f"_{args.split}{suffix}"
    (reports / f"bm25{report_suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Plain BM25: initial ESCI reference", "",
             f"Split: `{args.split}`. Queries: **{expected_queries:,}**. Candidate pairs: **{candidate_total:,}**.", "",
             "| Metric | Mean | Query-level bootstrap 95% interval |", "| --- | --- | --- |"]
    for metric, values in aggregates.items():
        interval = values.get("bootstrap_95_percent_ci")
        ci = f"{interval[0]:.4f}–{interval[1]:.4f}" if interval else "—"
        mean = values["mean"]
        lines.append(f"| {metric} | {mean:.4f} | {ci} |" if mean is not None else f"| {metric} | undefined | {ci} |")
    lines += ["", "## Configuration and interpretation", "",
              f"BM25: k1={args.k1}, b={args.b}. Corpus: {stats['documents']:,} products.",
              "Text fields: title, description, bullet points, brand, colour, concatenated with equal weighting.",
              "Unicode alphanumeric tokenization, case folding and HTML cleanup; no stemming, stopword removal or truncation.",
              "ESCI direct gains: E=1, S=0.1, C=0.01, I=0, following section 3.1 of the benchmark paper.",
              "The archived upstream qrels conversion swaps Substitute/Complement ordering; this implementation follows the paper.",
              "Scores use corpus-wide positive smoothed IDF, TF saturation, and length normalization.",
              "All supplied candidates are ranked, with ties resolved by product ID rather than relevance.",
              "This is an initial untuned candidate-ranking baseline. It does not measure whole-catalog recall or deployment latency.",
              "Full-rank NDCG and NDCG@10 are separate measures; do not compare one with published values of the other.",
              "Exact recall is averaged only over queries with at least one Exact candidate; see JSON for metric denominators.",
              f"All-zero-score queries: {zero_queries:,}.", "",
              "Sources: [ESCI paper](https://arxiv.org/abs/2206.06588), "
              "[Lucene BM25 IDF reference](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java).", "",
              f"Complete rankings and per-query metrics: `{output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}`.",
              "The JSON records configuration, input/implementation hashes, uncertainty settings, and processing times.", ""]
    (reports / f"bm25{report_suffix}.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "queries": expected_queries, "metrics": aggregates}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=.75)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    main(parser.parse_args())
