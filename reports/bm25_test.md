# Plain BM25: initial ESCI reference

Split: `test`. Queries: **8,956**. Candidate pairs: **181,701**.

| Metric | Mean | Query-level bootstrap 95% interval |
| --- | --- | --- |
| ndcg_at_10 | 0.7008 | 0.6961–0.7055 |
| ndcg_full | 0.8240 | 0.8206–0.8274 |
| mrr_exact | 0.7553 | — |
| exact_at_1 | 0.6264 | — |
| recall_exact_at_10 | 0.6668 | — |

## Configuration and interpretation

BM25: k1=1.2, b=0.75. Corpus: 482,105 products.
Text fields: title, description, bullet points, brand, colour, concatenated with equal weighting.
Unicode alphanumeric tokenization, case folding and HTML cleanup; no stemming, stopword removal or truncation.
ESCI direct gains: E=1, S=0.1, C=0.01, I=0, following section 3.1 of the benchmark paper.
The archived upstream qrels conversion swaps Substitute/Complement ordering; this implementation follows the paper.
Scores use corpus-wide positive smoothed IDF, TF saturation, and length normalization.
All supplied candidates are ranked, with ties resolved by product ID rather than relevance.
This is an initial untuned candidate-ranking baseline. It does not measure whole-catalog recall or deployment latency.
Full-rank NDCG and NDCG@10 are separate measures; do not compare one with published values of the other.
Exact recall is averaged only over queries with at least one Exact candidate; see JSON for metric denominators.
All-zero-score queries: 122.

Sources: [ESCI paper](https://arxiv.org/abs/2206.06588), [Lucene BM25 IDF reference](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/search/similarities/BM25Similarity.java).

Complete rankings and per-query metrics: `outputs/bm25_test`.
The JSON records configuration, input/implementation hashes, uncertainty settings, and processing times.
