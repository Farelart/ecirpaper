# Full ESCI enrichment comparison

8,956 queries, 181,701 candidate pairs, 164,900 products.

| Variant | NDCG@10 | Full NDCG | Exact at rank 1 | Change in NDCG@10 | Paired 95% interval |
| --- | --- | --- | --- | --- | --- |
| plain | 0.7009 | 0.8240 | 0.6265 | — | — |
| keywords | 0.7044 | 0.8263 | 0.6343 | +0.0035 | [+0.0023, +0.0047] |
| synonyms | 0.7118 | 0.8303 | 0.6436 | +0.0109 | [+0.0094, +0.0124] |
| situations | 0.7055 | 0.8266 | 0.6331 | +0.0046 | [+0.0033, +0.0059] |
| all | 0.7131 | 0.8315 | 0.6476 | +0.0122 | [+0.0104, +0.0139] |

This corpus differs from the initial 482,105-product plain baseline; this experiment reruns plain BM25 for paired comparisons.

All variants use original full metadata plus their selected expansion fields. BM25 corpus statistics are recomputed separately for each variant. Relevance judgments are used only for evaluation. No parameter or prompt tuning was performed on these test results.

Paired bootstrap intervals use 1,000 query resamples with seed 42. Intervals are exploratory and not adjusted for multiple comparisons. This is candidate ranking, not whole-catalog recall, latency, or multi-turn evaluation.

Enrichment includes source-span keyword filtering and deterministic cleanup. See generation manifests and pilot audits for input truncation, batching, retries and semantic limitations.

## Outcome and validation

All three fields improve NDCG@10 by **0.0122 absolute (1.73% relative)** over
plain BM25. The paired 95% interval is **[+0.0104, +0.0139]**. Synonyms account
for the largest single-field gain; this is not a causal decomposition of the
combined method. Exact matches at rank 1 rise from **62.65% to 64.76%**.

The combined method improves NDCG@10 for 4,029 queries, worsens it for 3,021,
and ties for 1,906. Descriptively, the one-token query group has a lower mean
score after combined enrichment; longer-query groups have higher means.
No subgroup significance or general conversational-search claim is made.

Generation reported approximately **$1.42** for the full test-product set.
The complete accounted total, including both pilots, retained reservations for
uncertain charges and resumed requests, is **$1.9225503872**, below the $5 cap.
Raw inputs, outputs, provider usage and the budget ledger are retained.

Coverage checks verified all **164,900** expected products, with no extras or
duplicates. Each of the five rankings contains all **181,701** official pairs
across **8,956** queries exactly once. An independent SQL calculation reproduced
the plain and combined NDCG@10 means. All 14 verification tests passed.

These results support a modest improvement over this fixed plain BM25 setup.
They do not establish superiority to an optimized BM25 system, query-style
expansion, dense retrieval or a reranker. Source-span filtering alone does not
ensure semantic grounding or handle negations; the pilot audit's concerns remain.

Files: `reports/full_generation.json`, `reports/enrichment_evaluation_validation.json`,
`reports/full_experiment_protocol.json`, and `outputs/enrichment_evaluation/`.
