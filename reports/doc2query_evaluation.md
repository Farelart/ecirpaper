# Full doc2query-T5 comparison

| Method | NDCG@10 | Full NDCG | Exact at 1 |
| --- | --- | --- | --- |
| plain | 0.7009 | 0.8240 | 0.6265 |
| structured | 0.7131 | 0.8315 | 0.6476 |
| doc2query_t5 | 0.7091 | 0.8298 | 0.6459 |

Pinned pretrained doc2query-T5 with 3 queries per product: economical configuration, not reproduction of the original 40-query MS MARCO experiment. Same saved metadata inputs as structured LLM; full original metadata remains in BM25.

All 8,956 queries and 181,701 candidates evaluated. Candidate ranking does not establish whole-catalog recall or online latency. Local inference adds no API charges; energy and hardware costs remain unmeasured. Paired bootstrap intervals use 1,000 query resamples and are exploratory.
