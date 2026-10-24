# Full embedding and hybrid comparison

All 8,956 queries, 181,701 pairs and 164,900 products.

| Method | NDCG@10 | Full NDCG | Exact at 1 |
| --- | --- | --- | --- |
| plain | 0.7009 | 0.8240 | 0.6265 |
| structured | 0.7131 | 0.8315 | 0.6476 |
| dense | 0.7338 | 0.8434 | 0.6855 |
| hybrid_plain | 0.7259 | 0.8388 | 0.6732 |
| hybrid_structured | 0.7325 | 0.8425 | 0.6842 |

Candidate ranking only. Query vectors were encoded offline in batches for effectiveness evaluation. These timings do not establish online per-search latency, index cost, or whole-catalog recall.

E5-small uses a pinned checkpoint, full original metadata capped at 512 model tokens, required query/passage prefixes, masked mean pooling and L2 normalization. Hybrid uses equal-weight reciprocal rank fusion with k=60. No ESCI training or test-result tuning.

Paired intervals are exploratory, with 1,000 query bootstrap resamples and no multiple-comparison correction. Local API spending is zero; compute and energy are not free. See the encoding summary for measured runtime and representation bytes.
