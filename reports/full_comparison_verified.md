# Verified full comparison

| Method | NDCG@10 |
| --- | --- |
| plain | 0.7009 |
| structured | 0.7131 |
| dense | 0.7338 |
| hybrid_plain | 0.7259 |
| hybrid_structured | 0.7325 |
| doc2query_t5 | 0.7091 |

All methods: 8,956 queries, 181,701 candidate pairs and 164,900 products. Complete coverage and independent NDCG validation passed.

Structured generation logged API charges: $1.4240369976. Total experimental accounting with pilots and uncertain reservations: $1.9225503872. T5 and E5 are local: API charges $0; compute/energy costs unknown.

T5 recorded batch processing: 23,000.246 seconds (6 h 23 min); E5 product model inference: 2,191.881 seconds (36.5 min), query model inference: 50.708 seconds. These have different timing boundaries and are not comparable total indexing times. Sleep and power interruptions occurred. T5 offline evaluation took 16.25 seconds; it is not serving latency.

Index construction time, index size for actual search engines and online whole-catalog retrieval latency remain outstanding. Hosted T5 cost is not measured. SPLADE and uniCOIL are not included.

Abstract updated with verified effectiveness only. No external submission made.
