# Shared reranker comparison

Accuracy uses all 8,956 official judged-candidate sets. Latency uses exact full-corpus retrieval on a fixed random sample of100 queries, three shuffled repeats. These are different evaluation scopes; no whole-catalog recall claim.

| Method | Reranked candidates | NDCG@10 | Baseline | Total median / p95(ms) | Reranker median(ms) |
|---|---:|---:|---:|---:|---:|
| plain | 10 | 0.7206 | 0.7009 | 17.06 / 21.13 | 12.30 |
| plain | 20 | 0.7375 | 0.7009 | 26.17 / 31.64 | 21.05 |
| plain | 50 | 0.7391 | 0.7009 | 56.68 / 65.59 | 52.21 |
| plain | 100 | 0.7391 | 0.7009 | 110.19 / 127.84 | 105.59 |
| structured | 10 | 0.7266 | 0.7131 | 16.51 / 20.11 | 11.54 |
| structured | 20 | 0.7379 | 0.7131 | 25.22 / 29.62 | 20.18 |
| structured | 50 | 0.7391 | 0.7131 | 55.02 / 63.21 | 50.49 |
| structured | 100 | 0.7391 | 0.7131 | 106.53 / 121.35 | 101.85 |
| ling_doc2query | 10 | 0.7278 | 0.7150 | 16.22 / 20.12 | 11.68 |
| ling_doc2query | 20 | 0.7379 | 0.7150 | 25.03 / 29.95 | 20.21 |
| ling_doc2query | 50 | 0.7391 | 0.7150 | 55.01 / 62.76 | 50.54 |
| ling_doc2query | 100 | 0.7391 | 0.7150 | 106.31 / 119.94 | 101.41 |
| e5 | 10 | 0.7393 | 0.7338 | 20.14 / 23.10 | 11.57 |
| e5 | 20 | 0.7389 | 0.7338 | 28.81 / 33.21 | 20.30 |
| e5 | 50 | 0.7391 | 0.7338 | 58.38 / 66.41 | 49.61 |
| e5 | 100 | 0.7391 | 0.7338 | 110.30 / 124.83 | 101.86 |
| splade | 10 | 0.7389 | 0.7368 | 21.18 / 24.20 | 11.34 |
| splade | 20 | 0.7390 | 0.7368 | 30.13 / 34.10 | 20.16 |
| splade | 50 | 0.7391 | 0.7368 | 59.34 / 66.98 | 49.28 |
| splade | 100 | 0.7391 | 0.7368 | 110.68 / 125.15 | 100.97 |

Pinned MiniLM reranker, revision 233902d25c440f23af6f7d6e94d2946bac0bee0a, float16, pair max256 tokens, batch32, one T4, one CPU thread. Same original metadata for all methods; no generated expansion given to the reranker; no ESCI training. Neural retrievers retain their previous float32 settings. Each latency request includes fresh retrieval (including neural query encoding), pair tokenization/transfers/inference and deterministic top10. Network, model loading and cold starts excluded.

N50/N100 may rerank the complete official candidate set for many queries. This cannot establish that full-catalog first-stage shortlists have equal quality. The shared quality score cache saves experiment compute; serving timings perform fresh scoring and do not use the cache.

Modal metered usage is retrieved separately in outputs/modal_rerank/billing_report.json. Do not treat this whole experimental job bill as per-query serving cost or assign it to one method. Preparation costs remain in the previous verified report.

Independent validation passed: 181,701 unique pairs, all20 configurations on8,956 queries,6,000 timed requests. Maximum NDCG discrepancy 4.44e-16.

At N50 the entire provided candidate set is reranked for 8918 of8,956 queries; at N100, 8956. Convergence is largely expected from identical reranker scores on identical complete candidate sets.

Exploratory paired query bootstrap (1,000 resamples, seed42, no multiple-comparison adjustment):
- ['structured', 10] minus ['ling_doc2query', 10]: -0.001243;95% interval [-0.002560,+0.000105].
- ['structured', 20] minus ['e5', 10]: -0.001441;95% interval [-0.003347,+0.000609].
- ['structured', 10] minus ['e5', 10]: -0.012751;95% interval [-0.015285,-0.009975].

Cloud worker finished in870.76s(14.51min); independent validation subsequently passed after correcting the validator export path. The failed check compared JSON exports with different field names; semantic query and document equality was then checked in full. All previous outputs retained. App-specific billing for this run and its failed startup is pending in the provider report. Existing reported experiment app total is$4.01109904; no API calls were made.

## Provider billing update

Successful reranker app: metered compute$0.18909833; total including egress$0.18969745. All currently reported ESCI experiment apps total$4.20079649. Workspace payable is zero after credits/allowances; resource cost is not zero. Failed startup app ap-5X57gzpPDPklrOBMwicMmH is not present in the provider report, so its charge remains unknown rather than assumed zero. These costs cover the experimental job, not an independently measured cost per method or serving request. Latest snapshots: outputs/modal_rerank/billing_report_latest.json and billing_summary_latest.json.

Final billing follow-up: failed startup remains absent after three additional provider checks. Charge recorded as provider-unreported/unknown, not zero; attempt retained in accounting. All currently reported experiment app charges total$4.20079649. Billing monitor stopped; experimental results remain verified.
