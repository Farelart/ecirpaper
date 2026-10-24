# Verified full ESCI comparison

All authorized comparisons are complete: 164,900 products, 8,956 official queries, and 181,701 judged query-product pairs. No ESCI-specific training or tuning.

Structured expansion improves plain BM25 and this three-query T5 baseline. The matched Ling generated-query control performs slightly better and uses fewer logged API dollars. E5 and SPLADE rank candidates better; BM25 variants have lower measured local CPU search latency. These results support an empirical trade-off study, not a claim that structured generation is uniquely superior.

| Method | Candidate NDCG@10 | Median ms | p95 ms | Index construction s | Serialized MiB |
|---|---:|---:|---:|---:|---:|
| Plain BM25 | 0.7009 | 0.82 | 2.19 | 14.69 | 107.46 |
| Structured Ling + BM25 | 0.7131 | 0.83 | 2.16 | 16.23 | 113.84 |
| T5 queries + BM25 | 0.7091 | 0.81 | 2.19 | 22.54 | 113.63 |
| Ling queries + BM25 | 0.7150 | 0.80 | 2.12 | 15.04 | 110.50 |
| E5-small | 0.7338 | 12.89 | 14.50 | 0.48 | 241.55 |
| SPLADE | 0.7368 | 39.27 | 46.30 | 10.19 | 149.50 |
| BM25 + E5 RRF | 0.7259 | 15.16 | 17.14 | 15.18 | 349.02 |
| Structured BM25 + E5 RRF | 0.7325 | 15.13 | 17.13 | 16.72 | 355.39 |

Quality ranks all provided judged candidates. Timing is a separate exact full-corpus top10 experiment. Hybrid quality uses full-candidate RRF, whereas timing fuses 1,000 hits per component with fixed RRF k60. Their quality and timing describe separate protocols, not the same end-to-end ranking configuration.

BM25 index construction includes tokenization plus assembly/serialization, excluding expansion generation. E5/SPLADE index construction covers representation assembly/serialization only, excluding document encoding. Hybrid construction sums component measurements including lexical tokenization; underlying hybrid summaries record assembly-only sums. Sizes exclude common product-ID mapping, metadata, model weights and resident-memory overhead.

## Preparation measurements

| Preparation | Logged API / reported compute cost | Recorded processing time |
|---|---:|---|
| Structured Ling, full corpus | API US$1.424037 | Reconstructed first-request-to-last-response span: 44.80min |
| Matched Ling queries, full corpus | API US$0.995010 | Reconstructed first-request-to-last-response span: 62.76min |
| Hosted T5, one T4 | Compute US$1.607373 | 7,196.832s / 119.95min saved-batch processing |
| SPLADE, one T4 | Compute US$1.522311 | Products 6,561.428s / 109.36min; query batches 16.060s |
| E5-small, local MPS | Monetary compute cost unmeasured | Product inference 2,191.881s / 36.53min; query batches 50.708s |

API figures include logged invalid-output/retry requests. Shared API ledger usage, including uncertain reservations, is US$2.424933 of the unchanged US$4.40 ceiling. Earlier Gemini pilot accounting is outside this ledger and remains in historical reports. Generation spans have now been reconstructed across sessions from each logged request start and duration. Structured generation spans 2,688.022s; Ling query generation spans 3,765.473s. Adding index construction gives 45.07min and 63.01min respectively. These sums are measured phases from separate runs, not a single uninterrupted end-to-end timer.

All experiment apps appearing in the downloaded Modal usage report, including reported pilots and failed attempts, total **US$3.16115345** including egress, below the US$30 ceiling. Full T5 app including egress: US$1.60778359; SPLADE: US$1.52273621. Monthly workspace summary: US$3.43528563 ephemeral-app usage and US$0 payable after credits/allowances. Workspace totals and the detailed report differ in scope/aggregation; neither is attributed entirely to these two runs. Charges are provider-reported snapshots, subject to adjustments; absent failed-attempt charges do not establish zero usage.

API charges and cloud resource costs are different deployment measures. Saved-batch timings exclude some orchestration/checkpoint/restart overhead; full cloud elapsed time is unknown. Cloud T5 preparation does not replace local T5 quality rankings. Local E5 compute/energy costs and serving monetary costs remain unknown.

## Paired quality comparisons

- structured_minus_plain: mean ΔNDCG@10 +0.01215064; paired 95% interval [+0.01028948, +0.01401513].
- structured_minus_doc2query_t5: mean ΔNDCG@10 +0.00391836; paired 95% interval [+0.00206524, +0.00577098].
- structured_minus_ling_doc2query: mean ΔNDCG@10 -0.00199668; paired 95% interval [-0.00355306, -0.00023442].
- dense_minus_structured: mean ΔNDCG@10 +0.02073738; paired 95% interval [+0.01688981, +0.02432456].
- splade_minus_dense: mean ΔNDCG@10 +0.00301186; paired 95% interval [-0.00015748, +0.00607233].

Exploratory query bootstrap: 1,000 resamples, seed 42, standard-library RNG; no multiple-comparison correction. Structured-minus-Ling-query interval excludes zero below it; SPLADE-minus-E5 includes zero. Small BM25 latency differences should not be interpreted as a reliable ordering without independent-run uncertainty analysis.

## Verification and limits

- Independent SQL validates all eight rankings, full coverage, unique valid ranks, and NDCG agreement within 1e-12.
- Each method has 26,868 finite timings covering every query once per repeat. Saved median/p95 were independently recomputed.
- SPLADE IDs, settings and all 86 checkpoint matrices were verified. All weights are finite/positive; independent sparse-dot computation agrees with 381 saved scores across 21 queries within float32 tolerance.
- BM25 index scores agree with frozen references for 100 queries. E5 full-corpus dot products were independently checked, and every timed dense result was checked for finite bounded cosine values.
- Retrieval: Apple M4/16GiB, one CPU thread, warm indexes, 20 warmups and three shuffled passes. Neural query encoding included; network, model loading and ANN excluded. Python3.9/PyTorch2.8.0/NumPy2.0.2/SciPy1.13.1/Transformers4.57.6.
- Both Ling strategies: inclusionai/ling-3.0-flash, identical saved shortened metadata, temperature0.2, reasoning disabled, 220 output tokens/product; generation prompts differ.
- E5/SPLADE receive full metadata capped at512tokens, versus shortened generation inputs. Hardware/input differences limit cross-family attribution. T5 generates three queries/product, not the original40-query setup.
- No multi-turn conversation, whole-catalog recall, network serving, ANN or cheapest-total-cost result is established. uniCOIL was not part of this completed experiment.

Evidence: final_experiments_verified.json; retrieval_benchmark.json; ranking_sql_validation_final.json; final_paired_comparisons.json; splade_encoding_validation.json; modal_t5_full_verified.json. Earlier checkpoints, quality results and pilot records remain intact.

## Recovered generation timing boundaries

The generation spans include inter-request pauses and recovery gaps, and exclude setup before first request and final export after last response. Some failed requests lack durations (24 structured HTTP errors; two Ling connection errors and one HTTP error), so their timing cannot be audited separately. All 164,900 output product IDs occur in timed request records. Overlapping durations must not be summed as elapsed generation time.

Observed peak timed-request overlap was 64 for structured generation and32 for Ling query generation; concurrency varied over the runs. Thus44.80min versus62.76min is an observed deployment comparison, not a controlled strategy-speed comparison. The union of timed intervals was41.25min and54.45min, respectively; that measures time with at least one timed request in flight, not complete pipeline active time. UTC starts plus monotonic durations reconstruct approximate response times; no continuous job timer was saved. See generation_timing_recovered.json and scripts/recover_generation_timing.py for hashes and calculations.
