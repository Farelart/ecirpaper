# Verified common Modal T4 comparison

All eight configurations have independently verified complete candidate rankings and three retrieval timing passes over all8,956 queries. Corpus:164,900 products;181,701 judged pairs. No ESCI-specific training or tuning.

The primary comparison now uses T4 for all neural preparation and a single Modal T4 host for all retrieval methods. Earlier local results remain preserved. Ling generation remains an API measurement.

| Method | NDCG@10 | Preparation min | Index s | Combined min | Index MiB | Median/p95 ms | Preparation cost |
|---|---:|---:|---:|---:|---:|---:|---|
| Plain BM25 | 0.7009 | 0.00 | 33.76 | 0.56 | 107.46 | 4.80/6.59 | No generation; CPU cost unallocated |
| Structured Ling + BM25 | 0.7131 | 44.80 | 36.71 | 45.41 | 113.84 | 4.75/6.53 | API $1.424037 |
| T5 queries + BM25 | 0.7091 | 119.95 | 35.71 | 120.54 | 113.63 | 4.74/6.53 | T4 compute $1.607373 |
| Ling queries + BM25 | 0.7150 | 62.76 | 36.09 | 63.36 | 110.50 | 4.77/6.51 | API $0.995010 |
| E5-small | 0.7338 | 24.32 | 31.57 | 24.85 | 241.55 | 7.82/8.54 | T4 compute $0.36649437 |
| SPLADE | 0.7368 | 109.36 | 46.92 | 110.14 | 149.80 | 9.31/10.15 | T4 compute $1.522311 |
| Plain BM25 + E5 | 0.7259 | 24.32 | 65.33 | 25.41 | 349.02 | 18.49/24.48 | E5 component $0.36649437 |
| Structured BM25 + E5 | 0.7325 | 69.12 | 68.27 | 70.26 | 355.39 | 18.23/24.02 | API $1.424037 + E5 $0.36649437 |

## What the measurements support

Structured expansion improves plain BM25 and this three-query T5 configuration. The matched Ling generated-query control ranks slightly better and has lower logged API charges. E5 has higher quality than structured expansion, lower recorded preparation time and lower reported preparation compute cost in these deployments. Structured BM25 has lower retrieval latency. SPLADE has the highest mean quality, but its small advantage over E5 is not reliably separated by the exploratory paired interval.

The common-host medians narrow the structured/E5 search-time gap to approximately1.65x:4.75ms versus7.82ms. The previous approximately15x local CPU comparison must not be used as the primary deployment-speed claim. Small latency differences between BM25 variants should not be interpreted as a reliable ordering.

## Timing boundaries and deployment costs

- Neural preparation uses the same T4 resource class and CPU/memory allocation across distinct preparation jobs; retrieval methods all ran sequentially in one common T4 invocation.
- BM25 uses CPU postings; neural query encoding and exact dot products use CUDA; GPU-to-CPU score transfer and deterministic CPU top10 selection included. No network serving or ANN.
- Preparation sums recorded phases: Ling spans include gaps/recovery, neural batch timers exclude checkpoint/loading overhead. Hybrid sums are component sums, not measured concurrent elapsed time.
- Index construction includes lexical tokenization separately, or neural representation assembly/serialization/GPU transfer. Index sizes exclude shared metadata/IDs and model weights.
- Frozen lexical token counts preserve earlier scoring. Expansion-field order changes do not change BM25; one T5 tokenization runtime discrepancy is frozen rather than applied.
- Quality ranks judged candidates. Hybrid quality uses full-candidate RRF, timed hybrids use1000 hits per component with k60.
- Ling API generation is a separate deployment measure, not T4-hosted. Model/input/output allowance matched between the two Ling strategies, but observed concurrency differs.
- Shortened metadata for expansion versus full512token inputs for E5/SPLADE limits cross-family attribution. No ESCI fine-tuning, no universal cheapest-method or accuracy claim.

Model loading, checkpoint I/O, full setup/final export, storage and network serving charges are not included consistently in preparation phase sums. These columns therefore do not measure complete end-to-end pipeline wall time. API generation records do include pauses between first request and last reconstructed response. Do not compare them as intrinsic model throughput.

Reported experiment-app usage, including all reported prior pilots and failed attempts: **$4.01109904** including egress, below the$30 ceiling. E5 preparation app: compute$0.36649437, total$0.36682888. Common eight-method benchmark app: compute$0.48260798, total$0.48311671. The common benchmark bill covers all indexes, evaluations and timing passes; it is not a per-method serving-cost estimate.

The workspace monthly summary reports$4.29770780 ephemeral-app usage and$0 payable after credits/allowances; it has broader scope than the experiment app list. API shared ledger remains$2.424933 of its unchanged$4.40 ceiling, including uncertain reservations. Earlier outside-ledger Gemini pilot records remain separate.

## Paired candidate-quality differences

- structured_minus_plain: mean ΔNDCG@10 +0.01215064;95% interval [+0.01028948,+0.01401513].
- structured_minus_doc2query_t5: mean ΔNDCG@10 +0.00391836;95% interval [+0.00206524,+0.00577098].
- structured_minus_ling_doc2query: mean ΔNDCG@10 -0.00199668;95% interval [-0.00355306,-0.00023442].
- e5_minus_structured: mean ΔNDCG@10 +0.02074733;95% interval [+0.01691355,+0.02435902].
- splade_minus_e5: mean ΔNDCG@10 +0.00301424;95% interval [-0.00015357,+0.00607836].

Exploratory bootstrap:1,000 query resamples, seed42, standard-library RNG; no multiple-comparison correction.

## Validation

Independent SQL confirms every ranking has181,701 rows,8,956 queries, complete unique query-product coverage, valid ranks and matching NDCG. Each method has26,868 finite timing rows with complete query coverage per repetition; median/p95 independently recomputed.

All E5 checkpoint vectors independently verified:164,900 products,8,956 queries,384 dimensions,float32, finite values, unit L2 norms, pinned revision and identical input hashes to SPLADE. CUDA dense/sparse scores were independently checked against CPU dot products on21 full-corpus queries before timing. SPLADE checkpoints and pins had already passed complete coverage/numerical validation.

The downloader took longer than expected. A reserved SQL alias in the verification script was corrected; saved artifacts passed without any inference rerun. All earlier local outputs and cloud preparation records are preserved.

Sources:modal_common_t4_verified.json;modal_common_t4_validation.json;modal_common_t4_paired_comparisons.json;e5_t4_encoding_validation.json;modal_common_t4_tokenization_audit.json;generation_timing_recovered.json.

T5 provenance: effectiveness and lexical index construction use the preserved local MPS-generated queries; the T4 rerun measures preparation with the same pinned sampling configuration. Its combined phase total is therefore not timing of one artifact-identical end-to-end run.
