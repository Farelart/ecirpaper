# Completed timing benchmark

All eight full-corpus CPU retrieval benchmarks and index measurements are now complete and independently checked. Current results and scope are in final_experiments_verified.md and retrieval_benchmark.json. Neural query encoding is included; network serving remains unmeasured.

## Historical measurement notes

# Timing measurement status

Updated 2026-10-05. Preparation, offline evaluation and online serving must be reported separately.

| Method or phase | Recorded measurement | Limitation |
| --- | --- | --- |
| Structured LLM expansion | Per-request elapsed seconds in `outputs/budget_enrichment/test/batches.jsonl`; request timestamps and retry status | Requests ran concurrently with several worker settings and checkpoint resumes. Summed request durations are not full-run wall time. No single uninterrupted preparation timer was recorded. |
| E5 product encoding | 2,191.880882013 seconds of recorded model inference for all 164,900 products | About 36.5 minutes; excludes loading, tokenization, I/O and unsuccessful execution. This is offline batched inference. |
| E5 query encoding | 50.708143886 seconds of recorded model inference for all 8,956 queries | Offline batches; cannot divide by query count and call that online latency. |
| E5 candidate ranking | 0.144938568 seconds for candidate vector selection, dot products and sorting across 8,956 queries | Precomputed vectors and supplied candidate lists; excludes query encoding, whole-catalog retrieval, metrics and I/O. |
| Plain and structured BM25 evaluation | Effectiveness results complete | No comparable per-method scoring or corpus preparation timers were recorded in the completed field-ablation evaluator. |
| Hybrid evaluation | Pipeline stage duration recorded | Separate fusion timers were not recorded. A shared evaluation stage includes multiple methods and cannot be attributed to one hybrid. |
| doc2query-T5 generation | Per-batch inference wall time and CPU time; model-load and warmup timing; final summary totals automatically | Still running. Completed batch timings are retained across resumes. Batch time excludes some checkpoint writing and other orchestration work. |
| doc2query-T5 evaluation | Timers added before the pending stage: corpus statistics, candidate scoring/sorting, ranking loop and evaluation wall time | No measurements yet. Scoring includes candidate document tokenization. These phases are not an online serving benchmark. |

The pending T5 evaluator was syntax checked after timing instrumentation. Its ranking logic and model settings are unchanged.

No method has a measured comparable online serving benchmark here. A separate controlled benchmark is required for a speed claim: same hardware, same retrieval task and corpus, consistent index preparation, query encoding included for neural methods, warmup disclosed, repeated queries, and median/p95 latency. Record index construction time and stored representation size separately. Do not infer a serving speed winner from the offline evaluation timers.

Preserve existing effectiveness rankings. Missing completed-phase timings require a separate timing run rather than invented retrospective values. Any such run during ongoing T5 inference must disclose resource contention or wait until inference finishes.

## Interrupted elapsed time during T5 generation

At the 2026-10-05 17:38 UTC progress check, T5 was still running with 71,472 products checkpointed, compared with 70,496 at 16:39 UTC. macOS power logs recorded repeated sleep/wake events; the latest wake at 18:38:47 Africa/Porto-Novo reported battery charge 1%. These interruptions make elapsed-time extrapolation unreliable. Do not attribute the paused elapsed time solely to T5 computation or use it as a controlled cross-method runtime comparison. Preserve batch timings and checkpoints; no duplicate inference was launched.
