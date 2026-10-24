# Quality and total cost protocol

Goal: identify the quality/cost tradeoff for English product candidate ranking,
without training a model. Do not declare one method cheapest based only on API
charges or NDCG per dollar. Compare the Pareto frontier: methods offering better
quality at higher cost may suit different workloads.

## Methods and controls

Evaluate all 8,956 official test queries and their 181,701 candidate pairs over
the same 164,900 products. Keep label gains, metrics and deterministic ties fixed.
No prompts, retrieval parameters or hybrid weights selected on test results.

1. Plain BM25 (completed).
2. Structured Ling expansion + BM25, with field ablations (completed).
3. Pretrained doc2query-T5 + BM25. Pin checkpoint
   `castorini/doc2query-t5-base-msmarco` at
   `53926b5364548f0c81b39a31723184d3664aaf06`. Reuse the exact saved metadata
   inputs supplied to Ling; concatenate field values, cap input at 512 T5 tokens.
   Initial economical configuration: 3 sampled queries per product, top-k=10,
   temperature=1, maximum sequence length 64. This is a budget configuration,
   not a reproduction of the 40-query MS MARCO experiment. Preserve repeated
   predictions and report actual expansion length. Run a metadata-only hash
   sample solely to measure throughput before the full generation. It is not
   a reduced retrieval evaluation and no relevance labels enter generation.
   The full rerun uses batch size 16, selected using metadata-only throughput
   measurements; the earlier interrupted batch-4 output remains separate.
   Padding uses multiples of 128 with attention masks; the reusable MPS cache is
   cleared between batches to avoid accumulating GPU allocations.
4. Query-style expansion using the same Ling LLM, if the shared API budget
   permits: separates model choice from expansion structure.
5. Pretrained `intfloat/e5-small-v2` revision
   `ffb93f3bd4047442299a41ebb6fa998a38507c52` as the compact embedding baseline,
   followed by `intfloat/e5-base-v2` revision
   `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` for a larger representation if
   feasible. Use the required `passage: ` / `query: ` prefixes, masked mean
   pooling, L2 normalization and cosine similarity. Cap input at 512 model
   tokens, use float32 representations and original full metadata in the same
   field order as BM25, with the same HTML cleaning. No ESCI fine-tuning. Record
   truncation counts, batch size and hardware. These checkpoints are selected
   before evaluating their test scores.
   E5-small inference uses batch size 32 with stable input-length bucketing to
   reduce padding. Models run sequentially on the M4 GPU for uncontended timing.
   After a checkpointed MPS memory failure at 78,944 products, E5-small resumes
   with four padding widths (128/256/384/512) and GPU cache cleanup between batches.
   Input tokens, precision, pooling and model revision are unchanged. Preserve
   the execution note in `reports/e5_memory_recovery.json`.
6. Hybrid plain BM25 + dense and structured BM25 + dense, using equal-weight
   reciprocal rank fusion with k=60 across each complete candidate list.
   Do not fit weights on test queries. Equal score ties resolved by product ID.

All methods use the same original metadata availability. Record each model's
input truncation and representation size; different tokenizers do not imply
equal token counts. The original full metadata remains in the BM25 document.
T5 was trained on MS MARCO, not on these products: report domain mismatch.

## Cost accounting

Keep three measures separate:

- Actual cash billed: provider usage charges including retries and unknown-charge
  reservations. Local inference has zero API charges; electricity/hardware
  expenses remain unmeasured unless independently measured.
- Compute: wall time, process CPU time, device, batch size, precision, model
  revision, warm-up and setup time, input/output token counts, observed memory.
  CPU time does not measure GPU time or energy. Mark projections as estimates.
- Deployment cost scenarios: explicitly sourced hardware hourly prices multiplied
  by runtime *measured on that hardware*, or separately labelled electricity /
  amortization assumptions. Never multiply Mac throughput by a cloud GPU's hourly
  price and present that as measured cloud cost.

Report offline product processing, index construction, saved representation
bytes, query encoding, and online ranking/search separately. Embeddings encode
each product once but also encode every new query; LLM and T5 expansion move
neural inference offline. Hybrid carries the costs of both components.

Total cost for a specified workload = initial product processing + index build
+ changed-product processing + serving / storage over a stated period + query
processing for the stated query volume. Use e.g. 10,000 / 100,000 / 1,000,000
searches and 0% / 1% / 10% monthly catalog updates as *scenarios*, not observed
production workloads. Report setup both with and without cached model downloads.

The local existing BM25 evaluator is an effectiveness reference, not an optimized
search engine. Benchmark all methods under one consistent serving harness.
Time queries from raw text to ranked candidates with representations resident in
memory, including online tokenization / encoding but excluding dataset I/O and
metric calculation. Use fixed query order, one warm-up pass, three measured
passes, report p50/p95 and throughput. Cached query embeddings can be a separate
scenario, not the main dense latency number. Full-catalog search requires a
separate index experiment; ESCI candidate labels do not establish exhaustive
catalog recall. Do not compare candidate scoring latency against full-catalog
search latency.

## Budget and statistical reporting

The earlier shared $4.40 API ledger remains active; together with the Gemini
pilot it stays within the user's $5 expenditure limit. Local inference does not
add API charges and no paid compute instance is created by this protocol.
Report experiment cash spending separately from estimated operational costs.
The previous structured generation costs about $1.424 for the full products;
the approximately $1.923 total includes pilots and unknown-charge reservations.

Use NDCG@10 as primary effectiveness metric, paired query bootstrap differences
against plain BM25 and structured expansion, alongside full NDCG / Exact-at-1.
Additional comparisons are exploratory with multiple-comparison limitations.
Cost effectiveness conclusions must state model, hardware, corpus size,
query volume and catalog update assumptions. Do not claim a universal winner.

Sources: [docT5query implementation](https://github.com/castorini/docTTTTTquery),
[pinned checkpoint](https://huggingface.co/castorini/doc2query-t5-base-msmarco/tree/53926b5364548f0c81b39a31723184d3664aaf06),
[E5 encoding instructions](https://huggingface.co/intfloat/e5-small-v2).
