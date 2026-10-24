# Enrichment within a $5 budget

Target: **164,900 distinct products in official English test candidates**.
Full generation and ranking evaluation are complete for all 164,900 test products. See `reports/enrichment_evaluation.md`; the total accounted spend including pilots is $1.9225503872.

## Tested option

Model: `inclusionai/ling-3.0-flash`, through the existing Vercel AI Gateway.
The live model endpoint quotes **$0.021 per million input tokens** and
**$0.063 per million output tokens**. Pilot response costs confirmed these rates.
The model's public landing page lists different standard prices; the saved live
pricing snapshot and measured charges are the evidence for this run.

The revised pilot covers all 200 previous products, with **$0.0018265338** in
reported generation charges including retries, plus a **$0.000068208** preliminary
probe. At this pilot average, full generation is roughly **$1.51**, including
a retry allowance proportional to observed pilot attempts. This is an estimate,
not a guaranteed quote; the quality sample is balanced rather than representative.

## Spending control

Each product has a 450-token estimated input cap, a 500-input-token cost reservation,
and a 220-token output cap. Eight products are generated per initial API request;
reasoning is disabled. Maximum initial reservations for 164,900 products are
**$4.1375** at the verified rate, including a 3% reservation margin.

A persistent shared SQLite ledger caps new-pipeline spending/reservations at
**$4.40**, including pilots, probes, retries and uncertain charges. Together with
the old Gemini pilot ($0.493229), the accounted ceiling is **$4.893229**.
No failed request is assumed free. Prices are rechecked at startup and every
minute; a price increase, missing billing, unexpected token usage or HTTP failure
stops new requests. The runner may stop with incomplete coverage instead of
exceeding the budget. The ledger must never be deleted to resume a run.

API pricing and reported billing remain provider-controlled. This local limit
reserves before dispatch and checks actual charges; it is not a provider-enforced
account spending cap. It assumes the observed tokenizer and listed capped token
rates remain applicable; unexpected charges trigger a stop, not a promise that
an external billing system cannot change. No new provider or account is required.

## Prompt and quality

The revised prompt removes minimum quotas, limits output to 0–8 keywords,
0–3 situations and 0–3 synonyms, and prohibits specifications inferred from
brand/model knowledge. Each batch uses IDs to verify product/output association.
Keywords are filtered to exact normalized phrases in the supplied product text.
Arrays are deterministically deduplicated/capped; invalid JSON and wrong product
IDs are rejected. Unresolved products get bounded individual retries.

Metadata context is shortened to meet the budget; exact per-product inputs,
raw outputs, removals and provider usage are saved. The title is initially capped
at 140 characters, brand at 40, colour at 30, and each description/bullet field
at 350, with further shortening when needed for the token limit. The BM25 original
product text remains complete; caps apply only to LLM context.

The same fixed 20-product audit flagged **9/20** outputs for grounding or synonym
concerns, versus 16/20 in the first pilot. This is a qualitative, single-reviewer
comparison of whole configurations, not evidence that the prompt alone caused
an improvement. Examples remaining: a hair net called a hair tie, cotton inferred
for a shirt, and tire compatibility expressed as a range instead of discrete sizes.
No retrieval improvement has yet been measured. A cheap pipeline is feasible;
semantic correctness is not guaranteed by valid JSON or source-span filtering.

## Reproduce

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_ling_tokenizer.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_budget_enrichment.py --scope test --plan-only
.venv/bin/python scripts/run_full_budget_enrichment.py --credentials-file /path/to/.env
```

The tokenizer files are pinned to inclusionAI/Ling-3.0-flash revision
`ef06d91fe382109ae82647da88ff99b0f11745b0` in `data/models/ling_tokenizer/`;
model weights are not needed. The full wrapper runs batches, then at most two
individual retry passes; it reports completion only when all 164,900 products
have saved valid enrichment. Output is `outputs/budget_enrichment/test/`.

Pilot outputs: `outputs/budget_enrichment/pilot/products_enriched.jsonl`.
Detailed audit: `reports/budget_enrichment_pilot_review.json`.
Prompt: `prompts/enrichment_english_v2.txt`.

Sources: [live model/pricing endpoint](https://ai-gateway.vercel.sh/v1/models),
[model page](https://vercel.com/ai-gateway/models/ling-3.0-flash),
[official tokenizer/model](https://huggingface.co/inclusionAI/Ling-3.0-flash).
