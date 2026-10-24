# Product enrichment pilot

The locally prepared sample contains 200 distinct products: 50 in each of four
metadata strata (description and bullets present; description only; bullets only;
neither). Selection uses fixed seed 42 and product metadata availability, without
relevance labels or query text. All products appear in the official exploration
(train) split, and products also appearing in test candidates are excluded.
This balanced sample checks failure modes; it is not a prevalence-weighted sample
and is not a retrieval evaluation.

The English prompt adapts the current Qualiwo three-field enrichment: keywords,
situations and synonyms. It keeps the original temperature 0.4 and 1,500 output
token cap. It adds explicit grounding instructions and permits fewer entries when
supported alternatives are scarce. ESCI metadata fields replace the original
merchant attributes/categories. These changes must be disclosed in the paper.

Generation input includes title, description (first 4,000 characters), bullet
points (first 2,000 characters), brand and colour. Original metadata and truncation
flags are retained in the sample. The baseline continues to score full product
text; these caps apply only to generation context.

Selected model: `google/gemini-3-flash`, matching the current enrichment code.
Destination: `https://ai-gateway.vercel.sh/v1/chat/completions`.
No query strings or relevance judgments are sent to the model. Credentials are
read at runtime, never copied into the experiment repository or result files.

Files:
- `prompts/enrichment_english_v1.txt`: exact English prompt.
- `outputs/enrichment_pilot/products.jsonl`: originals and exact generation inputs.
- `outputs/enrichment_pilot/manifest.json`: sample/prompt hashes and configuration.
- `scripts/run_enrichment_pilot.py`: preparation and bounded generation runner.

Prepare without API calls:

```sh
.venv/bin/python scripts/run_enrichment_pilot.py --prepare-only
```

Generate using `AI_GATEWAY_API_KEY2` or `AI_GATEWAY_API_KEY` in the environment,
or an explicitly selected credentials file:

```sh
.venv/bin/python scripts/run_enrichment_pilot.py --credentials-file /path/to/.env
```

The runner checks one response before launching four concurrent requests. It
logs raw outputs, schema validation status, response model, finish reason and
provider token usage. It resumes without repeating saved attempts and does not
silently switch models. Semantic grounding needs manual review after generation;
valid JSON alone is not evidence of faithful enrichment.

Provider references:
[REST API](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions/rest-api),
[model](https://vercel.com/ai-gateway/models/gemini-3-flash).
