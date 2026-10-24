# Completed product-enrichment pilot

200 products generated with Gemini 3 Flash. All 200 passed structural validation; no API failures.
Provider-reported total cost: **$0.493229**. Input: 102,376 tokens; output including reasoning: 147,347 tokens.
Mean API request duration: 5.32 seconds with four concurrent workers; this is generation time, not search latency.

## Qualitative review

A fixed sample of 20 products (five per metadata stratum) was inspected against the exact generation inputs. **16/20** had at least one grounding or synonym concern. This is an exploratory single-reviewer flag count, including ambiguous synonyms; it is not a validated hallucination rate. The sample is balanced by metadata availability, not population-weighted.

| Product | Concern |
| --- | --- |
| B07ZKJB8GF | Connectors, cable construction and stated device connections are grounded in metadata; uses are plausible. |
| B07DCJ3GLT | Outdoor awning and sun tarp may imply different construction or waterproofness; source explicitly says not waterproof and contains conflicting shape/hardware statements. |
| B0831S1PX9 | Ray tracing capable laptop is not explicit in supplied text, although inferred from the named GPU; keep capability claims source-grounded. |
| B07L2TLCHM | Stackable, rustproof and matte finish are absent from supplied metadata. |
| B07JZDNWQT | Ultrasound, non-abrasive cleaning, braces and implants are stated; source medical/marketing claims are not independently verified by this audit. |
| B07Z83JD3N | Metal, steel and 1/5 scale are absent from supplied metadata. |
| B00N41RRRQ | Elastic, mesh, breathable and equestrian-specific context are not provided in title or short description. |
| B085BL4PCY | Plot, author, series and e-novella format are stated; mystery/thriller uses are plausible. |
| B006RODC1M | Low-light and evergreen go beyond supplied text; avoiding direct sun alone does not establish low-light tolerance. |
| B01FL2T3ZE | Breast shield can denote a pump flange, a different product; nursing shield is more precise. |
| B07MN6G7VX | Heavy cotton, relaxed fit, pre-shrunk, crew neck and breathable are not stated in supplied metadata. |
| B002XDRCY4 | Protein, vitamins, recovery and meal replacement are stated in source; this audit does not validate manufacturer health claims. |
| B07W3SH4BD | Team bib, jersey bib and kitchen bib are ambiguous and can denote a different product; other uses are grounded or plausible. |
| B004BTE78C | Steel and rollers are absent from supplied metadata. Winter traction situations are plausible. |
| B00XA30QI0 | Outdoor suitability is not established by supplied metadata; beginner use is plausible but not a certified suitability claim. |
| 1623970180 | Title-only record does not establish manuscript, 18th-century recipes or authentic colonial content. |
| B0851BRKTY | Shatterproof and impact protection go beyond the supplied title; hardness and tempered glass do not guarantee shatterproofness. |
| B07F8BMP7Q | Waterproof, insulated, synthetic, heel lock and all-mountain suitability are absent from the supplied title. |
| 1418542563 | Men-specific curriculum, masculinity, leadership and discipleship are absent from the title-only input. |
| 1465418318 | Photography, picture-book and encyclopedia formats are not established by title/brand; shark education uses are plausible. |

## Implication

The pipeline works, but the prompt still fills near-target entry counts even with sparse evidence. The next prompt should remove minimum quotas, require specifications to be explicitly present, separate plausible uses from factual properties, and avoid broad alternative product names. Do not quietly filter outputs using relevance judgments. Preserve this pilot as version 1.
No retrieval lift has been measured by this product-only pilot. A subsequent ranking experiment must enrich all candidates for its selected queries, using a fixed prompt and scoring protocol.
At the observed balanced-sample average, 164,900 products would cost roughly **$407**. This is a rough extrapolation, not a representative workload estimate or quote; token lengths, caching and retries can change it. Full generation has not been started.

Full originals/inputs: `outputs/enrichment_pilot/products.jsonl`. Full generation records: `outputs/enrichment_pilot/generations.jsonl`. Fixed review packet: `outputs/enrichment_pilot/audit_packet.json`. Detailed annotations: `reports/enrichment_pilot_review.json`.
Pilot setup and prompt adaptations: `reports/enrichment_pilot_setup.md`.
