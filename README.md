# Structured offline product expansion on Amazon ESCI

## Cost effectiveness extension

The next comparisons include pretrained doc2query-T5, dense embeddings and
hybrid ranking. See [the cost protocol](reports/cost_effectiveness_protocol.md)
for separate API spending, compute, indexing and online serving measurements.
Local inference has no API bill; its electricity and hardware costs are not zero.

Keep optional neural dependencies separate from the original generation runtime:

```sh
python3 -m venv .venv-neural
.venv-neural/bin/python -m pip install -r requirements-neural.txt
.venv-neural/bin/python scripts/run_doc2query_t5.py
```

This command measures throughput on 32 metadata-selected products, without
evaluating retrieval quality. All quality comparisons still use 8,956 queries.
For resumable full generation:

```sh
.venv-neural/bin/python scripts/run_doc2query_t5.py --limit 0 --output outputs/doc2query_t5/full
```

The 3-query T5 setting is explicitly an economical configuration, not a
reproduction of the original 40-query MS MARCO run. Outputs record the pinned
model revision, input hash, device, generation settings, wall/CPU time and
observed process memory. No model is trained or fine-tuned.

Run the full dense and hybrid comparison, then all T5 products and evaluation:

```sh
.venv/bin/python scripts/prepare_neural_inputs.py
.venv-neural/bin/python scripts/run_remaining_experiments.py
```

Download the pinned E5-small checkpoint files before the local-only encoder
runs. `reports/remaining_experiments_status.json` records the current stage;
stage logs are in `outputs/remaining_experiments/`. Encoding and T5 generation
save resumable checkpoints. Re-running the pipeline reuses completed stage
artifacts and resumes unfinished inference. Evaluations refuse to overwrite
existing result directories. A failed evaluation must be inspected and its
partial results preserved before choosing a fresh output directory.

This repository investigates whether LLM-generated keywords, synonyms, and
usage situations improve lexical product ranking. The first step is to inspect
the official ESCI release before finalizing the experimental protocol.

## Data

Official source: https://github.com/amazon-science/esci-data

The download script pins an upstream revision and checks the two Parquet files
against their Git LFS SHA-256 checksums. Large raw data and local environments
are excluded from Git. Provenance is recorded in `data/esci_manifest.json`.
The upstream README, license, notice, and ranking evaluation preparation script
are preserved under `data/source/`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_esci.py
.venv/bin/python scripts/inspect_esci.py
```

## Proposed first experiment

Use the English (US), small-version candidate-ranking task. Score each query's
official candidates with one shared BM25 index, comparing identical source
product fields with and without document-only expansion. Generate expansions
from product metadata only, without queries or relevance labels. Preserve
keywords, synonyms, and situations separately for component ablations.

Choose a development subset from training query IDs; reserve official test
query IDs for final evaluation. Fix the candidate lists, label gains, metric
definitions, and expansion budgets before evaluating variants. Compare structured
expansion with query-style expansion and suitable retrieval baselines.

ESCI candidate ranking does not provide exhaustive relevance judgments for
whole-catalog retrieval, and it does not evaluate multi-turn conversation.
This protocol is provisional, pending the dataset inspection and pilot.

## Joined English ranking dataset

Build the experiment-ready metadata join after downloading the official files:

```sh
.venv/bin/python scripts/build_esci_dataset.py
```

The builder writes `data/processed/esci_us_small/`:

| File | Contents |
| --- | --- |
| `train.parquet` | 419,653 labeled query-product pairs across 20,888 official training-split queries, with product metadata |
| `test.parquet` | 181,701 labeled query-product pairs across 8,956 official test-split queries, with product metadata |
| `products.parquet` | 482,105 distinct candidate products, containing metadata only, for indexing and document enrichment |
| `queries.parquet` | 29,844 queries with official split, source category, and candidate count |
| `manifest.json` | Source revision/checksums, output checksums, schema, counts, and validation results |

Each joined row contains `example_id`, `query_id`, `query`, `product_id`,
`product_locale`, `esci_label`, `split`, `source`, `product_title`,
`product_description`, `product_bullet_point`, `product_brand`, and
`product_color`. Products are joined by `(product_id, product_locale)`.
Original text, missing metadata, and ESCI labels are preserved. These are
Amazon's split names; building this dataset does not train a model.

Parquet row order is unspecified and is not a retrieval ranking. Group rows by
`query_id` to get the full candidate list; rank those candidates with each
algorithm. Do not use the relevance label to initialize candidate order or
resolve score ties. Evaluation uses direct relevance gains: E=1, S=0.1, C=0.01, I=0, following the ESCI paper.

The builder verifies raw-file checksums, unique keys, absence of cross-split
query IDs, complete metadata joins, and exact preservation of official candidate
pairs and labels in the saved files. It builds atomically and refuses to
silently overwrite an existing output folder. Use `--output-dir` to choose a
new location when rebuilding.

See `reports/esci_us_small_build.json` for the build record and
`reports/esci_us_small_training_samples.json` for three complete query-to-
candidate examples. Those examples show training-split queries only and list
candidates by product ID.

Example local read (from the repository root):

```python
import duckdb
connection = duckdb.connect()
rows = connection.execute(
    "SELECT query, product_id, product_title, esci_label "
    "FROM read_parquet('data/processed/esci_us_small/train.parquet') "
    "WHERE query_id = ? ORDER BY product_id",
    [1],
).fetchall()
```

## Plain BM25 reference

Run the verification checks, then evaluate the fixed baseline:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_bm25.py --split test
```

The baseline concatenates title, description, bullet points, brand and colour,
with BM25 k1=1.2 and b=0.75. It uses Unicode alphanumeric tokens, case folding
and HTML cleanup, without stemming, stopword removal or text truncation.
Document frequencies and average length come from all 482,105 product records;
queries and relevance labels do not enter these statistics. No model is trained.

Every supplied candidate is scored. Ties use product ID, independently of labels.
The evaluator calculates NDCG@10 and full-ranking NDCG against all supplied
judgments, plus Exact MRR, Exact hit rate at rank 1 and Exact recall@10.
Query bootstrap intervals are reported for the two NDCG metrics. Future method
comparisons should use paired per-query differences with the same queries.

Rankings, per-query scores and run metadata are saved to `outputs/bm25_test/`;
a compact report is saved to `reports/bm25_test.md` and `.json`. Large outputs
and the verified corpus-statistics cache are ignored by Git. Existing run
folders are preserved; use `--output-dir` for a new run location.

This is an untuned candidate-ranking reference. These processing times do not
measure deployed search latency, and the experiment does not measure retrieval
from the entire catalogue or multi-turn conversation.

## Enrichment quality pilot

A 200-product metadata-only pilot is complete. See `reports/enrichment_pilot.md`
for generation cost and the fixed 20-product qualitative review, and
`reports/enrichment_pilot_setup.md` for sampling and prompt adaptations.
All structural checks passed; grounding and synonym issues remain. This pilot
does not measure ranking improvement. The version-1 outputs are preserved in
`outputs/enrichment_pilot/`.

## Budget enrichment

The revised Ling pilot and runnable full-test-product pipeline are described in
`reports/budget_enrichment.md`. The shared budget ledger limits the new pipeline
to $4.40, leaving the previous $0.493229 pilot within a combined $5 envelope.
The 200-product revised pilot is complete; generation and evaluation for all 164,900 test products are complete.
Read the quality review before scaling: some synonym and grounding concerns remain.

## Full experiment results

All 8,956 official English test queries were evaluated across plain BM25 and
four enrichment variants. See `reports/enrichment_evaluation.md` for results,
paired uncertainty and limitations. Combined enrichment raises NDCG@10 from
0.7009 to 0.7131 on the same 164,900-product corpus. The total accounted spend,
including pilots, is $1.9225503872. Full rankings and per-query scores are in
`outputs/enrichment_evaluation/`; generated product records are in
`outputs/budget_enrichment/test/products_enriched.jsonl`.
