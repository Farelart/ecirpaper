"""Exact BM25 scoring with corpus-wide document frequencies.

This candidate-ranking implementation needs no stored inverted index: corpus
statistics are computed once, and only the official candidate documents are
scored. It is not a whole-catalog retrieval or serving-latency benchmark.
"""
import html
import math
import re
from collections import Counter

FIELDS = ("product_title", "product_description", "product_bullet_point",
          "product_brand", "product_color")
ANALYZER = "unicode_alphanumeric_casefold_html_v1"
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
TAG = re.compile(r"</?[A-Za-z][^>]*>")


def tokenize(text):
    return TOKEN.findall(TAG.sub(" ", html.unescape(text or "")).casefold())


def document_text(values):
    return " ".join(value or "" for value in values)


def bm25_score(query_terms, tokens, statistics, k1=1.2, b=0.75):
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("Require k1 > 0 and 0 <= b <= 1")
    if not tokens or not statistics["average_length"]:
        return 0.0
    counts = Counter(tokens)
    normalization = k1 * (1 - b + b * len(tokens) / statistics["average_length"])
    total = 0.0
    n = statistics["documents"]
    for term in sorted(set(query_terms)):
        frequency = counts.get(term, 0)
        if frequency:
            df = statistics["document_frequency"].get(term, 0)
            if not 0 < df <= n:
                raise ValueError(f"Invalid corpus document frequency for {term!r}")
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            total += idf * frequency * (k1 + 1) / (frequency + normalization)
    return total


def rank_candidates(query, candidates, statistics, k1=1.2, b=0.75):
    terms = tokenize(query)
    scored = [(item["product_id"], bm25_score(terms, tokenize(document_text(
        item[field] for field in FIELDS)), statistics, k1, b)) for item in candidates]
    return sorted(scored, key=lambda item: (-item[1], item[0]))
