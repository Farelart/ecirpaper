import math
import unittest

from ecirpaper.metrics import evaluate_ranking, ndcg
from ecirpaper.retrieval import bm25_score, rank_candidates, tokenize, FIELDS


class EvaluationTests(unittest.TestCase):
    def test_missing_relevant_products_still_contribute_to_ideal(self):
        labels = {p: "E" for p in "ABCDE"}
        # A one-hit list cannot normalize itself to a perfect ranking at 10.
        expected = 1 / sum(1 / math.log2(i + 2) for i in range(5))
        self.assertAlmostEqual(ndcg(["A"], labels, 10), expected)
        with self.assertRaises(ValueError):
            evaluate_ranking(["A"], labels)

    def test_direct_esci_gains_and_full_ideal(self):
        labels = {"exact": "E", "substitute": "S", "complement": "C", "irrelevant": "I"}
        ideal = ["exact", "substitute", "complement", "irrelevant"]
        self.assertAlmostEqual(ndcg(ideal, labels), 1.0)
        ranking = ["substitute", "irrelevant", "exact", "complement"]
        expected = (.1 + 1 / math.log2(4) + .01 / math.log2(5)) / (
            1 + .1 / math.log2(3) + .01 / math.log2(4))
        self.assertAlmostEqual(ndcg(ranking, labels), expected)

    def test_reject_duplicate_or_unknown_candidates(self):
        for ranking in [["A", "A"], ["A", "unknown"]]:
            with self.assertRaises(ValueError):
                evaluate_ranking(ranking, {"A": "E", "B": "I"})

    def test_no_exact_matches_is_defined_without_relabeling(self):
        metrics = evaluate_ranking(["A", "B"], {"A": "S", "B": "I"})
        self.assertEqual(metrics["ndcg_full"], 1.0)
        self.assertEqual(metrics["mrr_exact"], 0.0)
        self.assertIsNone(metrics["recall_exact_at_10"])


class RetrievalTests(unittest.TestCase):
    def test_score_matches_hand_calculation(self):
        stats = {"documents": 3, "average_length": 4,
                 "document_frequency": {"rare": 1, "common": 3}}
        tokens = ["rare", "rare", "common", "padding"]
        expected = math.log(1 + 2.5 / 1.5) * (2 * 2.2) / (2 + 1.2)
        self.assertAlmostEqual(bm25_score(["rare"], tokens, stats), expected)
        self.assertGreater(bm25_score(["rare"], tokens, stats), bm25_score(["common"], tokens, stats))
        self.assertEqual(bm25_score(["absent"], tokens, stats), 0)

    def test_longer_documents_receive_length_normalization(self):
        stats = {"documents": 2, "average_length": 5, "document_frequency": {"target": 2}}
        self.assertGreater(bm25_score(["target"], ["target"], stats),
                           bm25_score(["target"], ["target"] + ["padding"] * 8, stats))

    def test_ties_ignore_labels_and_input_order(self):
        candidates = [{**{field: "" for field in FIELDS}, "product_id": p, "esci_label": label}
                      for p, label in [("B", "E"), ("A", "I")]]
        stats = {"documents": 2, "average_length": 0, "document_frequency": {}}
        self.assertEqual(rank_candidates("absent", candidates, stats), [("A", 0), ("B", 0)])

    def test_analyzer_handles_html_case_and_numbers(self):
        self.assertEqual(tokenize("<b>750 ML</b> Stainless-Steel &amp; Blue"),
                         ["750", "ml", "stainless", "steel", "blue"])


if __name__ == "__main__":
    unittest.main()
