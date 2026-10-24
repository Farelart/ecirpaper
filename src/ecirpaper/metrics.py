"""Candidate-ranking metrics with fixed, complete relevance judgments.

Direct gains follow ESCI paper section 3.1: E=1, S=.1, C=.01, I=0.
They must not be exponentiated or inferred from the returned ranking.
"""
import math

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def ndcg(ranking, judgments, k=None):
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate product in ranking")
    if any(product not in judgments for product in ranking):
        raise ValueError("Unjudged candidate in ranking")
    if any(label not in GAINS for label in judgments.values()):
        raise ValueError("Invalid ESCI label")
    cutoff = len(judgments) if k is None else k
    if cutoff <= 0:
        raise ValueError("Cutoff must be positive")
    ideal = sorted((GAINS[label] for label in judgments.values()), reverse=True)[:cutoff]
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    if not idcg:
        raise ValueError("NDCG undefined: all candidates have zero gain")
    dcg = sum(GAINS[judgments[product]] / math.log2(i + 2)
              for i, product in enumerate(ranking[:cutoff]))
    return dcg / idcg


def evaluate_ranking(ranking, judgments):
    if len(ranking) != len(judgments) or set(ranking) != set(judgments):
        raise ValueError("Ranking must contain every official candidate exactly once")
    exact_positions = [i + 1 for i, product in enumerate(ranking) if judgments[product] == "E"]
    return {
        "ndcg_at_10": ndcg(ranking, judgments, 10),
        "ndcg_full": ndcg(ranking, judgments),
        "mrr_exact": 1 / exact_positions[0] if exact_positions else 0.0,
        "exact_at_1": float(bool(exact_positions) and exact_positions[0] == 1),
        "recall_exact_at_10": (
            sum(position <= 10 for position in exact_positions) / len(exact_positions)
            if exact_positions else None
        ),
    }
