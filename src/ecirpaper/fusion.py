"""Fixed rank fusion for complete candidate lists."""
def reciprocal_rank_fusion(first, second, k=60):
    if k <= 0: raise ValueError('Positive k required')
    if len(first) != len(set(first)) or len(second) != len(set(second)):
        raise ValueError('Duplicate candidate')
    if set(first) != set(second): raise ValueError('Candidate sets must match')
    scores = {pid: 1/(k+rank) for rank,pid in enumerate(first,1)}
    for rank,pid in enumerate(second,1): scores[pid] += 1/(k+rank)
    return sorted(scores.items(),key=lambda item:(-item[1],item[0]))
