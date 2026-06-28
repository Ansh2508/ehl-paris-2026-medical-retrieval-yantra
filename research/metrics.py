"""Evaluation = Mean Reciprocal Rank. The number that decides every submit."""
from __future__ import annotations
from typing import Hashable, Sequence, Mapping


def reciprocal_rank(ranking: Sequence[Hashable], true_id: Hashable) -> float:
    """RR = 1/rank of the true target; 0 if absent. rank is 1-indexed."""
    try:
        return 1.0 / (list(ranking).index(true_id) + 1)
    except ValueError:
        return 0.0


def mrr(rankings: Mapping[Hashable, Sequence[Hashable]],
        truth: Mapping[Hashable, Hashable]) -> float:
    """Mean RR over all queries in `truth`. rankings: qid -> ranked target ids."""
    if not truth:
        return 0.0
    total = sum(reciprocal_rank(rankings.get(q, []), t) for q, t in truth.items())
    return total / len(truth)
