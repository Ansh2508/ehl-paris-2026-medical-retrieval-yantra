"""Per-level rank fusion of Branch A (embedding) and Branch B (MIND).
Rank fusion (not raw-score mixing) so different score scales combine cleanly."""
from __future__ import annotations
from typing import Sequence


def _normalized_rank_scores(ranking: Sequence[str]) -> dict[str, float]:
    n = len(ranking)
    if n <= 1:
        return {t: 1.0 for t in ranking}
    return {tid: (n - 1 - i) / (n - 1) for i, tid in enumerate(ranking)}


def rank_fusion(rankings: list[Sequence[str]], weights: list[float]) -> list[str]:
    """Weighted fusion of several rankings over the SAME gallery.
    Returns the fused ranking best->worst."""
    agg: dict[str, float] = {}
    for r, w in zip(rankings, weights):
        for tid, s in _normalized_rank_scores(r).items():
            agg[tid] = agg.get(tid, 0.0) + w * s
    return sorted(agg, key=lambda t: -agg[t])


def fuse_per_level(ranking_a: Sequence[str], ranking_b: Sequence[str],
                   level: str, fusion_w: dict) -> list[str]:
    """level in {'l1','l2','l3'}. w = weight on Branch A; (1-w) on Branch B.
    Recall: B(MIND) ~useless on L2, anchor on L1 & L3 — so w is high on L2."""
    w = fusion_w[level]
    return rank_fusion([ranking_a, ranking_b], [w, 1.0 - w])
