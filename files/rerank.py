"""Reranking that exploits the gallery BIJECTION.

Each split is N queries <-> N targets, one-to-one. Greedy per-query argmax lets two queries
both claim the same rank-1 target (one is wrong). A global optimal assignment (Hungarian) on
the score matrix resolves those collisions -> the true target moves up -> direct MRR lift.
This structural prior is our edge (no competitor research doc exploits it). Oracle-gated like
everything else: keep it only if it raises oracle MRR.

A "score matrix" S here is [Q, G] with HIGHER = more similar (convert a MIND distance D via S=-D).
"""
from __future__ import annotations
import numpy as np


def greedy_rankings(S) -> dict:
    """Per-query argsort (best->worst). Baseline (no bijection constraint)."""
    S = np.asarray(S, dtype=float)
    return {i: list(np.argsort(-S[i])) for i in range(S.shape[0])}


def hungarian_rankings(S) -> dict:
    """Optimal one-to-one assignment, then each query ranks its ASSIGNED target first and the
    rest by score. Returns {query_idx: [gallery_idx best->worst]}. Assumes a square bijection;
    for non-square it still assigns min(Q,G) and falls back to argmax for any unassigned query."""
    from scipy.optimize import linear_sum_assignment
    S = np.asarray(S, dtype=float)
    rows, cols = linear_sum_assignment(-S)            # maximize total similarity
    assign = {int(r): int(c) for r, c in zip(rows, cols)}
    out = {}
    for i in range(S.shape[0]):
        a = assign.get(i, int(np.argmax(S[i])))
        rest = [j for j in np.argsort(-S[i]).tolist() if j != a]
        out[i] = [a] + rest
    return out


def _mrr_idx(rankings: dict) -> float:
    """MRR when the true target of query i is gallery index i (oracle convention)."""
    total = 0.0
    for i, order in rankings.items():
        order = list(order)
        total += 1.0 / (order.index(i) + 1) if i in order else 0.0
    return total / max(len(rankings), 1)


if __name__ == "__main__":
    # Self-test: a constructed collision where greedy ties on a target but Hungarian resolves it.
    # 3 queries, true target = own index. Build S so q0 and q1 both peak at target 0.
    S = np.array([
        [0.90, 0.10, 0.05],   # q0: true=0, peaks at 0  (correct)
        [0.88, 0.80, 0.10],   # q1: true=1, but peaks at 0 (collision with q0) -> greedy puts true at rank2
        [0.10, 0.20, 0.95],   # q2: true=2, peaks at 2  (correct)
    ])
    g = _mrr_idx(greedy_rankings(S))
    h = _mrr_idx(hungarian_rankings(S))
    print(f"greedy   MRR = {g:.4f}")
    print(f"hungarian MRR = {h:.4f}")
    assert h > g, "Hungarian should resolve the collision and beat greedy here"
    print("SELF-TEST PASS — Hungarian assignment resolves the rank-1 collision (q1: rank2 -> rank1).")
