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


def _cosine_dist(A, B):
    """1 - cosine similarity. [n,d],[m,d] -> [n,m] distance (lower = more similar)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return 1.0 - A @ B.T


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """k-reciprocal re-ranking (Zhong, Zheng, Cao & Li, CVPR 2017). Unsupervised, no labels:
    encodes each sample's k-reciprocal neighbours and re-scores by Jaccard distance, blended
    with the original distance. Returns a reranked query x gallery distance (lower = better).
    k1/k2 auto-shrink for our small galleries (40-100)."""
    q_g_dist = np.asarray(q_g_dist, np.float32)
    Q, G = q_g_dist.shape
    k1 = max(2, min(k1, G - 1)); k2 = max(1, min(k2, G - 1))
    original_dist = np.concatenate([
        np.concatenate([np.asarray(q_q_dist, np.float32), q_g_dist], axis=1),
        np.concatenate([q_g_dist.T, np.asarray(g_g_dist, np.float32)], axis=1)], axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = (original_dist / (np.max(original_dist, axis=0, keepdims=True) + 1e-12)).T
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)
    all_num = original_dist.shape[0]
    for i in range(all_num):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        krec = fwd[np.where(bwd == i)[0]]
        krec_exp = krec
        for cand in krec:
            c_fwd = initial_rank[cand, :int(round(k1 / 2.)) + 1]
            c_bwd = initial_rank[c_fwd, :int(round(k1 / 2.)) + 1]
            c_krec = c_fwd[np.where(c_bwd == cand)[0]]
            if len(np.intersect1d(c_krec, krec)) > 2. / 3 * len(c_krec):
                krec_exp = np.append(krec_exp, c_krec)
        krec_exp = np.unique(krec_exp)
        w = np.exp(-original_dist[i, krec_exp])
        V[i, krec_exp] = w / (np.sum(w) + 1e-12)
    original_dist = original_dist[:Q]
    if k2 != 1:
        Vqe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            Vqe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = Vqe
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(Q):
        tmp = np.zeros((1, all_num), np.float32)
        nz = np.where(V[i] != 0)[0]
        imgs = [invIndex[ind] for ind in nz]
        for j in range(len(nz)):
            tmp[0, imgs[j]] += np.minimum(V[i, nz[j]], V[imgs[j], nz[j]])
        jaccard[i] = 1 - tmp / (2. - tmp)
    final = jaccard * (1 - lambda_value) + original_dist * lambda_value
    return final[:Q, Q:]


def rankings_from_dist(D) -> dict:
    """Distance matrix [Q,G] (lower=better) -> {q_idx: [g_idx best->worst]}."""
    D = np.asarray(D)
    return {i: list(np.argsort(D[i])) for i in range(D.shape[0])}


def kreciprocal_rankings(q_emb, g_emb, k1=20, k2=6, lambda_value=0.3) -> dict:
    """Embeddings -> k-reciprocal reranked rankings (computes the q-q / g-g / q-g blocks)."""
    return rankings_from_dist(re_ranking(_cosine_dist(q_emb, g_emb),
                                         _cosine_dist(q_emb, q_emb),
                                         _cosine_dist(g_emb, g_emb), k1, k2, lambda_value))


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

    # k-reciprocal: synthetic retrieval (true target i = a noisy view of patient i); must run + be valid
    rng = np.random.RandomState(0)
    N, d = 14, 16
    base = rng.randn(N, d)
    q_emb = base + 0.35 * rng.randn(N, d)
    g_emb = base + 0.35 * rng.randn(N, d)
    kr = kreciprocal_rankings(q_emb, g_emb, k1=6, k2=3, lambda_value=0.3)
    assert all(sorted(v) == list(range(N)) for v in kr.values()), "k-reciprocal must return valid permutations"
    base_mrr = _mrr_idx(greedy_rankings(-_cosine_dist(q_emb, g_emb)))
    print(f"k-reciprocal: baseline MRR={base_mrr:.4f}  reranked MRR={_mrr_idx(kr):.4f}")
    print("SELF-TEST PASS — k-reciprocal runs and returns valid rankings.")
