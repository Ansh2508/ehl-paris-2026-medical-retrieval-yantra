"""⭐ THE KEYSTONE. Build offline L1/L2/L3 evaluation from the 350 labelled d1 pairs
by SYNTHESIZING fake-L2 and fake-L3, so fusion weights and the veto are tuned on
real numbers instead of guesses. See CLAUDE.md §6.

Run a no-data sanity check:  python oracle.py --self-test
"""
from __future__ import annotations
import random
from typing import Callable, Sequence
import torch

from config import CFG
from metrics import mrr
import augment

# A rank_fn ranks a gallery (list of volumes) for a query volume, returning gallery
# indices best->worst. Works for MIND (mind_rank_fn) and the encoder (cosine) alike.
RankFn = Callable[[torch.Tensor, Sequence[torch.Tensor]], list[int]]

LEVELS = ("l1", "l2", "l3")


def make_holdout(pairs: list[dict], n: int = CFG.holdout_pairs, seed: int = CFG.seed):
    """pairs: the 350 labelled d1 pairs (each with preprocessed query+target tensors).
    Return (train_pairs, holdout_pairs). IMPORTANT: train the encoder on train_pairs
    and tune w/thresholds on holdout_pairs — no double-dipping."""
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    hold = {i for i in idx[:n]}
    return [p for i, p in enumerate(pairs) if i not in hold], [pairs[i] for i in idx[:n]]


def _apply(transform, vol: torch.Tensor) -> torch.Tensor:
    """Apply a MONAI dict-transform to one volume tensor [1,R,R,R]."""
    return transform({"image": vol})["image"]


def build_level(holdout: list[dict], level: str, cfg=CFG):
    """Return (query_vols, target_vols) for a level. Index i is the matching pair.
    l1: identity. l2: independent l2_transforms on q and target. l3: l3_transforms.
    (l2/l3 import MONAI lazily via augment.py — they raise if MONAI is absent.)"""
    if level == "l1":
        return [p["query"] for p in holdout], [p["target"] for p in holdout]
    tf = augment.l2_transforms(cfg) if level == "l2" else augment.l3_transforms(cfg)
    # independent draws for query and target (they must NOT share the transform)
    q = [_apply(tf, p["query"]) for p in holdout]
    g = [_apply(tf, p["target"]) for p in holdout]
    return q, g


def evaluate(rank_fn: RankFn, holdout: list[dict], cfg=CFG, levels=LEVELS) -> dict:
    """Offline MRR per level + mean. THE number that gates every change/submit.
    A level whose synthesis fails (e.g. MONAI missing locally) is skipped with a
    note rather than crashing the run; `mean` averages the levels that ran."""
    out: dict = {}
    for level in levels:
        try:
            q_vols, g_vols = build_level(holdout, level, cfg)
        except Exception as e:  # e.g. MONAI not installed -> l2/l3 skipped locally
            out[level] = None
            print(f"[oracle] skipped {level}: {type(e).__name__}: {e}")
            continue
        rankings, truth = {}, {}
        for i, qv in enumerate(q_vols):
            rankings[i] = rank_fn(qv, g_vols)   # gallery indices best->worst
            truth[i] = i                        # matching target shares the index
        out[level] = mrr(rankings, truth)
    done = [out[l] for l in levels if out.get(l) is not None]
    out["mean"] = sum(done) / len(done) if done else 0.0
    return out


def tune_fusion_weights(rank_fn_a: RankFn, rank_fn_b: RankFn, holdout: list[dict],
                        cfg=CFG, grid=(0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
                        levels=LEVELS) -> dict:
    """For each level, sweep w (weight on Branch A) and keep the w with best MRR.
    Expected shape: L2 -> high w (MIND useless), L1/L3 -> lower w (MIND anchors).
    Returns {level: {'w': best_w, 'mrr': best_mrr}}."""
    import fuse
    results: dict = {}
    for level in levels:
        try:
            q_vols, g_vols = build_level(holdout, level, cfg)
        except Exception as e:
            results[level] = {"w": None, "mrr": None, "skipped": f"{type(e).__name__}: {e}"}
            continue
        # precompute each branch's per-query ranking (over gallery indices) once
        A = [rank_fn_a(qv, g_vols) for qv in q_vols]
        B = [rank_fn_b(qv, g_vols) for qv in q_vols]
        truth = {i: i for i in range(len(q_vols))}
        best_w, best_mrr = None, -1.0
        for w in grid:
            rankings = {i: fuse.rank_fusion([A[i], B[i]], [w, 1.0 - w]) for i in range(len(q_vols))}
            m = mrr(rankings, truth)
            if m > best_mrr:
                best_w, best_mrr = w, m
        results[level] = {"w": best_w, "mrr": best_mrr}
    return results


# --------------------------------------------------------------------------- #
# No-data self-test: synthesize tiny "patients", confirm the plumbing + that
# MIND matches across a synthetic contrast change on L1. Runs on CPU, no MONAI.
# --------------------------------------------------------------------------- #
def _synthetic_holdout(n: int = 8, R: int = 24, seed: int = 0) -> list[dict]:
    from mind import _box_filter
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(n):
        base = torch.randn(1, 1, R, R, R, generator=g)
        base = _box_filter(base, 2)[0]                       # smooth -> anatomy-like structure
        base = (base - base.mean()) / (base.std() + 1e-5)
        query = base                                          # "ceT1"
        # target = same anatomy, different intensity mapping ("T2") + noise
        target = torch.sigmoid(1.5 * base) + 0.05 * torch.randn(1, R, R, R, generator=g)
        target = (target - target.mean()) / (target.std() + 1e-5)
        pairs.append({"query": query, "target": target})
    return pairs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="oracle keystone — self-test / plumbing check")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--res", type=int, default=24)
    args = ap.parse_args()

    if args.self_test:
        import mind
        holdout = _synthetic_holdout(n=args.n, R=args.res)
        res = evaluate(mind.mind_rank_fn, holdout)
        print("oracle self-test (MIND scorer):", {k: (round(v, 3) if isinstance(v, float) else v)
                                                   for k, v in res.items()})
        l1 = res.get("l1") or 0.0
        rand = sum(1.0 / r for r in range(1, args.n + 1)) / args.n  # random-baseline MRR
        assert l1 > max(0.5, 1.5 * rand), f"MIND L1 sanity failed (l1={l1:.3f}, rand≈{rand:.3f})"
        print(f"SELF-TEST PASS — MIND matches across synthetic contrast on L1 "
              f"(l1={l1:.3f} >> random≈{rand:.3f}).")
    else:
        ap.print_help()
