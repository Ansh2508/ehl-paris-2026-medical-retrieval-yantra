"""K-fold patient-grouped cross-validation of the MIND pipeline.

MIND is training-free, so evaluating it on K different hold-out splits has NO fold leakage —
turns our single-split 0.61 into a robust mean ± std. (The encoder is excluded: honest CV
would need K retrains; we report it single-fold elsewhere.) Per-level alignment policy:
register dataset2-like only. Optionally add the Hungarian rerank.

  python cv.py --data-root <root> --train-csv <root>/dataset1/train_pairs.csv --folds 5
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
import torch

from config import CFG
import preprocess
import oracle
import mind
import register
import rerank
from metrics import mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-holdout", type=int, default=CFG.holdout_pairs)
    ap.add_argument("--resolution", type=int, default=CFG.resolution)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hungarian", action="store_true", help="add the bijection rerank")
    args = ap.parse_args()
    CFG.data_root = __import__("pathlib").Path(args.data_root); CFG.resolution = args.resolution
    dev = torch.device(args.device)

    pairs = [{"query_id": r["query_id"], "target_id": r["target_id"],
              "query_path": r["query_image"], "target_path": r["target_image"]}
             for r in csv.DictReader(open(args.train_csv, newline=""))]

    # preprocess every image ONCE, reuse across folds
    need = {}
    for p in pairs:
        need[p["query_id"]] = p["query_path"]; need[p["target_id"]] = p["target_path"]
    id2vol = {i: preprocess.preprocess_volume(path) for i, path in need.items()}
    print(f"{len(pairs)} pairs, {len(id2vol)} volumes; {args.folds}-fold CV "
          f"(hungarian={args.hungarian})")

    per_level = {l: [] for l in oracle.LEVELS}
    means = []
    for k in range(args.folds):
        _, hold = oracle.make_holdout(pairs, n=args.n_holdout, seed=CFG.seed + k)
        holdout = [{"query": id2vol[p["query_id"]], "target": id2vol[p["target_id"]]} for p in hold]
        ref = holdout[0]["query"]
        res = {}
        for level in oracle.LEVELS:
            qv, gv = oracle.build_level(holdout, level, CFG)
            if level == "l2":                                   # per-level policy: register d2 only
                qv = [register.register_to_ref(v, ref) for v in qv]
                gv = [register.register_to_ref(v, ref) for v in gv]
            qd = [v.to(dev) for v in qv]; gd = [v.to(dev) for v in gv]
            S = -mind.mind_score_matrix(qd, gd).numpy()
            rk = rerank.hungarian_rankings(S) if args.hungarian else rerank.greedy_rankings(S)
            res[level] = mrr(rk, {i: i for i in range(len(qd))})
        for l in oracle.LEVELS:
            per_level[l].append(res[l])
        means.append(float(np.mean([res[l] for l in oracle.LEVELS])))
        print(f"  fold {k} (seed {CFG.seed + k}): " +
              " ".join(f"{l}={res[l]:.3f}" for l in oracle.LEVELS) + f"  mean={means[-1]:.3f}")

    print(f"\n=== {args.folds}-fold patient-grouped CV (mean ± std) ===")
    for l in oracle.LEVELS:
        print(f"  {l}: {np.mean(per_level[l]):.3f} ± {np.std(per_level[l]):.3f}")
    print(f"  mean: {np.mean(means):.3f} ± {np.std(means):.3f}")


if __name__ == "__main__":
    main()
