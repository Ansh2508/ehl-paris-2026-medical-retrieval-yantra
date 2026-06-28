"""L2-focused experiment: does stronger registration lift dataset2 (rigid+elastic)?
Compare registration modes on the SAME synthesized fakeL2 (fair), scored by MIND (+Hungarian).
Caveat under test: deformable to an arbitrary reference may over-normalize (hurt)."""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
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
    ap.add_argument("--n-holdout", type=int, default=40)
    ap.add_argument("--resolution", type=int, default=CFG.resolution)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    CFG.data_root = Path(args.data_root); CFG.resolution = args.resolution
    dev = torch.device(args.device)

    pairs = [{"query_path": r["query_image"], "target_path": r["target_image"]}
             for r in csv.DictReader(open(args.train_csv, newline=""))]
    _, hold = oracle.make_holdout(pairs, n=args.n_holdout)
    holdout = [{"query": preprocess.preprocess_volume(p["query_path"]),
                "target": preprocess.preprocess_volume(p["target_path"])} for p in hold]
    ref = holdout[0]["query"]
    qv, gv = oracle.build_level(holdout, "l2", CFG)        # synth fakeL2 ONCE -> fair across modes
    truth = {i: i for i in range(len(qv))}
    print(f"L2 registration-mode ablation: n={len(qv)} pairs")

    for mode in ["none", "rigid", "affine", "deformable"]:
        t = time.time()
        if mode == "none":
            q = [v.to(dev) for v in qv]; g = [v.to(dev) for v in gv]
        else:
            q = [register.register_to_ref(v, ref, mode=mode).to(dev) for v in qv]
            g = [register.register_to_ref(v, ref, mode=mode).to(dev) for v in gv]
        S = -mind.mind_score_matrix(q, g).numpy()
        mg = mrr(rerank.greedy_rankings(S), truth)
        mh = mrr(rerank.hungarian_rankings(S), truth)
        print(f"  L2 mode={mode:11s}: MIND={mg:.3f}  MIND+Hung={mh:.3f}   ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
