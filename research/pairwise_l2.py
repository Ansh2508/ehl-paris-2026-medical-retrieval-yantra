"""L2: pairwise registration-as-scoring (the doc's 'move 5').

Instead of registering everything to ONE reference, register the query to EACH candidate
directly and score by post-alignment MIND distance — the true pair should snap into alignment
best. Compared head-to-head with common-reference rigid on the SAME fakeL2. ~N^2 registrations.
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
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
    qv, gv = oracle.build_level(holdout, "l2", CFG)         # synth fakeL2 ONCE (fair comparison)
    n = len(qv); truth = {i: i for i in range(n)}
    print(f"pairwise vs common-ref on fakeL2: n={n}")

    # baseline: common-reference rigid (our current L2 approach)
    t = time.time()
    qr = [register.register_to_ref(v, ref, "rigid").to(dev) for v in qv]
    gr = [register.register_to_ref(v, ref, "rigid").to(dev) for v in gv]
    Sb = -mind.mind_score_matrix(qr, gr).numpy()
    print(f"  common-ref rigid : MIND={mrr(rerank.greedy_rankings(Sb), truth):.3f}  "
          f"+Hung={mrr(rerank.hungarian_rankings(Sb), truth):.3f}   ({time.time()-t:.0f}s)")

    # pairwise registration-as-scoring: register q_i -> g_j, score by post-alignment MIND
    t = time.time()
    gdesc = [mind.mind_descriptor(v.to(dev)) for v in gv]   # candidate descriptors (own frame)
    Sp = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            rq = register.register_to_ref(qv[i], gv[j], "rigid").to(dev)
            Sp[i, j] = -mind.mind_distance(mind.mind_descriptor(rq), gdesc[j])
    print(f"  pairwise reg-score: MIND={mrr(rerank.greedy_rankings(Sp), truth):.3f}  "
          f"+Hung={mrr(rerank.hungarian_rankings(Sp), truth):.3f}   ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
