"""Driver: first REAL per-level MRR for a scorer (default MIND) on the 350 d1 pairs.

Flow: load train_pairs -> hold out N -> preprocess those volumes -> oracle synth
fakeL2/fakeL3 -> per-level MRR (MIND, gallery descriptors precomputed once on GPU)
-> optional W&B. This is the number that gates everything (CLAUDE.md §8).

Usage (on a box with the data + MONAI):
  python run_oracle.py --data-root /shared-docker/data \
    --train-csv /shared-docker/data/dataset1/train_pairs.csv
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import torch

from config import CFG
import preprocess
import oracle
import metrics
import mind


def load_pairs(train_csv: str) -> list[dict]:
    rows = list(csv.DictReader(open(train_csv, newline="")))
    return [{"query_path": r["query_image"], "target_path": r["target_image"]} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--resolution", type=int, default=CFG.resolution)
    ap.add_argument("--n-holdout", type=int, default=CFG.holdout_pairs)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb-project", default="ehl-mri-retrieval", help="empty string disables W&B")
    ap.add_argument("--run-name", default="mind-oracle")
    ap.add_argument("--align", choices=["none", "rigid"], default="none",
                    help="rigid = register every volume to a canonical reference (tests finding 2)")
    ap.add_argument("--levels", default="l1,l2,l3", help="comma-sep subset of l1,l2,l3")
    args = ap.parse_args()
    levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())

    CFG.data_root = Path(args.data_root)
    CFG.resolution = args.resolution
    dev = torch.device(args.device)

    pairs = load_pairs(args.train_csv)
    _, hold = oracle.make_holdout(pairs, n=args.n_holdout)
    print(f"loaded {len(pairs)} train pairs; holdout={len(hold)}; res={args.resolution}; dev={dev}")

    # preprocess holdout volumes once (MONAI on CPU; augment also runs on CPU below)
    t0 = time.time()
    holdout = []
    for p in hold:
        holdout.append({
            "query": preprocess.preprocess_volume(p["query_path"]),
            "target": preprocess.preprocess_volume(p["target_path"]),
        })
    print(f"preprocessed {len(holdout)} pairs in {time.time() - t0:.1f}s")

    # canonical reference for rigid alignment = an undistorted L1 query (the registered frame)
    ref = holdout[0]["query"]
    print(f"align={args.align}; levels={levels}")

    # per-level MRR with MIND (gallery descriptors precomputed once per level on GPU)
    results: dict = {}
    for level in levels:
        t = time.time()
        try:
            qv, gv = oracle.build_level(holdout, level, CFG)   # CPU tensors (synth on CPU)
        except Exception as e:
            print(f"  skip {level}: {type(e).__name__}: {e}")
            results[level] = None
            continue
        if args.align == "rigid":
            import register
            qv = [register.register_to_ref(v, ref) for v in qv]
            gv = [register.register_to_ref(v, ref) for v in gv]
            print(f"    registered {len(qv)+len(gv)} vols ({time.time() - t:.1f}s so far)")
        qv = [v.to(dev) for v in qv]
        gv = [v.to(dev) for v in gv]
        D = mind.mind_score_matrix(qv, gv)                      # [Q,G], lower = more similar
        rankings = {i: torch.argsort(D[i]).tolist() for i in range(len(qv))}
        truth = {i: i for i in range(len(qv))}
        results[level] = metrics.mrr(rankings, truth)
        print(f"  {level}: MRR={results[level]:.4f}  ({time.time() - t:.1f}s)")
    done = [results[l] for l in levels if results.get(l) is not None]
    results["mean"] = sum(done) / len(done) if done else 0.0
    print("RESULT:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items()})

    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=args.run_name,
                             config={"scorer": "mind", "resolution": args.resolution,
                                     "n_holdout": args.n_holdout, "align": args.align,
                                     "levels": ",".join(levels)},
                             settings=wandb.Settings(silent=True))
            wandb.log({f"oracle/{k}": v for k, v in results.items() if isinstance(v, float)})
            print("wandb:", run.get_url())
            wandb.finish()
        except Exception as e:
            print("wandb skipped:", e)


if __name__ == "__main__":
    main()
