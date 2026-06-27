"""Final integration + ablation on the oracle holdout — the table that decides the recipe.

Per level (per-level alignment policy: register dataset2-like only), compares:
  MIND | MIND+Hungarian | MIND+k-reciprocal | ENC | MIND+ENC fusion | fusion+rerank
so the oracle picks the best per level. Oracle-gated (CLAUDE.md 8).

Usage:
  python combine.py --data-root <root> --train-csv <root>/dataset1/train_pairs.csv \
    --encoder /shared-docker/yantra-gs/outputs_gs/encoder_v1.pt
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch

from config import CFG
import preprocess
import oracle
import mind
import mutual_info
import register
import rerank
from metrics import mrr


def _rank_norm(S):
    """Per-row rank-normalized scores in [0,1], 1=best. S: [Q,G] similarity."""
    S = np.asarray(S, float)
    order = np.argsort(-S, axis=1)
    out = np.zeros_like(S)
    G = S.shape[1]
    for i in range(S.shape[0]):
        out[i, order[i]] = np.linspace(1.0, 0.0, G)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--encoder", default="", help="encoder checkpoint (optional)")
    ap.add_argument("--n-holdout", type=int, default=CFG.holdout_pairs)
    ap.add_argument("--resolution", type=int, default=CFG.resolution)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fuse-w", type=float, default=0.5, help="weight on MIND in MIND+ENC fusion")
    args = ap.parse_args()
    CFG.data_root = Path(args.data_root); CFG.resolution = args.resolution
    dev = torch.device(args.device)

    pairs = [{"query_path": r["query_image"], "target_path": r["target_image"]}
             for r in csv.DictReader(open(args.train_csv, newline=""))]
    _, hold = oracle.make_holdout(pairs, n=args.n_holdout)
    holdout = [{"query": preprocess.preprocess_volume(p["query_path"]),
                "target": preprocess.preprocess_volume(p["target_path"])} for p in hold]
    ref = holdout[0]["query"]
    print(f"holdout={len(holdout)}  encoder={'yes' if args.encoder else 'no'}")

    enc = None
    if args.encoder:
        from encoder import Encoder
        ck = torch.load(args.encoder, map_location=dev)
        enc = Encoder(CFG, backbone=ck.get("backbone", "resnet18")).to(dev)
        enc.load_state_dict(ck["model"]); enc.eval()

    results: dict = {}   # variant -> {level: mrr}
    for level in oracle.LEVELS:
        qv, gv = oracle.build_level(holdout, level, CFG)
        if level == "l2":                                   # per-level alignment: register d2-like only
            qv = [register.register_to_ref(v, ref) for v in qv]
            gv = [register.register_to_ref(v, ref) for v in gv]
        qd = [v.to(dev) for v in qv]; gd = [v.to(dev) for v in gv]
        Q = len(qd)
        truth = {i: i for i in range(Q)}

        Dm = mind.mind_score_matrix(qd, gd).numpy()         # MIND distance [Q,G]
        Sm = -Dm
        Snmi = mutual_info.nmi_score_matrix(qd, gd, downsample=2).numpy()   # NMI similarity (entropy)
        Smn = 0.5 * _rank_norm(Sm) + 0.5 * _rank_norm(Snmi)                 # MIND + NMI rank-fusion
        variants = {
            "MIND": rerank.greedy_rankings(Sm),
            "MIND+Hung": rerank.hungarian_rankings(Sm),
            "MIND+krecip": rerank.rankings_from_dist(
                rerank.re_ranking(Dm, mind.mind_score_matrix(qd, qd).numpy(),
                                  mind.mind_score_matrix(gd, gd).numpy())),
            "NMI": rerank.greedy_rankings(Snmi),
            "MIND+NMI": rerank.greedy_rankings(Smn),
            "MIND+NMI+Hung": rerank.hungarian_rankings(Smn),
        }
        if enc is not None:
            qe = torch.stack([enc.encode(v) for v in qd])
            ge = torch.stack([enc.encode(v) for v in gd])
            Se = (qe @ ge.t()).cpu().numpy()
            Sf = args.fuse_w * _rank_norm(Sm) + (1 - args.fuse_w) * _rank_norm(Se)
            variants["ENC"] = rerank.greedy_rankings(Se)
            variants["MIND+ENC"] = rerank.greedy_rankings(Sf)
            variants["MIND+ENC+Hung"] = rerank.hungarian_rankings(Sf)

        for name, rk in variants.items():
            results.setdefault(name, {})[level] = mrr(rk, truth)
        print(f"[{level}] " + "  ".join(f"{n}={results[n][level]:.3f}" for n in variants))

    print("\n=== ABLATION (per-level MRR + mean) ===")
    names = list(results.keys())
    hdr = f"{'variant':16s} " + " ".join(f"{l:>6s}" for l in oracle.LEVELS) + f" {'mean':>7s}"
    print(hdr)
    for n in names:
        vals = [results[n].get(l) for l in oracle.LEVELS]
        m = np.mean([v for v in vals if v is not None])
        print(f"{n:16s} " + " ".join(f"{v:6.3f}" if v is not None else "   -  " for v in vals) + f" {m:7.3f}")
    # best-per-level (the deployable recipe)
    best = {l: max(names, key=lambda n: results[n].get(l, -1)) for l in oracle.LEVELS}
    bestmean = np.mean([results[best[l]][l] for l in oracle.LEVELS])
    print("\nbest-per-level:", {l: f"{best[l]}={results[best[l]][l]:.3f}" for l in oracle.LEVELS},
          f"=> mean {bestmean:.3f}")


if __name__ == "__main__":
    main()
