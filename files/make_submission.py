"""Build the 377-row MIND submission with the proven per-level alignment policy:
dataset2 -> rigid-register every volume to a dataset1 reference (de-rotates L2);
dataset1 & dataset3 -> no alignment. Ranks each split's gallery by MIND distance.

Oracle says this recipe gives L1 0.99 / L2 0.47 / L3 0.37. Submitting it gives a REAL
LB number and tells us how well the oracle predicts the leaderboard.

Usage (box with data):
  python make_submission.py --data-root <root> --out submission_mind.csv
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import torch

from config import CFG
import preprocess
import mind
import submit


def read_manifest(path: str, id_col: str, img_col: str) -> list[tuple[str, str]]:
    return [(r[id_col], r[img_col]) for r in csv.DictReader(open(path, newline=""))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="submission_mind.csv")
    ap.add_argument("--resolution", type=int, default=CFG.resolution)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-reg-d2", action="store_true", help="disable dataset2 registration")
    args = ap.parse_args()

    CFG.data_root = Path(args.data_root)
    CFG.resolution = args.resolution
    dev = torch.device(args.device)
    root = Path(args.data_root)

    # canonical reference for dataset2 alignment = a dataset1 train query (the registered frame)
    tp = list(csv.DictReader(open(root / "dataset1" / "train_pairs.csv", newline="")))
    ref = preprocess.preprocess_volume(tp[0]["query_image"])
    print(f"reference = {tp[0]['query_id']} (dataset1 train query)")

    sets = []
    for ds in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            qcsv = root / ds / f"{split}_queries.csv"
            gcsv = root / ds / f"{split}_gallery.csv"
            if qcsv.exists() and gcsv.exists():
                sets.append((ds, split, str(qcsv), str(gcsv)))

    def prep(img_paths, do_reg):
        vols = [preprocess.preprocess_volume(p) for p in img_paths]
        if do_reg:
            import register
            vols = [register.register_to_ref(v, ref) for v in vols]
        return [v.to(dev) for v in vols]

    rows: list[tuple[str, list[str]]] = []
    for ds, split, qcsv, gcsv in sets:
        t = time.time()
        q = read_manifest(qcsv, "query_id", "query_image")
        g = read_manifest(gcsv, "target_id", "target_image")
        do_reg = (ds == "dataset2") and not args.no_reg_d2
        qv = prep([p for _, p in q], do_reg)
        gv = prep([p for _, p in g], do_reg)
        D = mind.mind_score_matrix(qv, gv)                  # [Q,G], lower = more similar
        gids = [tid for tid, _ in g]
        for i, (qid, _) in enumerate(q):
            order = torch.argsort(D[i]).tolist()
            rows.append((qid, [gids[j] for j in order]))
        print(f"  {ds}/{split}: {len(q)}q x {len(g)}g (reg={do_reg})  {time.time() - t:.1f}s")

    submit.write_submission(rows, args.out)
    submit.validate(rows, strict=True)
    print(f"WROTE {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
