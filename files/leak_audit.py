"""Data-integrity / leakage audit (run FIRST, per the ReID-leak playbook).

Checks whether NIfTI headers (affine / qoffset / shape / pixdim / file size) let you match
query<->target WITHOUT image content. FINDING on this challenge: dataset3 (L3) targets are
resampled into the query's physical space, so each query's affine uniquely identifies its true
target (MRR~1.0 by affine alone). We REPORT this and do NOT exploit it (content-honest) — see
REFLECTION.md "Data-integrity audit".

  python leak_audit.py --data-root <root>
"""
from __future__ import annotations
import argparse
import csv
import os
from collections import Counter
from pathlib import Path
import numpy as np
import nibabel as nib


def resolve(p, root):
    fp = Path(root) / p
    if fp.exists():
        return fp
    alt = fp.with_name(fp.name[:-3]) if fp.name.endswith(".nii.gz") else fp.with_name(fp.name + ".gz")
    return alt if alt.exists() else fp


def fingerprint(path):
    img = nib.load(str(path))           # lazy: header/affine without loading voxels
    h = img.header
    return {"shape": tuple(int(x) for x in img.shape),
            "pixdim": tuple(round(float(x), 3) for x in h["pixdim"][1:4]),
            "affine": np.round(img.affine, 2).tobytes(),
            "qoffset": tuple(round(float(h[k]), 2) for k in ("qoffset_x", "qoffset_y", "qoffset_z")),
            "fsize": os.path.getsize(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--limit", type=int, default=120)
    a = ap.parse_args()
    root = a.data_root

    tp = list(csv.DictReader(open(f"{root}/dataset1/train_pairs.csv")))[:a.limit]
    share = {k: 0 for k in ("affine", "shape", "pixdim", "qoffset")}
    for r in tp:
        q = fingerprint(resolve(r["query_image"], root))
        t = fingerprint(resolve(r["target_image"], root))
        for k in share:
            share[k] += int(q[k] == t[k])
    print(f"[d1 train pairs n={len(tp)}] query/target share:", {k: f"{v}/{len(tp)}" for k, v in share.items()})

    for ds in ("dataset2", "dataset3"):
        try:
            q = list(csv.DictReader(open(f"{root}/{ds}/val_queries.csv")))
            g = list(csv.DictReader(open(f"{root}/{ds}/val_gallery.csv")))
        except FileNotFoundError:
            continue
        qf = [fingerprint(resolve(r["query_image"], root)) for r in q]
        gf = [fingerprint(resolve(r["target_image"], root)) for r in g]
        print(f"--- {ds} val: {len(q)}q / {len(g)}g ---")
        for key in ("affine", "shape", "pixdim", "qoffset", "fsize"):
            gc = Counter(x[key] for x in gf)
            exact1 = sum(1 for x in qf if gc.get(x[key], 0) == 1)
            leak = " <-- PERFECT LEAK" if (len(q) and exact1 == len(q)) else ""
            print(f"   {key:8s}: exactly-1 gallery match {exact1}/{len(q)}  "
                  f"(distinct gallery {key}: {len(gc)}/{len(g)}){leak}")


if __name__ == "__main__":
    main()
