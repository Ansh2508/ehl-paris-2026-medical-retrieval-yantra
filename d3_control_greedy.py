
"""
dataset3 — INTEGRITY CONTROL for the deformable L3 1.0 (GREEDY version, no Hungarian confound).
Greedy (per-query NN, no assignment constraint), so agreement reflects the descriptor alone.
  (A) shared vs separate-template greedy agreement -> rules out warp co-location
  (B) greedy vs Hungarian agreement -> quantifies how much Hungarian moved L3
MEASURED (val, 20): (A) 20/20 anatomical ; (B) 20/20, Hungarian changed 0/20 (L3 owes nothing to assignment).
Run: python3 d3_control.py --data-root /shared-docker/data
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F  # noqa
_HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(_HERE, "d3_convexadam.py")).read().split("def main()")[0])

def greedy_rank(S, qids, gids):
    return {qids[i]: [gids[j] for j in np.argsort(-S[i])] for i in range(len(qids))}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True); a = ap.parse_args()
    root = Path(a.data_root).resolve()
    def rd(p): return list(csv.DictReader(open(p, newline="")))
    def fixp(rel):
        p = root / rel
        if p.exists(): return p
        if str(p).endswith(".nii.gz"):
            alt = Path(str(p)[:-3]); return alt if alt.exists() else p
        return p
    qc = rd(root/"dataset3/val_queries.csv"); gc = rd(root/"dataset3/val_gallery.csv")
    qids = [r["query_id"] for r in qc]; gids = [r["target_id"] for r in gc]
    qimg = {r["query_id"]: fixp(r["query_image"]) for r in qc}; gimg = {r["target_id"]: fixp(r["target_image"]) for r in gc}
    garrs = [load_raw(p, CFG["size"]) for p in gimg.values()]; qarrs = [load_raw(p, CFG["size"]) for p in qimg.values()]
    # (A1) shared template, greedy
    tmpl = torch.from_numpy(build_template(garrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
    gv = {k: deformable_to_template(load_raw(p, CFG["size"]), tmpl) for k,p in gimg.items()}
    qv = {k: deformable_to_template(load_raw(p, CFG["size"]), tmpl) for k,p in qimg.items()}
    gd = {k: ssc_field(v) for k,v in gv.items()}; qd = {k: ssc_field(v) for k,v in qv.items()}
    S_shared = trimmed_S(qids, gids, qv, gv, qd, gd, CFG["trim"]); greedy_shared = greedy_rank(S_shared, qids, gids)
    # (A2) separate templates, greedy
    gtmpl = torch.from_numpy(build_template(garrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
    qtmpl = torch.from_numpy(build_template(qarrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
    gv2 = {k: deformable_to_template(load_raw(p, CFG["size"]), gtmpl) for k,p in gimg.items()}
    qv2 = {k: deformable_to_template(load_raw(p, CFG["size"]), qtmpl) for k,p in qimg.items()}
    gd2 = {k: ssc_field(v) for k,v in gv2.items()}; qd2 = {k: ssc_field(v) for k,v in qv2.items()}
    S_sep = trimmed_S(qids, gids, qv2, gv2, qd2, gd2, CFG["trim"]); greedy_sep = greedy_rank(S_sep, qids, gids)
    agree = sum(1 for q in qids if greedy_shared[q][0] == greedy_sep[q][0])
    print(f"(A) GREEDY shared-vs-separate-template top-1 agreement: {agree}/{len(qids)}")
    print("    HIGH => same match, NO bijection constraint AND NO shared frame => ANATOMICAL")
    hung_shared = hungarian_rank(S_shared, qids, gids)
    gh = sum(1 for q in qids if greedy_shared[q][0] == hung_shared[q][0])
    print(f"(B) GREEDY vs HUNGARIAN top-1 (shared): {gh}/{len(qids)} agree (Hungarian changed {len(qids)-gh}/{len(qids)})")
    print("    0 changed => L3's 1.0 owes NOTHING to assignment (pure descriptor + registration)")

if __name__ == "__main__":
    main()
