
import argparse, csv
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F  # noqa
import os
_HERE="/shared-docker/yantra"
exec(open(os.path.join(_HERE,"d3_convexadam.py")).read().split("def main()")[0])  # reuse all d3 fns

root=Path("/shared-docker/data")
def rd(p): return list(csv.DictReader(open(p,newline="")))
def fixp(rel):
    p=root/rel
    if p.exists(): return p
    if str(p).endswith(".nii.gz"):
        a=Path(str(p)[:-3]); return a if a.exists() else p
    return p

def greedy_rank(S,qids,gids):
    return {qids[i]:[gids[j] for j in np.argsort(-S[i])] for i in range(len(qids))}

def mrr_vs(order, qids):
    # we have no val ground-truth file; measure SELF-CONSISTENCY of greedy vs the shared-template greedy,
    # AND the separate-template AGREEMENT — both under greedy (no Hungarian confound).
    return order

print("=== D3 GREEDY CONTROL (no Hungarian) — val pool, separate templates ===", flush=True)
qc=rd(root/"dataset3/val_queries.csv"); gc=rd(root/"dataset3/val_gallery.csv")
qids=[r["query_id"] for r in qc]; gids=[r["target_id"] for r in gc]
qimg={r["query_id"]:fixp(r["query_image"]) for r in qc}; gimg={r["target_id"]:fixp(r["target_image"]) for r in gc}

garrs=[load_raw(p,CFG["size"]) for p in gimg.values()]
qarrs=[load_raw(p,CFG["size"]) for p in qimg.values()]

# ---- (A) SHARED template, greedy ----
tmpl=torch.from_numpy(build_template(garrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
gv={k:deformable_to_template(load_raw(p,CFG["size"]),tmpl) for k,p in gimg.items()}
qv={k:deformable_to_template(load_raw(p,CFG["size"]),tmpl) for k,p in qimg.items()}
gd={k:ssc_field(v) for k,v in gv.items()}; qd={k:ssc_field(v) for k,v in qv.items()}
S_shared=trimmed_S(qids,gids,qv,gv,qd,gd,CFG["trim"])
greedy_shared=greedy_rank(S_shared,qids,gids)

# ---- (B) SEPARATE templates, greedy (the real co-location test) ----
gtmpl=torch.from_numpy(build_template(garrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
qtmpl=torch.from_numpy(build_template(qarrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
gv2={k:deformable_to_template(load_raw(p,CFG["size"]),gtmpl) for k,p in gimg.items()}
qv2={k:deformable_to_template(load_raw(p,CFG["size"]),qtmpl) for k,p in qimg.items()}
gd2={k:ssc_field(v) for k,v in gv2.items()}; qd2={k:ssc_field(v) for k,v in qv2.items()}
S_sep=trimmed_S(qids,gids,qv2,gv2,qd2,gd2,CFG["trim"])
greedy_sep=greedy_rank(S_sep,qids,gids)

# ---- compare GREEDY top-1 across shared vs separate templates ----
agree=sum(1 for q in qids if greedy_shared[q][0]==greedy_sep[q][0])
print(f"\nGREEDY top-1 agreement, shared-template vs separate-template: {agree}/{len(qids)}")
print("  HIGH => the descriptor picks the same match WITHOUT the bijection constraint AND without a shared frame")
print("       => the L3 signal is anatomical, not Hungarian-manufactured and not warp-co-location")
print("  LOW  => the deformable 1.0 leaned on the shared frame / assignment, not anatomy")

# ---- also: how often do greedy and Hungarian AGREE on shared template? (quantifies how much Hungarian moved) ----
hung_shared=hungarian_rank(S_shared,qids,gids)
gh=sum(1 for q in qids if greedy_shared[q][0]==hung_shared[q][0])
print(f"\nGREEDY vs HUNGARIAN top-1 (shared template): {gh}/{len(qids)} agree")
print(f"  -> Hungarian changed {len(qids)-gh}/{len(qids)} assignments (collision fixes)")
