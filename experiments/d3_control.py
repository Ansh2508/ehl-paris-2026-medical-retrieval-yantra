
import numpy as np, csv, torch, torch.nn.functional as F
from pathlib import Path
# reuse every function from the convexadam script (everything above main())
exec(open("/shared-docker/yantra/d3_convexadam.py").read().split("def main()")[0])

root = Path("/shared-docker/data")
def read(p): return list(csv.DictReader(open(p, newline="")))
def fixp(rel):
    p = root/rel
    if p.exists(): return p
    if str(p).endswith(".nii.gz"):
        a = Path(str(p)[:-3]); return a if a.exists() else p
    return p

# VAL only. Build TWO SEPARATE templates so a query and its true gallery target
# do NOT share one warped frame. If the 1.0 was anatomy -> ranking unchanged.
# If it was warp-induced co-location -> ranking scatters.
qc = read(root/"dataset3/val_queries.csv"); gc = read(root/"dataset3/val_gallery.csv")
qids = [r["query_id"] for r in qc]; gids = [r["target_id"] for r in gc]
qimg = {r["query_id"]: fixp(r["query_image"]) for r in qc}
gimg = {r["target_id"]: fixp(r["target_image"]) for r in gc}

garrs = [load_raw(p, CFG["size"]) for p in gimg.values()]
qarrs = [load_raw(p, CFG["size"]) for p in qimg.values()]
gtmpl = torch.from_numpy(build_template(garrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])
qtmpl = torch.from_numpy(build_template(qarrs)).to(DEV).view(1,1,CFG["size"],CFG["size"],CFG["size"])  # DIFFERENT template

gv = {k: deformable_to_template(load_raw(p, CFG["size"]), gtmpl) for k,p in gimg.items()}
qv = {k: deformable_to_template(load_raw(p, CFG["size"]), qtmpl) for k,p in qimg.items()}
gd = {k: ssc_field(v) for k,v in gv.items()}; qd = {k: ssc_field(v) for k,v in qv.items()}
S  = trimmed_S(qids, gids, qv, gv, qd, gd, CFG["trim"])
order = hungarian_rank(S, qids, gids)

sub = {r["query_id"]: r["target_id_ranking"].split()
       for r in read("/shared-docker/yantra/submission_dataset3_convexadam.csv")
       if r["query_id"] in set(qids)}
agree = sum(1 for q in qids if order[q][0] == sub[q][0])
print(f"SEPARATE-TEMPLATE top-1 agreement with the shared-template 1.0 submission: {agree}/{len(qids)}")
print("  LOW  agreement => 1.0 came from SHARED warped geometry = co-location leak (discard)")
print("  HIGH agreement => robust to template choice = genuinely anatomical (honest, splice it)")
