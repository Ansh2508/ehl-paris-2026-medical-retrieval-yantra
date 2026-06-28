
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter
from scipy.optimize import linear_sum_assignment

CFG = dict(size=64, reg_iters=120, template_iters=80, ssc_sigma=1.0, trim=0.5)

# teammate's SSC-12 edge set: neighbour<->neighbour across the 3 axis-pairs
_AX=[(1,0,0),(-1,0,0)]; _AY=[(0,1,0),(0,-1,0)]; _AZ=[(0,0,1),(0,0,-1)]
SSC12 = [(p,q) for A,B in [(_AX,_AY),(_AX,_AZ),(_AY,_AZ)] for p in A for q in B]

def fix_path(root, rel):
    p = Path(rel) if os.path.isabs(rel) else root/rel
    if p.exists(): return p
    if str(p).endswith(".nii.gz"):
        alt = Path(str(p)[:-3])
        if alt.exists(): return alt
    return p
def read(p): return list(csv.DictReader(open(p, newline="")))

def load_raw(path, size):
    v = nib.load(str(path)).get_fdata().astype(np.float32); v = np.squeeze(np.nan_to_num(v))
    v = zoom(v, [size/s for s in v.shape], order=1)
    pos = v[v > 0]; lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (0, 1))   # [0,1] percentile
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

def ssc_field(vol, s=1.0):
    """SSC-12 descriptor as a FIELD (12,H,W,D) — needed for per-voxel trimmed scoring."""
    v = gaussian_filter(vol, s)
    d = np.stack([gaussian_filter((np.roll(v,p,(0,1,2)) - np.roll(v,q,(0,1,2)))**2, s) for p,q in SSC12])
    var = d.mean(0, keepdims=True) + 1e-6
    de = np.exp(-d/var)
    return (de / (np.linalg.norm(de, axis=0, keepdims=True) + 1e-8)).astype(np.float32)

def _sitk(arr): return sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkFloat32)

def _affine_to(fixed_img, moving_arr, iters):   # same registration that produced our 0.934
    m = _sitk(moving_arr)
    rinit = sitk.CenteredTransformInitializer(fixed_img, m, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R1 = sitk.ImageRegistrationMethod()
    R1.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R1.SetMetricSamplingStrategy(R1.RANDOM); R1.SetMetricSamplingPercentage(0.1)
    R1.SetInterpolator(sitk.sitkLinear)
    R1.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=60)
    R1.SetOptimizerScalesFromPhysicalShift()
    R1.SetShrinkFactorsPerLevel([4,2,1]); R1.SetSmoothingSigmasPerLevel([2,1,0])
    R1.SetInitialTransform(rinit, inPlace=False)
    try: rigid_t = R1.Execute(fixed_img, m)
    except Exception: rigid_t = rinit
    aff = sitk.AffineTransform(3)
    R2 = sitk.ImageRegistrationMethod()
    R2.SetMovingInitialTransform(rigid_t); R2.SetInitialTransform(aff, inPlace=True)
    R2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R2.SetMetricSamplingStrategy(R2.RANDOM); R2.SetMetricSamplingPercentage(0.1)
    R2.SetInterpolator(sitk.sitkLinear)
    R2.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=1e-4, numberOfIterations=iters)
    R2.SetOptimizerScalesFromPhysicalShift()
    R2.SetShrinkFactorsPerLevel([4,2,1]); R2.SetSmoothingSigmasPerLevel([2,1,0])
    try:
        aff_t = R2.Execute(fixed_img, m); total = sitk.CompositeTransform([rigid_t, aff_t])
    except Exception:
        total = rigid_t
    return sitk.GetArrayFromImage(sitk.Resample(m, fixed_img, total, sitk.sitkLinear, 0.0, m.GetPixelID()))

def build_template(all_arrs, iters):
    seed = np.mean(np.stack(all_arrs), axis=0); seed_img = _sitk(seed); warped = []; t0=time.time()
    for i,a in enumerate(all_arrs):
        warped.append(_affine_to(seed_img, a, iters))
        if i%30==0: print(f"  template {i}/{len(all_arrs)} ({time.time()-t0:.0f}s)", flush=True)
    return np.mean(np.stack(warped), axis=0)

def reg_all(imgs, ref_img, iters, tag):
    vols, descs = {}, {}; t0=time.time()
    for i,(k,p) in enumerate(imgs.items()):
        reg = _affine_to(ref_img, load_raw(p, CFG["size"]), iters)
        vols[k] = reg.astype(np.float32); descs[k] = ssc_field(reg, CFG["ssc_sigma"])
        if i%20==0: print(f"  {tag} reg+SSCfield {i}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
    return vols, descs

def trimmed_simmat(qids, gids, qv, gv, qd, gd, TRIM):
    n,m = len(qids), len(gids); S = np.zeros((n,m), np.float32); t0=time.time()
    for i,qi in enumerate(qids):
        for j,gj in enumerate(gids):
            msk = (qv[qi] > 0.05) | (gv[gj] > 0.05)
            dv = np.sort(np.abs(qd[qi]-gd[gj]).mean(0)[msk])     # per-voxel SSC distance, ascending
            if dv.size:
                k = max(1, int(round((1-TRIM)*dv.size)))          # keep best (1-TRIM)
                S[i,j] = -float(dv[:k].mean())                    # similarity = -trimmed mean
        if i%20==0: print(f"    score row {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return S

def greedy_rank(S, qids, gids):
    return {qids[i]: [gids[j] for j in np.argsort(-S[i])] for i in range(len(qids))}

def hungarian_rank(S, qids, gids):
    row,col = linear_sum_assignment(-S)                          # optimal 1:1 on the bijection
    assigned = {qids[r]: gids[c] for r,c in zip(row,col)}
    out = {}
    for i,qid in enumerate(qids):
        agt = assigned[qid]
        rest = [gids[j] for j in np.argsort(-S[i]) if gids[j]!=agt]
        out[qid] = [agt] + rest                                  # assigned target rank-1, rest by sim
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-greedy", default="submission_dataset2_trim_greedy.csv")
    ap.add_argument("--out-hungarian", default="submission_dataset2_trim_hungarian.csv")
    a = ap.parse_args(); root = Path(a.data_root).resolve()
    rows_g, rows_h = [], []
    for split in ["val","test"]:
        qc = read(root/f"dataset2/{split}_queries.csv"); gc = read(root/f"dataset2/{split}_gallery.csv")
        qids = [r["query_id"] for r in qc]; gids = [r["target_id"] for r in gc]
        qimg = {r["query_id"]: fix_path(root, r["query_image"]) for r in qc}
        gimg = {r["target_id"]: fix_path(root, r["target_image"]) for r in gc}
        print(f"dataset2/{split}: template@{CFG['size']}, affine-cascade reg, SSC-12 field, trim={CFG['trim']} ...", flush=True)
        tmpl = _sitk(build_template([load_raw(p, CFG["size"]) for p in gimg.values()], CFG["template_iters"]))
        gv, gd = reg_all(gimg, tmpl, CFG["reg_iters"], f"{split}/gal")
        qv, qd = reg_all(qimg, tmpl, CFG["reg_iters"], f"{split}/qry")
        S = trimmed_simmat(qids, gids, qv, gv, qd, gd, CFG["trim"])
        g = greedy_rank(S, qids, gids); h = hungarian_rank(S, qids, gids)
        for qid in qids:
            rows_g.append((qid, " ".join(g[qid]))); rows_h.append((qid, " ".join(h[qid])))
        print(f"  dataset2/{split} done", flush=True)
    for path, rows, tag in [(a.out_greedy, rows_g, "trim+greedy"), (a.out_hungarian, rows_h, "trim+Hungarian")]:
        with open(path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["query_id","target_id_ranking"])
            for qid,r in rows: w.writerow([qid,r])
        print(f"wrote {len(rows)} rows -> {path}  ({tag})", flush=True)

if __name__ == "__main__": main()
