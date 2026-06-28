
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter

CFG = dict(mind_size=64, reg_iters=120, template_iters=80, ssc_dilation=2, ssc_sigma=1.0)

def fix_path(root, rel):
    p = Path(rel) if os.path.isabs(rel) else root/rel
    if p.exists(): return p
    if str(p).endswith(".nii.gz"):
        alt = Path(str(p)[:-3])
        if alt.exists(): return alt
    return p
def read(p): return list(csv.DictReader(open(p, newline="")))
def znorm(S): return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-8)

def load_raw(path, size):
    v = nib.load(str(path)).get_fdata().astype(np.float32); v = np.nan_to_num(v)
    pos = v[v > 0]; lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (0, 1))
    v = np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)
    return zoom(v, [size/s for s in v.shape], order=1)

def ssc_desc(vol, sigma, dilation):
    """Self-Similarity Context (Heinrich 2013): self-similarity between PAIRS of
    6-neighbourhood points (octahedron edges), excluding the center voxel -> noise-robust.
    12-dimensional vs MIND's 6, more discriminative on deformed/cross-modal volumes."""
    vol = gaussian_filter(vol, sigma)
    d = dilation
    pts = [(d,0,0),(-d,0,0),(0,d,0),(0,-d,0),(0,0,d),(0,0,-d)]   # 6 face neighbours
    # 12 octahedron edges = all neighbour pairs EXCEPT the 3 opposite pairs (0,1)(2,3)(4,5)
    pairs = [(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]
    feats = []
    for i, j in pairs:
        a = np.roll(vol, pts[i], axis=(0,1,2)); b = np.roll(vol, pts[j], axis=(0,1,2))
        feats.append(gaussian_filter((a - b)**2, sigma))
    feats = np.stack(feats, axis=-1)
    var = np.mean(feats, axis=-1, keepdims=True) + 1e-8
    s = np.exp(-feats / var); s = s / (s.sum(axis=-1, keepdims=True) + 1e-8)
    return s.astype(np.float32).ravel()

def _sitk(arr): return sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkFloat32)

def _affine_to(fixed_img, moving_arr, iters):
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

def build_template(all_arrs, size, iters):
    seed = np.mean(np.stack(all_arrs), axis=0); seed_img = _sitk(seed); warped = []; t0 = time.time()
    for i, a in enumerate(all_arrs):
        warped.append(_affine_to(seed_img, a, iters))
        if i % 30 == 0: print(f"  template build {i}/{len(all_arrs)} ({time.time()-t0:.0f}s)", flush=True)
    return np.mean(np.stack(warped), axis=0)

def desc_to_ref(imgs, size, ref_img, iters, tag):
    out = {}; t0 = time.time()
    for i, (k, p) in enumerate(imgs.items()):
        reg = _affine_to(ref_img, load_raw(p, size), iters)
        out[k] = ssc_desc(reg, CFG["ssc_sigma"], CFG["ssc_dilation"])
        if i % 20 == 0: print(f"  {tag} affine+SSC {i}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
    return out

def smat(qe, te, qids, gids):
    Q = np.stack([qe[i] for i in qids]); G = np.stack([te[i] for i in gids])
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    return Q @ G.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="submission_dataset2_ssc.csv")
    a = ap.parse_args(); root = Path(a.data_root).resolve(); rows = []
    for split in ["val", "test"]:
        qc = read(root/f"dataset2/{split}_queries.csv"); gc = read(root/f"dataset2/{split}_gallery.csv")
        qids = [r["query_id"] for r in qc]; gids = [r["target_id"] for r in gc]
        qimg = {r["query_id"]: fix_path(root, r["query_image"]) for r in qc}
        gimg = {r["target_id"]: fix_path(root, r["target_image"]) for r in gc}
        print(f"dataset2/{split}: mean template from {len(gimg)} gallery, then affine + SSC...", flush=True)
        gal_arrs = [load_raw(p, CFG["mind_size"]) for p in gimg.values()]
        template = build_template(gal_arrs, CFG["mind_size"], CFG["template_iters"])
        tmpl_img = _sitk(template)
        gm = desc_to_ref(gimg, CFG["mind_size"], tmpl_img, CFG["reg_iters"], f"{split}/gal")
        qm = desc_to_ref(qimg, CFG["mind_size"], tmpl_img, CFG["reg_iters"], f"{split}/qry")
        S = znorm(smat(qm, gm, qids, gids))
        for qi, qid in enumerate(qids):
            rows.append((qid, " ".join([gids[j] for j in np.argsort(-S[qi])])))
        print(f"  dataset2/{split} done", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "target_id_ranking"])
        for qid, r in rows: w.writerow([qid, r])
    print(f"\nwrote {len(rows)} rows -> {a.out}  (affine-to-template + SSC descriptor, d2-only)", flush=True)

if __name__ == "__main__": main()
