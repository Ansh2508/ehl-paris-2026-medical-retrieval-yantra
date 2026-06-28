
import argparse, csv, os, time
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter

CFG = dict(mind_size=64, reg_iters=120)   # affine needs a few more iters than rigid

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

def mind_desc(vol, sigma=1.0, radius=2):
    vol = gaussian_filter(vol, sigma)
    offs = [(radius,0,0),(-radius,0,0),(0,radius,0),(0,-radius,0),(0,0,radius),(0,0,-radius)]
    feats = np.stack([gaussian_filter((vol - np.roll(vol, o, axis=(0,1,2)))**2, sigma) for o in offs], axis=-1)
    var = np.mean(feats, axis=-1, keepdims=True) + 1e-8
    m = np.exp(-feats / var); m = m / (m.sum(axis=-1, keepdims=True) + 1e-8)
    return m.astype(np.float32).ravel()

def _sitk(arr): return sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkFloat32)

def register_affine(moving_arr, fixed_img, iters):
    """Rigid init -> AFFINE refinement (12-DOF: rotation+translation+scale+shear), Mattes MI."""
    m = _sitk(moving_arr)
    # stage 1: rigid initialization (stabilizes the affine)
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
    # stage 2: affine, initialized from rigid
    aff = sitk.AffineTransform(3)
    R2 = sitk.ImageRegistrationMethod()
    R2.SetMovingInitialTransform(rigid_t)
    R2.SetInitialTransform(aff, inPlace=True)
    R2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R2.SetMetricSamplingStrategy(R2.RANDOM); R2.SetMetricSamplingPercentage(0.1)
    R2.SetInterpolator(sitk.sitkLinear)
    R2.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5, minStep=1e-4, numberOfIterations=iters)
    R2.SetOptimizerScalesFromPhysicalShift()
    R2.SetShrinkFactorsPerLevel([4,2,1]); R2.SetSmoothingSigmasPerLevel([2,1,0])
    try:
        aff_t = R2.Execute(fixed_img, m)
        total = sitk.CompositeTransform([rigid_t, aff_t])
    except Exception:
        total = rigid_t
    out = sitk.Resample(m, fixed_img, total, sitk.sitkLinear, 0.0, m.GetPixelID())
    return sitk.GetArrayFromImage(out)

def mind_registered(imgs, size, ref_arr, iters, tag):
    ref_img = _sitk(ref_arr); out = {}; t0 = time.time()
    for i, (k, p) in enumerate(imgs.items()):
        out[k] = mind_desc(register_affine(load_raw(p, size), ref_img, iters))
        if i % 20 == 0: print(f"  {tag} affine+mind {i}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
    return out

def smat(qe, te, qids, gids):
    Q = np.stack([qe[i] for i in qids]); G = np.stack([te[i] for i in gids])
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    return Q @ G.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="submission_dataset2_affinemind.csv")
    a = ap.parse_args(); root = Path(a.data_root).resolve()
    rows = []
    for split in ["val", "test"]:
        qc = read(root/f"dataset2/{split}_queries.csv"); gc = read(root/f"dataset2/{split}_gallery.csv")
        qids = [r["query_id"] for r in qc]; gids = [r["target_id"] for r in gc]
        qimg = {r["query_id"]: fix_path(root, r["query_image"]) for r in qc}
        gimg = {r["target_id"]: fix_path(root, r["target_image"]) for r in gc}
        # common reference = first gallery volume of this split (register q AND gallery to it)
        ref = load_raw(list(gimg.values())[0], CFG["mind_size"])
        print(f"dataset2/{split}: AFFINE register {len(qimg)} queries + {len(gimg)} gallery -> MIND", flush=True)
        gm = mind_registered(gimg, CFG["mind_size"], ref, CFG["reg_iters"], f"{split}/gal")
        qm = mind_registered(qimg, CFG["mind_size"], ref, CFG["reg_iters"], f"{split}/qry")
        S = znorm(smat(qm, gm, qids, gids))   # pure affine + MIND (Rafay's method)
        for qi, qid in enumerate(qids):
            rows.append((qid, " ".join([gids[j] for j in np.argsort(-S[qi])])))
        print(f"  dataset2/{split} done", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "target_id_ranking"])
        for qid, r in rows: w.writerow([qid, r])
    print(f"\nwrote {len(rows)} rows -> {a.out}  (affine+MIND, d2-only)", flush=True)

if __name__ == "__main__": main()
