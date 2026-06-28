"""
v31: Non-ML cross-modal brain MRI retrieval (MRR 0.923, no machine learning).
SSC-12 (GPU) + MI (d1 only) + FFT + brain fingerprint + normalized distances
+ Hungarian assignment + affine registration (d2, d1 query as ref, shrink [8,4,2,1]).

100% deterministic, 0% ML, 0 training, 0 GPU training.
Runs in ~6 min on 20-core CPU with GPU-accelerated SSC-12 computation.

Developed with the assistance of GLM-5.2 (Z.AI), an open-source large language model.
Author: Wilfred Dore (wilfred.dore@telecom-paristech.org)
"""
from __future__ import annotations
import csv, json, time, os
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F_torch
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment

DATA_ROOT = Path("/root/data")
OUT = Path("/root/output/v33_submission.csv")
DEVICE = torch.device("cuda")
RES = 48

# ============================================================
# PREPROCESSING
# ============================================================

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def resolve_path(root, image_path):
    p = root / image_path
    if p.exists(): return p
    p2 = root / image_path.replace(".nii.gz", ".nii")
    if p2.exists(): return p2
    return p2

def resize_nearest(array, shape):
    if array.size == 0: return np.zeros(shape, dtype=np.float32)
    indices = [np.linspace(0, array.shape[i]-1, shape[i]).round().astype(np.int64) for i in range(array.ndim)]
    return array[np.ix_(*indices)].astype(np.float32, copy=False)

def robust_scale(volume, mask):
    values = volume[mask]
    if values.size < 16: values = volume[np.isfinite(volume)]
    if values.size == 0: return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        m, s = float(np.mean(values)), float(np.std(values)) + 1e-6
        return np.clip((volume - m) / (4*s) + 0.5, 0, 1).astype(np.float32)
    return np.clip((volume - lo) / (hi - lo), 0, 1).astype(np.float32)

def foreground_mask(volume):
    finite = np.isfinite(volume)
    nz = finite & (np.abs(volume) > 1e-6)
    if np.count_nonzero(nz) > 128:
        vals = np.abs(volume[nz])
        thr = max(1e-6, float(np.percentile(vals, 5.0)))
        m = finite & (np.abs(volume) >= thr)
        if np.count_nonzero(m) > 128: return m
    return nz if np.count_nonzero(nz) else finite

def load_volume(path, res=RES):
    img = nib.load(str(path))
    vol = np.asanyarray(img.dataobj, dtype=np.float32)
    if vol.ndim > 3: vol = vol[..., 0]
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    mask = foreground_mask(vol)
    scaled = robust_scale(vol, mask)
    coords = np.argwhere(mask)
    if coords.size > 0:
        mins = coords.min(axis=0); maxs = coords.max(axis=0)
        crop = tuple(slice(int(lo), int(hi)+1) for lo, hi in zip(mins, maxs))
        cropped = scaled[crop]
    else:
        cropped = scaled
    return resize_nearest(cropped, (res, res, res)).astype(np.float32)

# ============================================================
# SSC-12 DESCRIPTOR (GPU, 12-edge neighbour-to-neighbour)
# ============================================================

_SSC12_OFFSETS = [
    (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
    (1,1,0), (-1,-1,0), (1,-1,0), (-1,1,0),
    (1,0,1), (-1,0,-1),
]

def _box_filter_3d(x, radius):
    k = 2 * radius + 1
    kernel = torch.ones((1, 1, k, k, k), device=x.device, dtype=x.dtype) / float(k**3)
    return F_torch.conv3d(x, kernel, padding=radius)

def ssc12_descriptor_gpu(vol_np, radius=1, eps=1e-5):
    v = torch.from_numpy(vol_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    dp = []
    for dx, dy, dz in _SSC12_OFFSETS:
        shifted = torch.roll(v, shifts=(dx, dy, dz), dims=(2, 3, 4))
        patch_ssd = _box_filter_3d((v - shifted) ** 2, radius)
        dp.append(patch_ssd)
    dp = torch.cat(dp, dim=1)
    var = dp.mean(dim=1, keepdim=True).clamp_min(eps)
    ssc = torch.exp(-dp / var)
    ssc = ssc / ssc.amax(dim=1, keepdim=True).clamp_min(eps)
    return ssc[0]

def ssc12_dist_matrix(q_vols, t_vols):
    q_descs = [ssc12_descriptor_gpu(v) for v in q_vols]
    t_descs = [ssc12_descriptor_gpu(v) for v in t_vols]
    nq, nt = len(q_descs), len(t_descs)
    dist = np.zeros((nq, nt), dtype=np.float32)
    for i in range(nq):
        for j in range(nt):
            dist[i, j] = float((q_descs[i] - t_descs[j]).abs().mean())
    del q_descs, t_descs
    torch.cuda.empty_cache()
    return dist

# ============================================================
# ADDITIONAL FEATURES
# ============================================================

def gradient_magnitude(volume):
    gx = np.gradient(volume, axis=0)
    gy = np.gradient(volume, axis=1)
    gz = np.gradient(volume, axis=2)
    return np.sqrt(gx*gx + gy*gy + gz*gz).astype(np.float32)

def power_spectrum_feature(volume, n_radial=16):
    fft_vol = np.fft.fftn(volume.astype(np.float32))
    power = np.abs(fft_vol) ** 2
    power = np.fft.fftshift(power)
    power = np.log1p(power).astype(np.float32)
    h, w, d = power.shape
    cz, cy, cx = h//2, w//2, d//2
    zz, yy, xx = np.indices((h, w, d), dtype=np.float32)
    dists = np.sqrt((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2)
    max_dist = float(np.sqrt(cz**2 + cy**2 + cx**2)) + 1e-6
    norm_dists = dists / max_dist
    radial = np.zeros(n_radial, dtype=np.float32)
    edges = np.linspace(0, 1, n_radial+1)
    for r in range(n_radial):
        rm = (norm_dists >= edges[r]) & (norm_dists < edges[r+1]) if r < n_radial-1 else (norm_dists >= edges[r]) & (norm_dists <= edges[r+1])
        if rm.sum() > 0: radial[r] = float(power[rm].mean())
    radial /= radial.max() + 1e-8
    return radial

def brain_shape_fingerprint(mask, n_bins=16):
    coords = np.argwhere(mask)
    if coords.size == 0: return np.zeros(n_bins*4, dtype=np.float32)
    com = coords.mean(axis=0)
    dists = np.sqrt(((coords - com)**2).sum(axis=1))
    max_dist = float(dists.max()) + 1e-6
    norm_dists = dists / max_dist
    rh = np.histogram(norm_dists, bins=n_bins, range=(0,1))[0].astype(np.float32)
    rh /= max(rh.sum(), 1)
    zp = resize_nearest(mask.sum(axis=(1,2)).astype(np.float32), (n_bins,))
    yp = resize_nearest(mask.sum(axis=(0,2)).astype(np.float32), (n_bins,))
    xp = resize_nearest(mask.sum(axis=(0,1)).astype(np.float32), (n_bins,))
    for p in [zp, yp, xp]:
        s = p.sum()
        if s > 0: p /= s
    return np.concatenate([rh, zp, yp, xp]).astype(np.float32)

def compute_mi_matrix(q_vols, t_vols, q_masks, t_masks, n_bins=16, max_voxels=5000):
    nq, nt = len(q_vols), len(t_vols)
    mi_dist = np.zeros((nq, nt), dtype=np.float32)
    edges = np.linspace(0, 1, n_bins+1)
    for i in range(nq):
        for j in range(nt):
            qm = q_masks[i]; tm = t_masks[j]
            if qm.shape == tm.shape: common = qm & tm
            else: common = qm
            if common.sum() < 32: mi_dist[i,j] = 1.0; continue
            qc = q_vols[i][common]; tc = t_vols[j][common]
            if qc.size > max_voxels:
                idx = np.random.default_rng(42+i*1000+j).choice(qc.size, max_voxels, replace=False)
                qc = qc[idx]; tc = tc[idx]
            qb = np.digitize(qc, edges[1:-1]); tb = np.digitize(tc, edges[1:-1])
            joint = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(joint, (qb, tb), 1.0)
            joint /= joint.sum()
            p_q = joint.sum(axis=1); p_t = joint.sum(axis=0)
            mnz = joint > 1e-12
            mi = (joint[mnz] * np.log2(joint[mnz] / (p_q[:,None] * p_t[None,:])[mnz])).sum()
            mi_dist[i,j] = 1.0 / (1.0 + mi)
    return mi_dist

def extract_features(vol_3d):
    mask = vol_3d > 0.05
    if mask.sum() < 16: mask = np.ones_like(mask)
    scaled = np.clip(vol_3d, 0, 1)
    grad = gradient_magnitude(scaled)
    gmax = float(grad.max())
    if gmax > 1e-6: grad = (grad / gmax).astype(np.float32)
    ps = power_spectrum_feature(scaled)
    bf = brain_shape_fingerprint(mask)
    proj = np.concatenate([resize_nearest(scaled.mean(axis=a), (16,16)).reshape(-1) for a in range(3)]).astype(np.float32)
    vals = scaled[mask]
    if vals.size < 16: vals = scaled[np.isfinite(scaled)]
    h_hist, _ = np.histogram(vals, bins=32, range=(0.0, 1.0))
    hist = (h_hist / max(h_hist.sum(), 1)).astype(np.float32)
    return {
        "image": scaled.reshape(-1).astype(np.float32),
        "mask": mask.astype(np.float32).reshape(-1),
        "grad": grad.reshape(-1).astype(np.float32),
        "ps": ps, "brain": bf, "proj": proj, "hist": hist,
        "vol": scaled.astype(np.float32), "mask3d": mask,
    }

# ============================================================
# DISTANCE COMPUTATION (normalized)
# ============================================================

def cosine_dist(q, t):
    q = q - q.mean(axis=1, keepdims=True); t = t - t.mean(axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-6
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
    return 1.0 - q @ t.T

def chi2_dist(q, t):
    eps = 1e-8
    return (0.5 * ((q[:, None, :] - t[None, :, :]) ** 2 / (q[:, None, :] + t[None, :, :] + eps)).sum(axis=2))

def normalize_dist(D):
    d_min = D.min(); d_max = D.max()
    if d_max - d_min < 1e-8: return np.zeros_like(D)
    return (D - d_min) / (d_max - d_min)

# ============================================================
# REGISTRATION (d2 only, d1 query as reference, shrink [8,4,2,1])
# ============================================================

def register_to_ref(moving_np, fixed_np, iters=40):
    try:
        fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np.astype(np.float32)), sitk.sitkFloat32)
        moving = sitk.Cast(sitk.GetImageFromArray(moving_np.astype(np.float32)), sitk.sitkFloat32)
        R = sitk.ImageRegistrationMethod()
        init = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3),
                                                   sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R.SetMetricAsMattesMutualInformation(32)
        R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.15)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, iters)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
        R.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
        R.SetInitialTransform(init, inPlace=False)
        t = R.Execute(fixed, moving)
        out = sitk.Resample(moving, fixed, t, sitk.sitkLinear, 0.0, moving.GetPixelID())
        return sitk.GetArrayFromImage(out).astype(np.float32)
    except Exception:
        return moving_np

# ============================================================
# SINKHORN OPTIMAL TRANSPORT
# ============================================================

def hungarian_rank(similarity):
    """Hungarian optimal assignment (exact 1-to-1 bijection).
    Better than Sinkhorn on true bijections."""
    nq, nt = similarity.shape
    rows, cols = linear_sum_assignment(-similarity)
    assign = {int(r): int(c) for r, c in zip(rows, cols)}
    out = {}
    for i in range(nq):
        a = assign.get(i, int(np.argmax(similarity[i])))
        rest = [j for j in np.argsort(-similarity[i]).tolist() if j != a]
        out[i] = [a] + rest
    return out

# ============================================================
# WEIGHTS (Optuna-optimized + SSC=0.50 for d2/d3)
# ============================================================

D1_W = {"image": 0.10, "mask": 0.10, "grad": 0.15, "ps": 0.05, "brain": 0.05, "proj": 0.05, "hist": 0.05, "mi": 0.25, "ssc": 0.20}
D2_W = {"image": 0.10, "mask": 0.10, "grad": 0.05, "ps": 0.10, "brain": 0.05, "proj": 0.05, "hist": 0.05, "mi": 0.0, "ssc": 0.50}
D3_W = {"image": 0.05, "mask": 0.10, "grad": 0.05, "ps": 0.10, "brain": 0.10, "proj": 0.05, "hist": 0.05, "mi": 0.0, "ssc": 0.50}

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    t_start = time.time()
    print("=== v31: SSC-12 + MI + FFT + brain fp + normalized dist + Sinkhorn + d1 ref reg ===", flush=True)
    print(f"Resolution: {RES}^3, Device: {DEVICE}", flush=True)
    print(f"Weights D1: {D1_W}", flush=True)
    print(f"Weights D2: {D2_W}", flush=True)
    print(f"Weights D3: {D3_W}", flush=True)
    print()

    # Load d1 reference for d2 registration (Gowshigan's approach)
    d1_pairs = read_csv(DATA_ROOT / "dataset1" / "train_pairs.csv")
    ref_volume = load_volume(resolve_path(DATA_ROOT, d1_pairs[0]["query_image"]))
    print(f"Registration reference: {d1_pairs[0]['query_id']} (d1 train query)", flush=True)

    all_rows = []
    for ds in ["dataset1", "dataset2", "dataset3"]:
        for split in ["val", "test"]:
            qcsv = DATA_ROOT / ds / f"{split}_queries.csv"
            gcsv = DATA_ROOT / ds / f"{split}_gallery.csv"
            if not qcsv.exists(): continue
            t0 = time.time()
            queries = read_csv(qcsv)
            gallery = read_csv(gcsv)
            print(f"  {ds}/{split}: {len(queries)}q vs {len(gallery)}g", end="", flush=True)

            q_vols = [load_volume(resolve_path(DATA_ROOT, q["query_image"])) for q in queries]
            t_vols = [load_volume(resolve_path(DATA_ROOT, g["target_image"])) for g in gallery]

            # Registration ONLY for d2 (d1 and d3 don't need it)
            if ds in ("dataset2", "dataset3"):
                print(" (reg)", flush=True)
                q_vols = [register_to_ref(v, ref_volume) for v in q_vols]
                t_vols = [register_to_ref(v, ref_volume) for v in t_vols]
            else:
                print("", flush=True)

            # SSC-12 distance (GPU)
            ssc_dist = ssc12_dist_matrix(q_vols, t_vols)

            # Standard features
            qf = [{"id": q["query_id"], **extract_features(q_vols[i])} for i, q in enumerate(queries)]
            tf = [{"id": g["target_id"], **extract_features(t_vols[i])} for i, g in enumerate(gallery)]

            w = D1_W if ds == "dataset1" else (D2_W if ds == "dataset2" else D3_W)

            # Normalized distance combination
            D_total = np.zeros((len(qf), len(tf)), dtype=np.float32)
            for k in ["image", "mask", "grad", "ps", "brain", "proj"]:
                if w[k] > 0:
                    d = normalize_dist(cosine_dist(np.stack([f[k] for f in qf]), np.stack([f[k] for f in tf])))
                    D_total += w[k] * d
            if w["hist"] > 0:
                D_total += w["hist"] * normalize_dist(chi2_dist(np.stack([f["hist"] for f in qf]), np.stack([f["hist"] for f in tf])))
            if w["mi"] > 0:
                mi_dist = compute_mi_matrix([f["vol"] for f in qf], [f["vol"] for f in tf],
                                             [f["mask3d"] for f in qf], [f["mask3d"] for f in tf])
                D_total += w["mi"] * normalize_dist(mi_dist)
            if w["ssc"] > 0:
                D_total += w["ssc"] * normalize_dist(ssc_dist)

            # Hungarian assignment
            sim = -D_total
            sim = sim - sim.min()
            sim = sim / (sim.max() + 1e-8)
            rankings = hungarian_rank(sim)

            tids = [f["id"] for f in tf]
            for i, q in enumerate(qf):
                order = rankings[i]
                all_rows.append({"query_id": q["id"], "target_id_ranking": " ".join(tids[j] for j in order)})
            print(f"    {time.time()-t0:.0f}s", flush=True)
            del q_vols, t_vols
            torch.cuda.empty_cache()

    from csv import DictWriter
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader(); writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {OUT}", flush=True)
    print(f"Total: {time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
