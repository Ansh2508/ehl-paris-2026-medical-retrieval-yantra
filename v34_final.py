"""
v34: Non-ML cross-modal brain MRI retrieval -- HONEST/DEPLOYABLE version.
Addresses Gowshigan's audit of v33:
  #1  d3 co-location leak: apply independent random rigid per volume before
      registering to the d1 reference (destroys the trivial co-location that
      survived v33's shared-ref registration). Trimmed SSC (keep best 25% of
      residuals) makes the distance resection-robust.
  #3  Registration cascade: MOMENTS-init rigid -> affine, 200 iters total,
      shrink [8,4,2,1]. Replaces fragile single-stage GEOMETRY-init affine.
  #4  Trilinear/linear resize (was nearest-neighbour -> aliasing).
  #5  d2/d3 fusion reduced to SSC + trim only (dead-weight features dropped).
  #6  Greedy row-wise argmax ranking replaces Hungarian (no eval-bijection
      exploitation; deployable for single-query use).

d1 keeps the full feature set (MI + SSC + fingerprint) since d1 pairs are not
co-located and the bijection-independent features genuinely help there.

100% deterministic, 0% ML, 0 training, 0 GPU training.
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

DATA_ROOT = Path("/root/data")
OUT = Path("/root/output/v34_submission.csv")
DEVICE = torch.device("cuda")
RES = 48
D3_TRIM_KEEP = 0.25          # keep best 25% of d3 SSC residuals (drop worst 75%)
D3_RIGID_AMP = np.pi / 6     # +/- 30 deg random rotation per d3 volume
D3_RIGID_TRANS = 5.0         # +/- 5 voxels random translation per d3 volume

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

def resize_linear(array, shape):
    """Anti-aliased resize via torch interpolate (linear/bilinear/trilinear).
    Replaces nearest-neighbour downsampling which introduced aliasing (#4)."""
    if array.size == 0:
        return np.zeros(shape, dtype=np.float32)
    arr = np.ascontiguousarray(array.astype(np.float32))
    ndim = arr.ndim
    if ndim == 1:
        t = torch.from_numpy(arr)[None, None, :]
        mode = "linear"
    elif ndim == 2:
        t = torch.from_numpy(arr)[None, None, :, :]
        mode = "bilinear"
    elif ndim == 3:
        t = torch.from_numpy(arr)[None, None, :, :, :]
        mode = "trilinear"
    else:
        raise ValueError(f"resize_linear: unsupported ndim {ndim}")
    t = F_torch.interpolate(t, size=tuple(shape), mode=mode, align_corners=False)
    out = t[0, 0].numpy()
    # torch interpolate preserves float32; ensure exact dtype
    return out.astype(np.float32, copy=False)

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
    return resize_linear(cropped, (res, res, res)).astype(np.float32)

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

def ssc12_dist_matrix(q_vols, t_vols, keep_frac=1.0):
    """SSC-12 L1 distance matrix. If keep_frac < 1.0, use a trimmed mean that
    keeps only the best (smallest) keep_frac fraction of per-voxel residuals
    (#1). Trimming makes the distance robust to resected/changed regions --
    the resection voxels produce large SSC residuals that are dropped instead
    of dominating the average."""
    q_descs = [ssc12_descriptor_gpu(v) for v in q_vols]
    t_descs = [ssc12_descriptor_gpu(v) for v in t_vols]
    nq, nt = len(q_descs), len(t_descs)
    dist = np.zeros((nq, nt), dtype=np.float32)
    do_trim = keep_frac < 1.0
    for i in range(nq):
        for j in range(nt):
            diff = (q_descs[i] - t_descs[j]).abs().reshape(-1)
            if do_trim:
                k = max(1, int(diff.numel() * keep_frac))
                dist[i, j] = float(diff.topk(k, largest=False)[0].mean())
            else:
                dist[i, j] = float(diff.mean())
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
    zp = resize_linear(mask.sum(axis=(1,2)).astype(np.float32), (n_bins,))
    yp = resize_linear(mask.sum(axis=(0,2)).astype(np.float32), (n_bins,))
    xp = resize_linear(mask.sum(axis=(0,1)).astype(np.float32), (n_bins,))
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
    proj = np.concatenate([resize_linear(scaled.mean(axis=a), (16,16)).reshape(-1) for a in range(3)]).astype(np.float32)
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
# REGISTRATION -- MOMENTS rigid->affine cascade (#3); random rigid for d3 (#1)
# ============================================================

def _run_registration_stage(fixed, moving, transform, init, iters):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, iters)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    return R.Execute(fixed, moving)

def register_to_ref(moving_np, fixed_np, iters_per_stage=100):
    """MOMENTS-init rigid -> affine cascade (#3).
    Total ~200 iters. Far more robust than single-stage GEOMETRY affine on
    hard d2 cases. Falls back to the moving volume on failure."""
    try:
        fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np.astype(np.float32)), sitk.sitkFloat32)
        moving = sitk.Cast(sitk.GetImageFromArray(moving_np.astype(np.float32)), sitk.sitkFloat32)

        # Stage 1: rigid, MOMENTS initialization (uses image intensity mass centre)
        rigid_init = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS)
        t_rigid = _run_registration_stage(fixed, moving, sitk.Euler3DTransform(),
                                          rigid_init, iters_per_stage)
        moving_r = sitk.Resample(moving, fixed, t_rigid, sitk.sitkLinear,
                                 0.0, moving.GetPixelID())

        # Stage 2: affine, MOMENTS initialization on the rigid-aligned moving
        aff_init = sitk.CenteredTransformInitializer(
            fixed, moving_r, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.MOMENTS)
        t_aff = _run_registration_stage(fixed, moving_r, sitk.AffineTransform(3),
                                        aff_init, iters_per_stage)

        out = sitk.Resample(moving_r, fixed, t_aff, sitk.sitkLinear,
                            0.0, moving.GetPixelID())
        return sitk.GetArrayFromImage(out).astype(np.float32)
    except Exception:
        return moving_np

def apply_random_rigid(vol_np, seed):
    """Apply an independent random rigid transform to a volume (#1).
    Used on d3 volumes before registration to the shared reference, so that
    the original co-location of query/target pairs is destroyed and the
    registration result depends on actual anatomy alignment, not on a
    dataset-induced shared pose. Deterministic per seed."""
    try:
        img = sitk.GetImageFromArray(vol_np.astype(np.float32))
        sz = vol_np.shape
        # sitk indexes (x,y,z) = (cols, rows, slices) -> reversed from numpy
        center = [float(sz[2] // 2), float(sz[1] // 2), float(sz[0] // 2)]
        rng = np.random.default_rng(seed)
        rx, ry, rz = rng.uniform(-D3_RIGID_AMP, D3_RIGID_AMP, 3)
        tx, ty, tz = rng.uniform(-D3_RIGID_TRANS, D3_RIGID_TRANS, 3)
        euler = sitk.Euler3DTransform()
        euler.SetCenter(center)
        euler.SetRotation(rx, ry, rz)
        euler.SetTranslation((tx, ty, tz))
        out = sitk.Resample(img, img, euler, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        return sitk.GetArrayFromImage(out).astype(np.float32)
    except Exception:
        return vol_np.astype(np.float32, copy=False)

# ============================================================
# SINKHORN OPTIMAL TRANSPORT  ->  GREEDY ROW-WISE RANKING (#6)
# ============================================================

def greedy_rank(similarity):
    """Greedy row-wise argmax ranking (#6).
    Each query is ranked independently by its own similarity scores -- no
    global 1-to-1 bijection is enforced. This is the deployable, single-query
    form: it does not exploit the evaluation's gallery bijection the way the
    Hungarian assignment did. MRR drops to the honest ~0.725 range but the
    ranking is valid for real-world single-query retrieval."""
    nq, nt = similarity.shape
    out = {}
    for i in range(nq):
        out[i] = np.argsort(-similarity[i]).tolist()
    return out

# ============================================================
# WEIGHTS -- d1 keeps full feature set; d2/d3 reduced to SSC + trim (#5)
# ============================================================

D1_W = {"image": 0.10, "mask": 0.10, "grad": 0.15, "ps": 0.05, "brain": 0.05, "proj": 0.05, "hist": 0.05, "mi": 0.25, "ssc": 0.20}
# d2/d3: dead-weight features dropped. SSC carries the full weight; the
# trimmed mean inside ssc12_dist_matrix provides the resection robustness
# that the 9-feature fusion was failing to deliver.
D2_W = {"image": 0.0,  "mask": 0.0,  "grad": 0.0,  "ps": 0.0,  "brain": 0.0,  "proj": 0.0,  "hist": 0.0,  "mi": 0.0,  "ssc": 1.0}
D3_W = {"image": 0.0,  "mask": 0.0,  "grad": 0.0,  "ps": 0.0,  "brain": 0.0,  "proj": 0.0,  "hist": 0.0,  "mi": 0.0,  "ssc": 1.0}

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    t_start = time.time()
    print("=== v34: HONEST/DEPLOYABLE -- SSC-12 + trimmed SSC (d3) + MOMENTS rigid->affine + greedy rank ===", flush=True)
    print(f"Resolution: {RES}^3, Device: {DEVICE}", flush=True)
    print(f"d3 trim keep fraction: {D3_TRIM_KEEP} (drop worst {(1-D3_TRIM_KEEP)*100:.0f}%)", flush=True)
    print(f"Weights D1: {D1_W}", flush=True)
    print(f"Weights D2: {D2_W}", flush=True)
    print(f"Weights D3: {D3_W}", flush=True)
    print()

    # Load d1 reference for d2/d3 registration (Gowshigan's approach)
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

            # Registration for d2 and d3 (#3: MOMENTS rigid->affine cascade).
            # For d3, apply an independent random rigid per volume first (#1)
            # so that the dataset's query/target co-location is destroyed
            # before the shared-ref registration -- the leak that survived
            # v33's registration no longer survives.
            if ds == "dataset3":
                print(" (rand-rigid + reg + trim)", flush=True)
                q_vols = [apply_random_rigid(v, 1000 + i) for i, v in enumerate(q_vols)]
                t_vols = [apply_random_rigid(v, 2000 + i) for i, v in enumerate(t_vols)]
                q_vols = [register_to_ref(v, ref_volume) for v in q_vols]
                t_vols = [register_to_ref(v, ref_volume) for v in t_vols]
            elif ds == "dataset2":
                print(" (reg)", flush=True)
                q_vols = [register_to_ref(v, ref_volume) for v in q_vols]
                t_vols = [register_to_ref(v, ref_volume) for v in t_vols]
            else:
                print("", flush=True)

            # SSC-12 distance (GPU). d3 uses the trimmed mean (#1) to be
            # robust to resections; d1/d2 use the plain mean.
            if ds == "dataset3":
                ssc_dist = ssc12_dist_matrix(q_vols, t_vols, keep_frac=D3_TRIM_KEEP)
            else:
                ssc_dist = ssc12_dist_matrix(q_vols, t_vols)

            # Standard features (only computed/used where the weights need them)
            w = D1_W if ds == "dataset1" else (D2_W if ds == "dataset2" else D3_W)
            need_features = any(w[k] > 0 for k in ("image", "mask", "grad", "ps", "brain", "proj", "hist", "mi"))
            if need_features:
                qf = [{"id": q["query_id"], **extract_features(q_vols[i])} for i, q in enumerate(queries)]
                tf = [{"id": g["target_id"], **extract_features(t_vols[i])} for i, g in enumerate(gallery)]
            else:
                # SSC-only path (#5): skip the dead-weight feature extraction
                qf = [{"id": q["query_id"]} for q in queries]
                tf = [{"id": g["target_id"]} for g in gallery]

            # Normalized distance combination
            D_total = np.zeros((len(qf), len(tf)), dtype=np.float32)
            if need_features:
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

            # Greedy row-wise ranking (#6): deployable, no bijection exploitation
            sim = -D_total
            rankings = greedy_rank(sim)

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
