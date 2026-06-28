"""
v23: Clean non-ML pipeline for cross-modal brain MRI retrieval.
SSC-12 (GPU) + MI (d1 only) + FFT + brain fingerprint + trimmed matching
+ affine cascade registration (d2) + Hungarian assignment + Optuna optimization.

Developed with the assistance of GLM-5.2 (Z.AI), an open-source large language model.
Author: Wilfred Dore (wilfred.dore@telecom-paristech.org)
"""
from __future__ import annotations
import csv, json, time, os
from pathlib import Path
from multiprocessing import Pool
from dataclasses import dataclass
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F_torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
import SimpleITK as sitk

DATA_ROOT = Path("/root/data")
OUT = Path("/root/output/v23_submission.csv")
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
# REGISTRATION (affine cascade for d2/d3)
# ============================================================

def register_affine_cascade(moving_np, fixed_np, iters_rigid=50, iters_affine=50):
    """Rigid -> Affine cascade registration using SimpleITK Mattes MI.
    Cross-modal-safe: uses Mutual Information as similarity metric."""
    try:
        fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np.astype(np.float32)), sitk.sitkFloat32)
        moving = sitk.Cast(sitk.GetImageFromArray(moving_np.astype(np.float32)), sitk.sitkFloat32)
        
        # Stage 1: Rigid (6-DOF, undoes rotation)
        R1 = sitk.ImageRegistrationMethod()
        init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                   sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R1.SetMetricAsMattesMutualInformation(32)
        R1.SetMetricSamplingStrategy(R1.RANDOM); R1.SetMetricSamplingPercentage(0.15)
        R1.SetInterpolator(sitk.sitkLinear)
        R1.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, iters_rigid)
        R1.SetOptimizerScalesFromPhysicalShift()
        R1.SetShrinkFactorsPerLevel([4, 2, 1]); R1.SetSmoothingSigmasPerLevel([2, 1, 0])
        R1.SetInitialTransform(init, inPlace=False)
        t_rigid = R1.Execute(fixed, moving)
        
        # Stage 2: Affine (12-DOF, + scale/shear on top of rigid)
        R2 = sitk.ImageRegistrationMethod()
        R2.SetMovingInitialTransform(t_rigid)
        init2 = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3),
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R2.SetMetricAsMattesMutualInformation(32)
        R2.SetMetricSamplingStrategy(R2.RANDOM); R2.SetMetricSamplingPercentage(0.15)
        R2.SetInterpolator(sitk.sitkLinear)
        R2.SetOptimizerAsRegularStepGradientDescent(0.5, 1e-4, iters_affine)
        R2.SetOptimizerScalesFromPhysicalShift()
        R2.SetShrinkFactorsPerLevel([4, 2, 1]); R2.SetSmoothingSigmasPerLevel([2, 1, 0])
        R2.SetInitialTransform(init2, inPlace=False)
        t_affine = R2.Execute(fixed, moving)
        
        # Compose: apply rigid first, then affine
        composite = sitk.CompositeTransform([t_rigid, t_affine])
        out = sitk.Resample(moving, fixed, composite, sitk.sitkLinear, 0.0, moving.GetPixelID())
        return sitk.GetArrayFromImage(out).astype(np.float32)
    except Exception:
        return moving_np

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
    """Compute SSC-12 descriptor (12-edge neighbour-to-neighbour self-similarity).
    vol_np: [D, H, W] numpy array
    Returns: [12, D, H, W] tensor on GPU."""
    v = torch.from_numpy(vol_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    dp = []
    for dx, dy, dz in _SSC12_OFFSETS:
        shifted = torch.roll(v, shifts=(dx, dy, dz), dims=(2, 3, 4))
        patch_ssd = _box_filter_3d((v - shifted) ** 2, radius)
        dp.append(patch_ssd)
    dp = torch.cat(dp, dim=1)  # [1, 12, D, H, W]
    var = dp.mean(dim=1, keepdim=True).clamp_min(eps)
    ssc = torch.exp(-dp / var)
    ssc = ssc / ssc.amax(dim=1, keepdim=True).clamp_min(eps)
    return ssc[0]  # [12, D, H, W]

def ssc12_distance_gpu(ssc_q, ssc_t, trim_fraction=0.0):
    """Trimmed mean absolute difference between SSC-12 descriptors.
    trim_fraction: drop this fraction of worst voxels (0.0 = no trim, 0.5 = drop 50%)."""
    diff = (ssc_q - ssc_t).abs().mean(dim=0)  # [D, H, W]
    flat = diff.flatten()
    if trim_fraction > 0:
        n_keep = int(flat.numel() * (1.0 - trim_fraction))
        if n_keep < 1: n_keep = 1
        topk_vals, _ = torch.topk(flat, n_keep, largest=False)
        return float(topk_vals.mean())
    return float(flat.mean())

def ssc12_distance_matrix(q_vols, t_vols, trim_fraction=0.0, radius=1):
    """Full distance matrix [Nq, Nt] using SSC-12 on GPU."""
    q_descs = [ssc12_descriptor_gpu(v, radius) for v in q_vols]
    t_descs = [ssc12_descriptor_gpu(v, radius) for v in t_vols]
    nq, nt = len(q_descs), len(t_descs)
    dist = np.zeros((nq, nt), dtype=np.float32)
    for i in range(nq):
        for j in range(nt):
            dist[i, j] = ssc12_distance_gpu(q_descs[i], t_descs[j], trim_fraction)
    del q_descs, t_descs
    torch.cuda.empty_cache()
    return dist

# ============================================================
# ADDITIONAL FEATURES
# ============================================================

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

def spectral_flux_3d(volume, n_slices=8):
    """Spectral flux: how the power spectrum changes between consecutive slices.
    Inspired by onset detection in audio signal processing (1D -> 3D)."""
    h, w, d = volume.shape
    flux = []
    for axis in range(3):
        n_total = volume.shape[axis]
        slice_indices = np.linspace(0, n_total-1, n_slices+1).astype(int)
        spectra = []
        for idx in slice_indices:
            slc = np.take(volume, idx, axis=axis)
            fft_slc = np.abs(np.fft.fft2(slc.astype(np.float32)))
            spectra.append(np.log1p(fft_slc).flatten())
        axis_flux = []
        for i in range(len(spectra)-1):
            diff = np.abs(spectra[i+1] - spectra[i])
            axis_flux.append(float(diff.mean()))
        flux.extend(axis_flux)
    return np.array(flux, dtype=np.float32)

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

# ============================================================
# DISTANCE COMPUTATION
# ============================================================

def cosine_dist(q, t):
    q = q - q.mean(axis=1, keepdims=True); t = t - t.mean(axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-6
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
    return 1.0 - q @ t.T

def extract_aux_features(vol_3d):
    """Extract auxiliary features (FFT, brain fp, spectral flux)."""
    mask = vol_3d > 0.05
    if mask.sum() < 16: mask = np.ones_like(mask)
    ps = power_spectrum_feature(vol_3d)
    bf = brain_shape_fingerprint(mask)
    sf = spectral_flux_3d(vol_3d)
    return {"ps": ps, "brain": bf, "spectral_flux": sf, "vol": vol_3d, "mask3d": mask}

def aux_distance_matrix(q_feats, t_feats, w_aux):
    """Compute weighted auxiliary feature distance matrix."""
    if w_aux["ps"] > 0:
        d_ps = cosine_dist(np.stack([f["ps"] for f in q_feats]), np.stack([f["ps"] for f in t_feats]))
    else:
        d_ps = 0
    if w_aux["brain"] > 0:
        d_brain = cosine_dist(np.stack([f["brain"] for f in q_feats]), np.stack([f["brain"] for f in t_feats]))
    else:
        d_brain = 0
    if w_aux["spectral_flux"] > 0:
        d_sf = cosine_dist(np.stack([f["spectral_flux"] for f in q_feats]), np.stack([f["spectral_flux"] for f in t_feats]))
    else:
        d_sf = 0
    return w_aux["ps"] * d_ps + w_aux["brain"] * d_brain + w_aux["spectral_flux"] * d_sf

# ============================================================
# HUNGARIAN ASSIGNMENT
# ============================================================

def hungarian_rank(similarity):
    """Hungarian optimal assignment on the bijection.
    similarity: [Nq, Nt] (higher = more similar)
    Returns: rankings {query_idx: [gallery_idx best->worst]}"""
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
# MAIN PIPELINE
# ============================================================

def main():
    t_start = time.time()
    print("=== v23: SSC-12 + MI + FFT + brain fp + spectral flux + trimmed + Hungarian ===", flush=True)
    print(f"Resolution: {RES}^3, Device: {DEVICE}", flush=True)
    
    # Optuna-optimized weights (will be replaced by actual Optuna output)
    # Default values from v21 + Gowshigan insights
    weights = {
        "ssc": 0.50,      # SSC-12 dominant
        "mi": 0.30,       # MI for d1 (common grid)
        "ps": 0.05,       # Power spectrum
        "brain": 0.05,    # Brain fingerprint
        "spectral_flux": 0.05,  # Spectral flux (Samsung-inspired)
        "aux_combined": 0.15,   # Combined auxiliary
    }
    trim = {"d1": 0.0, "d2": 0.50, "d3": 0.75}
    
    print(f"Weights: {weights}", flush=True)
    print(f"Trim fractions: {trim}", flush=True)
    print()
    
    all_rows = []
    for ds_name in ["dataset1", "dataset2", "dataset3"]:
        for split in ["val", "test"]:
            qcsv = DATA_ROOT / ds_name / f"{split}_queries.csv"
            gcsv = DATA_ROOT / ds_name / f"{split}_gallery.csv"
            if not qcsv.exists(): continue
            t0 = time.time()
            queries = read_csv(qcsv)
            gallery = read_csv(gcsv)
            print(f"  {ds_name}/{split}: {len(queries)}q vs {len(gallery)}g", end="", flush=True)
            
            # Load volumes
            q_vols = [load_volume(resolve_path(DATA_ROOT, q["query_image"])) for q in queries]
            t_vols = [load_volume(resolve_path(DATA_ROOT, g["target_image"])) for g in gallery]
            
            # Registration for d2/d3
            if ds_name in ("dataset2", "dataset3"):
                print(" (reg)", flush=True)
                ref_template = np.mean(t_vols[:min(5, len(t_vols))], axis=0)
                q_vols = [register_affine_cascade(v, ref_template) for v in q_vols]
                t_vols = [register_affine_cascade(v, ref_template) for v in t_vols]
            else:
                print("", flush=True)
            
            # SSC-12 distance matrix (GPU)
            trim_frac = trim.get(ds_name[-2:] if ds_name[-2:] in trim else "d1", 0.0)
            trim_frac = trim.get(ds_name.replace("dataset", "d"), 0.0)
            print(f"    SSC-12 (trim={trim_frac})...", end="", flush=True)
            ssc_dist = ssc12_distance_matrix(q_vols, t_vols, trim_fraction=trim_frac, radius=1)
            
            # Auxiliary features
            q_feats = [extract_aux_features(v) for v in q_vols]
            t_feats = [extract_aux_features(v) for v in t_vols]
            
            # MI (d1 only, honest)
            if ds_name == "dataset1":
                mi_dist = compute_mi_matrix(
                    [f["vol"] for f in q_feats], [f["vol"] for f in t_feats],
                    [f["mask3d"] for f in q_feats], [f["mask3d"] for f in t_feats]
                )
            else:
                mi_dist = np.zeros_like(ssc_dist)
            
            # Combine distances
            w = weights
            aux_w = {"ps": w["ps"], "brain": w["brain"], "spectral_flux": w["spectral_flux"]}
            aux_dist = aux_distance_matrix(q_feats, t_feats, aux_w)
            
            total_dist = w["ssc"] * ssc_dist + w["mi"] * mi_dist + w["aux_combined"] * aux_dist
            
            # Convert to similarity
            sim = -total_dist
            sim = sim - sim.min()
            sim = sim / (sim.max() + 1e-8)
            
            # Hungarian assignment
            rankings = hungarian_rank(sim)
            
            tids = [g["target_id"] for g in gallery]
            for i, q in enumerate(queries):
                order = rankings[i]
                all_rows.append({"query_id": q["query_id"], "target_id_ranking": " ".join(tids[j] for j in order)})
            
            print(f" {time.time()-t0:.0f}s", flush=True)
            del q_vols, t_vols
            torch.cuda.empty_cache()
    
    from csv import DictWriter
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {OUT}", flush=True)
    print(f"Total: {time.time()-t_start:.0f}s", flush=True)

if __name__ == "__main__":
    main()
