from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def gaussian_filter(arr, sigma):
        from numpy import roll
        s = max(1, int(sigma * 3))
        out = arr.copy()
        for ax in range(arr.ndim):
            kernel = np.exp(-np.arange(-s, s+1)**2 / (2*sigma**2))
            kernel /= kernel.sum()
            for _ in range(arr.shape[ax]):
                pass
        return out

INPUT_ROOT = Path("/root/data")
OUT = Path("/root/output/v6_submission.csv")

DATASETS = ("dataset1", "dataset2", "dataset3")
SPLITS = ("val", "test")
RES = 48
N_BINS_MI = 32
MIND_RADIUS = 3
MIND_GRID = 6
MIND_PATCH = 3


@dataclass(frozen=True)
class ImageFeature:
    image_id: str
    meta: np.ndarray
    mask_r: np.ndarray
    image_r: np.ndarray
    grad_r: np.ndarray
    proj: np.ndarray
    hist: np.ndarray
    mind: np.ndarray
    brain_shape: np.ndarray
    vol_3d: np.ndarray
    power_spectrum: np.ndarray
    asymmetry: np.ndarray
    image_16: np.ndarray
    image_32: np.ndarray


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(root: Path, image_path: str) -> Path:
    candidates = [
        root / image_path,
        root / image_path.replace(".nii.gz", ".nii"),
        root / Path(image_path).name,
        root / Path(image_path.replace(".nii.gz", ".nii")).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[1]


def resize_nearest(array: np.ndarray, shape: tuple) -> np.ndarray:
    if array.size == 0:
        return np.zeros(shape, dtype=np.float32)
    indices = [np.linspace(0, array.shape[i] - 1, shape[i]).round().astype(np.int64) for i in range(array.ndim)]
    return array[np.ix_(*indices)].astype(np.float32, copy=False)


def robust_scale(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = volume[mask]
    if values.size < 16:
        values = volume[np.isfinite(volume)]
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mean, std = float(np.mean(values)), float(np.std(values)) + 1e-6
        return np.clip((volume - mean) / (4.0 * std) + 0.5, 0.0, 1.0).astype(np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def foreground_mask(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    nonzero = finite & (np.abs(volume) > 1e-6)
    if np.count_nonzero(nonzero) > 128:
        values = np.abs(volume[nonzero])
        threshold = max(1e-6, float(np.percentile(values, 5.0)))
        mask = finite & (np.abs(volume) >= threshold)
        if np.count_nonzero(mask) > 128:
            return mask
    return nonzero if np.count_nonzero(nonzero) else finite


def bbox_features(mask: np.ndarray):
    coords = np.argwhere(mask)
    shape = np.array(mask.shape, dtype=np.float32)
    if coords.size == 0:
        return np.zeros(10, dtype=np.float32), tuple(slice(0, int(s)) for s in mask.shape)
    mins, maxs = coords.min(axis=0).astype(np.float32), coords.max(axis=0).astype(np.float32)
    center = (mins + maxs) / 2.0
    extent = np.maximum(maxs - mins + 1.0, 1.0)
    com = coords.mean(axis=0).astype(np.float32)
    denom = np.maximum(shape - 1.0, 1.0)
    fg_frac = np.array([coords.shape[0] / max(float(np.prod(shape)), 1.0)], dtype=np.float32)
    meta = np.concatenate([center / denom, extent / np.maximum(shape, 1.0), com / denom, fg_frac]).astype(np.float32)
    crop = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(mins, maxs))
    return meta, crop


def projection_features(volume: np.ndarray) -> np.ndarray:
    projections = [resize_nearest(volume.mean(axis=a), (16, 16)).reshape(-1) for a in range(3)]
    return np.concatenate(projections).astype(np.float32)


def gradient_magnitude(volume: np.ndarray) -> np.ndarray:
    gx = np.gradient(volume, axis=0)
    gy = np.gradient(volume, axis=1)
    gz = np.gradient(volume, axis=2)
    return np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)


def intensity_histogram(volume: np.ndarray, mask: np.ndarray, n_bins: int = 32) -> np.ndarray:
    vals = volume[mask]
    if vals.size < 16:
        vals = volume[np.isfinite(volume)]
    if vals.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    h, _ = np.histogram(vals, bins=n_bins, range=(0.0, 1.0))
    return (h / max(h.sum(), 1)).astype(np.float32)


def compute_mind_descriptor(volume: np.ndarray, mask: np.ndarray, radius: int = MIND_RADIUS) -> np.ndarray:
    """Modality-Independent Neighbourhood Descriptor.
    Computes local self-similarity which is contrast-invariant.
    Returns a FIXED-SIZE descriptor by sampling on a regular grid."""
    vol = volume.copy().astype(np.float32)
    if HAS_SCIPY:
        vol = gaussian_filter(vol, sigma=1.0)
    
    h, w, d = vol.shape
    n_dirs = 6
    offsets = [
        (radius, 0, 0), (-radius, 0, 0),
        (0, radius, 0), (0, -radius, 0),
        (0, 0, radius), (0, 0, -radius),
    ]
    
    mind = np.zeros((h, w, d, n_dirs), dtype=np.float32)
    for i, (dz, dy, dx) in enumerate(offsets):
        shifted = np.roll(vol, (dz, dy, dx), axis=(0, 1, 2))
        diff_sq = (vol - shifted) ** 2
        mind[..., i] = diff_sq
    
    mind = mind / (mind.max() + 1e-8)
    mind = np.exp(-mind * 10.0).astype(np.float32)
    
    # Fixed grid sampling: MIND_GRID^3 points, each with 6 dirs
    n_samples = MIND_GRID
    samples = np.zeros((n_samples, n_samples, n_samples, n_dirs), dtype=np.float32)
    for iz, z in enumerate(np.linspace(0, h - 1, n_samples).astype(int)):
        for iy, y in enumerate(np.linspace(0, w - 1, n_samples).astype(int)):
            for ix, x in enumerate(np.linspace(0, d - 1, n_samples).astype(int)):
                samples[iz, iy, ix, :] = mind[z, y, x, :]
    
    return samples.reshape(-1)


def hemispheric_asymmetry(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Measure left-right brain asymmetry. Each brain has unique asymmetry pattern.
    Neuroscience: cortical folding patterns are individually unique."""
    h, w, d = volume.shape
    mid = w // 2
    left = volume[:, :mid, :]
    right = volume[:, mid:, :]
    if left.shape != right.shape:
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w, :]
        right = right[:, :min_w, :]
    
    left_mask = mask[:, :mid, :]
    right_mask = mask[:, mid:]
    if left_mask.shape != right_mask.shape:
        min_w = min(left_mask.shape[1], right_mask.shape[1])
        left_mask = left_mask[:, :min_w, :]
        right_mask = right_mask[:, :min_w, :]
    
    left_vals = left[left_mask] if left_mask.sum() > 16 else left.flatten()
    right_vals = right[right_mask] if right_mask.sum() > 16 else right.flatten()
    
    features = []
    features.append(float(np.mean(left_vals) - np.mean(right_vals)))
    features.append(float(np.std(left_vals) - np.std(right_vals)))
    features.append(float(np.percentile(left_vals, 25) - np.percentile(right_vals, 25)))
    features.append(float(np.percentile(left_vals, 75) - np.percentile(right_vals, 75)))
    features.append(float(np.median(left_vals) - np.median(right_vals)))
    features.append(float(left_mask.sum() - right_mask.sum()) / max(float(mask.sum()), 1.0))
    
    left_proj = resize_nearest(left.mean(axis=1), (8, 8)).flatten()
    right_proj = resize_nearest(right.mean(axis=1), (8, 8)).flatten()
    if left_proj.shape == right_proj.shape:
        features.extend((left_proj - right_proj).tolist())
    else:
        features.extend([0.0] * 64)
    
    return np.array(features[:10], dtype=np.float32)


def power_spectrum_feature(volume: np.ndarray, n_radial: int = 16, n_angular: int = 8) -> np.ndarray:
    """3D power spectrum with radial + angular binning.
    Translation-invariant by construction (FFT magnitude).
    Rotation in freq domain, radial mean is rotation-invariant."""
    fft_vol = np.fft.fftn(volume.astype(np.float32))
    power = np.abs(fft_vol) ** 2
    power = np.fft.fftshift(power)
    power = np.log1p(power).astype(np.float32)
    
    h, w, d = power.shape
    cz, cy, cx = h // 2, w // 2, d // 2
    zz, yy, xx = np.indices((h, w, d), dtype=np.float32)
    dists = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = float(np.sqrt(cz ** 2 + cy ** 2 + cx ** 2))
    norm_dists = dists / max_dist
    
    radial_profile = np.zeros(n_radial, dtype=np.float32)
    radial_edges = np.linspace(0, 1, n_radial + 1)
    for r in range(n_radial):
        if r < n_radial - 1:
            r_mask = (norm_dists >= radial_edges[r]) & (norm_dists < radial_edges[r + 1])
        else:
            r_mask = (norm_dists >= radial_edges[r]) & (norm_dists <= radial_edges[r + 1])
        if r_mask.sum() > 0:
            radial_profile[r] = float(power[r_mask].mean())
    
    radial_profile /= radial_profile.max() + 1e-8
    
    azim_profile = np.zeros(n_angular, dtype=np.float32)
    angles = np.arctan2(yy - cy, xx - cx)
    azim_edges = np.linspace(-np.pi, np.pi, n_angular + 1)
    for a in range(n_angular):
        if a < n_angular - 1:
            a_mask = (angles >= azim_edges[a]) & (angles < azim_edges[a + 1])
        else:
            a_mask = (angles >= azim_edges[a]) & (angles <= azim_edges[a + 1])
        inner_mask = a_mask & (norm_dists < 0.5)
        if inner_mask.sum() > 0:
            azim_profile[a] = float(power[inner_mask].mean())
    azim_profile /= azim_profile.max() + 1e-8
    
    return np.concatenate([radial_profile, azim_profile]).astype(np.float32)


def brain_shape_fingerprint(mask: np.ndarray, n_bins: int = 16) -> np.ndarray:
    shape = mask.shape
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros(n_bins * 4, dtype=np.float32)
    
    com = coords.mean(axis=0)
    dists = np.sqrt(((coords - com) ** 2).sum(axis=1))
    max_dist = float(dists.max()) + 1e-6
    norm_dists = dists / max_dist
    
    radial_hist = np.histogram(norm_dists, bins=n_bins, range=(0, 1))[0].astype(np.float32)
    radial_hist /= max(radial_hist.sum(), 1)
    
    z_proj = mask.sum(axis=(1, 2)).astype(np.float32)
    y_proj = mask.sum(axis=(0, 2)).astype(np.float32)
    x_proj = mask.sum(axis=(0, 1)).astype(np.float32)
    
    z_proj = resize_nearest(z_proj, (n_bins,))
    y_proj = resize_nearest(y_proj, (n_bins,))
    x_proj = resize_nearest(x_proj, (n_bins,))
    
    for p in [z_proj, y_proj, x_proj]:
        s = p.sum()
        if s > 0:
            p /= s
    
    return np.concatenate([radial_hist, z_proj, y_proj, x_proj]).astype(np.float32)


def mutual_information(vol_q: np.ndarray, vol_t: np.ndarray, mask: np.ndarray, n_bins: int = N_BINS_MI) -> float:
    """Compute MI between two volumes over a common mask."""
    q_vals = vol_q[mask]
    t_vals = vol_t[mask]
    if q_vals.size < 32:
        return 0.0
    
    q_binned = np.digitize(q_vals, np.linspace(0, 1, n_bins + 1)[1:-1])
    t_binned = np.digitize(t_vals, np.linspace(0, 1, n_bins + 1)[1:-1])
    
    joint = np.zeros((n_bins, n_bins), dtype=np.float64)
    for qb, tb in zip(q_binned, t_binned):
        joint[qb, tb] += 1
    joint /= joint.sum()
    
    p_q = joint.sum(axis=1)
    p_t = joint.sum(axis=0)
    
    mask_nz = joint > 0
    mi = (joint[mask_nz] * np.log2(joint[mask_nz] / (p_q[:, None] * p_t[None, :])[mask_nz])).sum()
    return float(mi)


def extract_feature(image_id: str, path: Path, crop: bool) -> ImageFeature:
    img = nib.load(str(path))
    volume = np.asanyarray(img.dataobj, dtype=np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = foreground_mask(volume)
    bbox_meta, crop_region = bbox_features(mask)

    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    shape = np.array(volume.shape[:3], dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)
    affine_meta = np.concatenate([
        np.log1p(shape) / 8.0,
        np.log1p(np.maximum(zooms, 1e-6)),
        affine[:3, :3].reshape(-1) / 256.0,
        affine[:3, 3] / 256.0,
    ])
    meta = np.concatenate([affine_meta, bbox_meta]).astype(np.float32)

    scaled = robust_scale(volume, mask)

    if not crop:
        vol32_raw = resize_nearest(volume, (RES, RES, RES))
        mask32_vol = resize_nearest(mask.astype(np.float32), (RES, RES, RES))
        mask32_bool = mask32_vol > 0.5
        if mask32_bool.sum() < 16:
            mask32_bool = np.ones((RES, RES, RES), dtype=bool)
        image_vol = robust_scale(vol32_raw, mask32_bool)
        mask_vol = mask32_vol
    else:
        vol_cropped = scaled[crop_region]
        mask_cropped = mask[crop_region].astype(np.float32)
        image_vol = resize_nearest(vol_cropped, (RES, RES, RES))
        mask_vol = resize_nearest(mask_cropped, (RES, RES, RES))

    mask32_bool = mask_vol > 0.5
    if mask32_bool.sum() < 16:
        mask32_bool = np.ones((RES, RES, RES), dtype=bool)

    grad_vol = gradient_magnitude(image_vol)
    grad_max = float(grad_vol.max())
    if grad_max > 1e-6:
        grad_vol = (grad_vol / grad_max).astype(np.float32)

    proj = projection_features(image_vol)
    hist = intensity_histogram(image_vol, mask32_bool)
    
    mind_desc = compute_mind_descriptor(image_vol, mask32_bool)
    brain_fp = brain_shape_fingerprint(mask32_bool)
    ps_feat = power_spectrum_feature(image_vol)
    asymmetry = hemispheric_asymmetry(image_vol, mask32_bool)
    
    image_16 = resize_nearest(image_vol, (16, 16, 16)).reshape(-1)
    image_32 = resize_nearest(image_vol, (32, 32, 32)).reshape(-1)

    return ImageFeature(
        image_id,
        meta,
        mask_vol.reshape(-1).astype(np.float32),
        image_vol.reshape(-1).astype(np.float32),
        grad_vol.reshape(-1).astype(np.float32),
        proj,
        hist,
        mind_desc,
        brain_fp,
        image_vol.astype(np.float32),
        ps_feat,
        asymmetry,
        image_16.astype(np.float32),
        image_32.astype(np.float32),
    )


def cosine_dist(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = q - q.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-6
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
    return 1.0 - q @ t.T


def l2_dist(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    combined = np.vstack([q, t])
    mean, std = combined.mean(0, keepdims=True), combined.std(0, keepdims=True) + 1e-6
    q, t = (q - mean) / std, (t - mean) / std
    return ((q[:, None, :] - t[None, :, :]) ** 2).mean(2)


def chi2_dist(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return (0.5 * ((q[:, None, :] - t[None, :, :]) ** 2 / (q[:, None, :] + t[None, :, :] + eps)).sum(axis=2))


def compute_mi_matrix(q_vols, t_vols, q_masks, t_masks, n_bins=16, max_voxels=5000):
    """Compute MI distance matrix (1/(1+MI)) between all query-target pairs.
    Subsamples voxels for speed."""
    nq, nt = len(q_vols), len(t_vols)
    mi_dist = np.zeros((nq, nt), dtype=np.float32)
    edges = np.linspace(0, 1, n_bins + 1)
    
    for i in range(nq):
        qv = q_vols[i]
        qm = q_masks[i]
        q_vals = qv[qm]
        if q_vals.size > max_voxels:
            idx = np.random.default_rng(42).choice(q_vals.size, max_voxels, replace=False)
            q_vals = q_vals[idx]
        q_binned_full = np.digitize(q_vals, edges[1:-1])
        
        for j in range(nt):
            tv = t_vols[j]
            tm = t_masks[j]
            if qm.shape == tm.shape:
                common_mask = qm & tm
            else:
                common_mask = qm
            if common_mask.sum() < 32:
                mi_dist[i, j] = 1.0
                continue
            
            q_common = qv[common_mask]
            t_common = tv[common_mask]
            if q_common.size > max_voxels:
                idx = np.random.default_rng(42 + i * 1000 + j).choice(q_common.size, max_voxels, replace=False)
                q_common = q_common[idx]
                t_common = t_common[idx]
            
            qb = np.digitize(q_common, edges[1:-1])
            tb = np.digitize(t_common, edges[1:-1])
            
            joint = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(joint, (qb, tb), 1.0)
            joint /= joint.sum()
            p_q = joint.sum(axis=1)
            p_t = joint.sum(axis=0)
            mask_nz = joint > 1e-12
            mi = (joint[mask_nz] * np.log2(joint[mask_nz] / (p_q[:, None] * p_t[None, :])[mask_nz])).sum()
            mi_dist[i, j] = 1.0 / (1.0 + mi)
    return mi_dist


def precompute_distances(qf, tf, use_mi=True):
    qi, ti = np.stack([f.image_r for f in qf]), np.stack([f.image_r for f in tf])
    qm, tm = np.stack([f.mask_r for f in qf]), np.stack([f.mask_r for f in tf])
    qg, tg = np.stack([f.grad_r for f in qf]), np.stack([f.grad_r for f in tf])
    qp, tp = np.stack([f.proj for f in qf]), np.stack([f.proj for f in tf])
    qme, tme = np.stack([f.meta for f in qf]), np.stack([f.meta for f in tf])
    qh, th = np.stack([f.hist for f in qf]), np.stack([f.hist for f in tf])
    qmind, tmind = np.stack([f.mind for f in qf]), np.stack([f.mind for f in tf])
    qbf, tbf = np.stack([f.brain_shape for f in qf]), np.stack([f.brain_shape for f in tf])
    qps, tps = np.stack([f.power_spectrum for f in qf]), np.stack([f.power_spectrum for f in tf])
    qasym, tasym = np.stack([f.asymmetry for f in qf]), np.stack([f.asymmetry for f in tf])
    q16, t16 = np.stack([f.image_16 for f in qf]), np.stack([f.image_16 for f in tf])
    q32, t32 = np.stack([f.image_32 for f in qf]), np.stack([f.image_32 for f in tf])
    
    dists = {
        "image": cosine_dist(qi, ti),
        "mask": cosine_dist(qm, tm),
        "grad": cosine_dist(qg, tg),
        "proj": cosine_dist(qp, tp),
        "meta": l2_dist(qme, tme),
        "hist": chi2_dist(qh, th),
        "mind": l2_dist(qmind, tmind),
        "brain": cosine_dist(qbf, tbf),
        "mi": np.zeros((len(qf), len(tf)), dtype=np.float32),
        "ps": cosine_dist(qps, tps),
        "asym": l2_dist(qasym, tasym),
        "img16": cosine_dist(q16, t16),
        "img32": cosine_dist(q32, t32),
    }
    
    if use_mi:
        q_vols = [f.vol_3d for f in qf]
        t_vols = [f.vol_3d for f in tf]
        q_masks_3d = [f.mask_r.reshape(RES, RES, RES) > 0.5 for f in qf]
        t_masks_3d = [f.mask_r.reshape(RES, RES, RES) > 0.5 for f in tf]
        dists["mi"] = compute_mi_matrix(q_vols, t_vols, q_masks_3d, t_masks_3d)
    
    return dists


def score_from_distances(distances, w):
    return (
        w[0] * distances["image"]
        + w[1] * distances["mask"]
        + w[2] * distances["grad"]
        + w[3] * distances["proj"]
        + w[4] * distances["meta"]
        + w[5] * distances["hist"]
        + w[6] * distances["mind"]
        + w[7] * distances["brain"]
        + w[8] * distances["mi"]
        + w[9] * distances["ps"]
        + w[10] * distances["asym"]
        + w[11] * distances["img16"]
        + w[12] * distances["img32"]
    )


def score_matrix(qf, tf, w, use_mi=False):
    return score_from_distances(precompute_distances(qf, tf, use_mi=use_mi), w)


def rank_pool(qf, tf, w, use_mi=False):
    D = score_matrix(qf, tf, w, use_mi=use_mi)
    tids = [f.image_id for f in tf]
    return [{"query_id": qf[i].image_id, "target_id_ranking": " ".join(tids[j] for j in np.argsort(D[i], kind="mergesort"))} for i in range(len(qf))]


from multiprocessing import Pool

def _extract_one(args):
    root, row, id_col, img_col, crop = args
    path = resolve_image_path(root, row[img_col])
    return extract_feature(row[id_col], path, crop)

def _extract_pairs_parallel(root, pairs, id_key, img_key, crop_mode):
    args_list = [(root, r, id_key, img_key, crop_mode) for r in pairs]
    with Pool(min(20, len(args_list))) as p:
        return p.map(_extract_one, args_list)

def load_features(root, csv_path, id_col, img_col, crop):
    rows = read_csv(csv_path)
    args_list = [(root, row, id_col, img_col, crop) for row in rows]
    with Pool(min(20, len(args_list))) as pool:
        features = pool.map(_extract_one, args_list)
    return features


D1_CANDIDATES = [
    (0.10, 0.20, 0.20, 0.05, 0.05, 0.05, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0),
    (0.05, 0.20, 0.20, 0.05, 0.05, 0.05, 0.0, 0.0, 0.35, 0.05, 0.0, 0.0, 0.0),
    (0.05, 0.15, 0.20, 0.05, 0.05, 0.05, 0.0, 0.0, 0.40, 0.05, 0.0, 0.0, 0.0),
    (0.05, 0.15, 0.15, 0.05, 0.05, 0.0, 0.0, 0.0, 0.40, 0.05, 0.05, 0.05, 0.05),
    (0.10, 0.15, 0.15, 0.05, 0.05, 0.05, 0.0, 0.0, 0.35, 0.05, 0.05, 0.05, 0.05),
    (0.05, 0.10, 0.15, 0.05, 0.05, 0.05, 0.0, 0.0, 0.40, 0.05, 0.05, 0.05, 0.05),
    (0.05, 0.15, 0.15, 0.05, 0.05, 0.05, 0.0, 0.0, 0.35, 0.05, 0.05, 0.10, 0.0),
    (0.05, 0.10, 0.15, 0.05, 0.05, 0.05, 0.0, 0.0, 0.35, 0.05, 0.05, 0.0, 0.10),
]

D2_CANDIDATES = [
    (0.15, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05, 0.0, 0.20, 0.0, 0.0, 0.0),
    (0.10, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05, 0.0, 0.20, 0.0, 0.05, 0.0),
    (0.10, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05, 0.0, 0.25, 0.0, 0.05, 0.0),
    (0.10, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.10, 0.0, 0.20, 0.05, 0.05, 0.0),
    (0.05, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.10, 0.0, 0.25, 0.05, 0.05, 0.0),
    (0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10, 0.10, 0.0, 0.25, 0.05, 0.05, 0.0),
    (0.10, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05, 0.0, 0.20, 0.05, 0.05, 0.05),
    (0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05, 0.0, 0.20, 0.05, 0.10, 0.10),
    (0.20, 0.25, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0),
    (0.15, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05, 0.0, 0.15, 0.0, 0.05, 0.0),
    (0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.10, 0.0, 0.15, 0.05, 0.0, 0.0),
    (0.10, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.0, 0.15, 0.05, 0.10, 0.10),
]

D3_CANDIDATES = [
    (0.10, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.15, 0.0, 0.20, 0.0, 0.0, 0.0),
    (0.05, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.20, 0.0, 0.20, 0.0, 0.05, 0.0),
    (0.05, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.20, 0.0, 0.15, 0.05, 0.05, 0.0),
    (0.05, 0.10, 0.05, 0.10, 0.05, 0.05, 0.10, 0.20, 0.0, 0.20, 0.05, 0.05, 0.0),
    (0.05, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10, 0.15, 0.0, 0.20, 0.05, 0.05, 0.05),
    (0.0, 0.10, 0.05, 0.10, 0.05, 0.05, 0.10, 0.25, 0.0, 0.20, 0.05, 0.05, 0.0),
    (0.0, 0.10, 0.05, 0.10, 0.05, 0.10, 0.10, 0.25, 0.0, 0.15, 0.05, 0.05, 0.0),
    (0.05, 0.10, 0.05, 0.10, 0.05, 0.10, 0.10, 0.20, 0.0, 0.20, 0.05, 0.10, 0.0),
    (0.10, 0.20, 0.10, 0.10, 0.05, 0.05, 0.05, 0.15, 0.0, 0.20, 0.0, 0.0, 0.0),
    (0.05, 0.15, 0.10, 0.10, 0.10, 0.05, 0.10, 0.20, 0.0, 0.10, 0.05, 0.0, 0.0),
    (0.0, 0.10, 0.05, 0.05, 0.05, 0.10, 0.15, 0.25, 0.0, 0.15, 0.05, 0.05, 0.0),
    (0.0, 0.05, 0.05, 0.05, 0.05, 0.10, 0.15, 0.30, 0.0, 0.10, 0.05, 0.05, 0.05),
]


def mrr_from_distances(distances, qf, tf, truth, w):
    D = score_from_distances(distances, w)
    tids = [f.image_id for f in tf]
    tid_array = np.array(tids, dtype=object)
    rr = []
    for i, f in enumerate(qf):
        order = np.argsort(D[i], kind="mergesort")
        rank = int(np.where(tid_array[order] == truth[f.image_id])[0][0]) + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def tune_dataset(root, name, candidates, crop_modes=(False, True), use_mi=False):
    print(f"=== Tuning {name} on train_pairs.csv (MI={use_mi}) ===")
    pairs = read_csv(root / "dataset1" / "train_pairs.csv")
    truth = {r["query_id"]: r["target_id"] for r in pairs}

    best = {"mrr": -1, "crop": None, "w": None}
    for crop_mode in crop_modes:
        label = "no-crop" if not crop_mode else "crop"
        t0 = time.time()
        qf = _extract_pairs_parallel(root, pairs, "query_id", "query_image", crop_mode)
        tf = _extract_pairs_parallel(root, pairs, "target_id", "target_image", crop_mode)
        t_feat = time.time() - t0
        distances = precompute_distances(qf, tf, use_mi=use_mi)
        results = sorted([{"w": w, "mrr": mrr_from_distances(distances, qf, tf, truth, w)} for w in candidates], key=lambda x: x["mrr"], reverse=True)
        print(f"  {label} ({t_feat:.0f}s): top mrr={results[0]['mrr']:.4f} w={results[0]['w']}")
        if results[0]["mrr"] > best["mrr"]:
            best = {"mrr": results[0]["mrr"], "crop": crop_mode, "w": results[0]["w"]}
    print(f"  BEST {name}: crop={best['crop']} mrr={best['mrr']:.4f} w={best['w']}")
    return best["w"], best["crop"]


def main():
    t_start = time.time()
    print(f"INPUT_ROOT: {INPUT_ROOT}  exists={INPUT_ROOT.exists()}")
    print(f"Resolution: {RES}^3, MI bins: {N_BINS_MI}, MIND radius: {MIND_RADIUS}")
    print(f"Features: volume+mask+gradient+proj+meta+hist+MIND+brain_fingerprint\n")

    d1_w, d1_crop = tune_dataset(INPUT_ROOT, "d1", D1_CANDIDATES, crop_modes=(False, True), use_mi=True)
    d2_w, _ = tune_dataset(INPUT_ROOT, "d2-like", D2_CANDIDATES, crop_modes=(True,), use_mi=False)
    d3_w, _ = tune_dataset(INPUT_ROOT, "d3-like", D3_CANDIDATES, crop_modes=(True,), use_mi=True)

    submission_rows = []
    summary = []
    for dataset in DATASETS:
        crop = d1_crop if dataset == "dataset1" else True
        w = d1_w if dataset == "dataset1" else (d2_w if dataset == "dataset2" else d3_w)
        for split in SPLITS:
            qcsv = INPUT_ROOT / dataset / f"{split}_queries.csv"
            gcsv = INPUT_ROOT / dataset / f"{split}_gallery.csv"
            if not qcsv.exists():
                continue
            t0 = time.time()
            print(f"=== {dataset}/{split} (crop={crop}) ===")
            qf = load_features(INPUT_ROOT, qcsv, "query_id", "query_image", crop)
            tf = load_features(INPUT_ROOT, gcsv, "target_id", "target_image", crop)
            use_mi = dataset in ("dataset1", "dataset3")
            rows = rank_pool(qf, tf, w, use_mi=use_mi)
            submission_rows.extend(rows)
            elapsed = time.time() - t0
            summary.append({"dataset": dataset, "split": split, "queries": len(qf), "targets": len(tf), "weights": list(w), "time_s": round(elapsed, 1)})
            print(f"  ranked {len(qf)}q vs {len(tf)}t in {elapsed:.1f}s w={w}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader()
        writer.writerows(submission_rows)
    print(json.dumps({
        "wrote": str(OUT),
        "rows": len(submission_rows),
        "total_time_s": round(time.time() - t_start, 1),
        "d1_weights": list(d1_w),
        "d2_weights": list(d2_w),
        "d3_weights": list(d3_w),
        "d1_crop": d1_crop,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
