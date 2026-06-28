"""Load + normalize + resample 3D NIfTI volumes, with a disk cache.

Robust to: variable input shapes, and the manifest/disk extension mismatch
(train_pairs.csv references `.nii.gz` while the extracted files are `.nii`).
See CLAUDE.md §5.  Run `python preprocess.py --dry-run ...` to sanity-check shapes.
"""
from __future__ import annotations
from pathlib import Path
import torch
from config import CFG


def resolve_path(path: str | Path, cfg=CFG) -> Path:
    """Resolve a manifest image path against cfg.data_root, tolerating the
    `.nii.gz` (manifest) vs `.nii` (on-disk) mismatch. Returns an existing Path
    when possible, else the original (so the loader raises a clear error)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(cfg.data_root) / p
    if p.exists():
        return p
    if p.name.endswith(".nii.gz"):
        alt = p.with_name(p.name[:-3])          # .nii.gz -> .nii
    elif p.name.endswith(".nii"):
        alt = p.with_name(p.name + ".gz")        # .nii -> .nii.gz
    else:
        alt = p
    return alt if alt.exists() else p


def _base_transform(cfg=CFG):
    """MONAI deterministic preprocessing pipeline (dict transform on key 'image').
    load(RAS) -> 1mm spacing -> percentile-clip -> z-score(nonzero) -> resize R^3."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRangePercentilesd, NormalizeIntensityd, Resized, EnsureTyped,
    )
    R = int(cfg.resolution)
    lo, hi = cfg.clip_pct
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=cfg.spacing_mm, mode="bilinear"),
        # clip to robust percentiles (kills outliers; rescales clipped range to [0,1])
        ScaleIntensityRangePercentilesd(keys="image", lower=lo, upper=hi,
                                        b_min=0.0, b_max=1.0, clip=True),
        # z-score over nonzero (brain/foreground) voxels -> contrast-robust-ish
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # fixed isotropic grid so variable input shapes become comparable tensors
        Resized(keys="image", spatial_size=(R, R, R), mode="trilinear", align_corners=False),
        EnsureTyped(keys="image", dtype=torch.float32),
    ])
    # NOTE: explicit skull-strip (SynthStrip) is a future upgrade (cfg.skull_strip);
    # NormalizeIntensity(nonzero) already de-emphasizes background for a first pass.


def preprocess_volume(path: str | Path, cfg=CFG) -> torch.Tensor:
    """path(.nii/.nii.gz) -> tensor [1, R, R, R] (R=cfg.resolution)."""
    out = _base_transform(cfg)({"image": str(resolve_path(path, cfg))})
    return torch.as_tensor(out["image"]).float()


def build_cache(manifest: list[dict], cfg=CFG):
    """manifest: [{'id': str, 'path': str|Path}]. Returns a MONAI PersistentDataset
    that caches each preprocessed tensor once and reuses it across runs.
    Each item -> {'image': tensor[1,R,R,R], 'id': str}."""
    from monai.data import PersistentDataset
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"image": str(resolve_path(m["path"], cfg)), "id": m["id"]} for m in manifest]
    return PersistentDataset(data=rows, transform=_base_transform(cfg), cache_dir=str(cache_dir))


if __name__ == "__main__":
    import argparse
    import csv
    ap = argparse.ArgumentParser(description="Preprocess sanity-check / dry-run.")
    ap.add_argument("--data-root", default=str(CFG.data_root))
    ap.add_argument("--manifest", help="CSV with image-path columns (e.g. train_pairs.csv)")
    ap.add_argument("--resolution", type=int, default=CFG.resolution)
    ap.add_argument("--n", type=int, default=2, help="number of volumes to process")
    args = ap.parse_args()

    CFG.data_root = Path(args.data_root)
    CFG.resolution = args.resolution

    paths: list[str] = []
    if args.manifest:
        with open(args.manifest, newline="") as f:
            for row in csv.DictReader(f):
                for col in ("query_image", "target_image", "image"):
                    if col in row and row[col]:
                        paths.append(row[col])
                if len(paths) >= args.n:
                    break
    if not paths:
        raise SystemExit("No image paths found — pass --manifest pointing at a manifest CSV.")

    print(f"Preprocessing {min(args.n, len(paths))} volume(s) at {CFG.resolution}^3 ...")
    for p in paths[:args.n]:
        v = preprocess_volume(p)
        print(f"  {Path(p).name:40s} -> shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={v.min():.2f} max={v.max():.2f} mean={v.mean():.3f} std={v.std():.3f}")
    print("OK — preprocess produced fixed-shape, z-scored tensors.")
