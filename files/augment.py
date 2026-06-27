"""Domain-randomization transforms (CLAUDE.md §7), applied INDEPENDENTLY to each scan.

Two roles:
  - l2_transforms / l3_transforms : used by oracle.py to SYNTHESIZE fakeL2 / fakeL3.
  - train_transforms              : domain randomization the contrastive ENCODER trains on.

Design (informed by SynthSeg domain-randomization & Sequence-Invariant Contrastive Learning):
registration handles L2's big rotation, so train geometry is LIGHT and the budget goes to
the ceT1<->T2 MODALITY GAP (heavy contrast randomization) + L3 resection simulation.
train_transforms is deliberately DIFFERENT from l2/l3_transforms so the oracle stays honest
(no train-on-exactly-what-you-measure leakage).
"""
from __future__ import annotations
from config import CFG

# NOTE: MONAI is imported lazily inside functions so this file imports on CPU (no MONAI).


def l2_transforms(cfg=CFG):
    """Independent rigid (rot ~15-25°, translation) + non-linear elastic — for oracle fakeL2."""
    from monai.transforms import Compose, RandAffined, Rand3DElasticd
    a = cfg.l2_aug
    rot = a["rotate_deg"] * 3.14159 / 180.0
    return Compose([
        RandAffined(keys="image", prob=a["prob"], rotate_range=(rot, rot, rot),
                    translate_range=(a["translate_vox"],) * 3, padding_mode="zeros"),
        Rand3DElasticd(keys="image", prob=a["prob"],
                       sigma_range=a["elastic_sigma"], magnitude_range=a["elastic_magnitude"]),
    ])


def l3_transforms(cfg=CFG):
    """Bias field + gamma + coarse dropout + small rotation — for oracle fakeL3 (keep alignment)."""
    from monai.transforms import (Compose, RandBiasFieldd, RandAdjustContrastd,
                                   RandCoarseDropoutd, RandAffined)
    a = cfg.l3_aug
    rot = a["small_rotate_deg"] * 3.14159 / 180.0
    return Compose([
        RandBiasFieldd(keys="image", prob=a["prob"], coeff_range=(0.0, a["bias_field_coeff"])),
        RandAdjustContrastd(keys="image", prob=a["prob"], gamma=a["gamma"]),
        RandCoarseDropoutd(keys="image", prob=a["prob"], holes=a["dropout_holes"],
                           spatial_size=(a["dropout_size"],) * 3, fill_value=0.0),
        RandAffined(keys="image", prob=a["prob"], rotate_range=(rot, rot, rot)),
    ])


def _make_resection(prob=0.4, max_frac=0.16):
    """A MONAI transform that carves a random connected ELLIPSOID cavity (set to 0) to
    simulate a surgical resection — more realistic L3 than scattered random holes."""
    import torch
    from monai.transforms import MapTransform, RandomizableTransform

    class RandSimulatedResectiond(RandomizableTransform, MapTransform):
        def __init__(self):
            MapTransform.__init__(self, keys="image")
            RandomizableTransform.__init__(self, prob)

        def __call__(self, data):
            d = dict(data)
            self.randomize(None)
            if not self._do_transform:
                return d
            for key in self.key_iterator(d):
                img = torch.as_tensor(d[key]).float()          # [1,D,H,W]
                _, D, H, W = img.shape
                cz, cy, cx = (self.R.uniform(0.3, 0.7) * s for s in (D, H, W))
                rz, ry, rx = (self.R.uniform(0.06, max_frac) * s for s in (D, H, W))
                zz = torch.arange(D).view(D, 1, 1)
                yy = torch.arange(H).view(1, H, 1)
                xx = torch.arange(W).view(1, 1, W)
                ell = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
                img[0][ell] = 0.0
                d[key] = img
            return d

    return RandSimulatedResectiond()


def train_transforms(cfg=CFG):
    """Domain randomization for the contrastive encoder. Light geometry (registration owns
    L2 rotation) + heavy MODALITY-GAP contrast randomization + simulated resection (L3).
    Apply INDEPENDENTLY to query and target."""
    from monai.transforms import (Compose, RandAffined, RandFlipd, ScaleIntensityd, RandBiasFieldd,
                                   RandAdjustContrastd, RandHistogramShiftd, RandScaleIntensityd,
                                   RandShiftIntensityd, RandGaussianSmoothd, RandGaussianNoised,
                                   NormalizeIntensityd)
    a3 = cfg.l3_aug
    light_rot = 10.0 * 3.14159 / 180.0
    return Compose([
        # --- light geometry (registration handles the big L2 rotation) ---
        RandAffined(keys="image", prob=0.5, rotate_range=(light_rot,) * 3,
                    translate_range=(5,) * 3, scale_range=(0.1,) * 3, padding_mode="zeros"),
        RandFlipd(keys="image", prob=0.3, spatial_axis=0),
        # rescale to [0,1] so the contrast augs (bias/gamma) operate on a POSITIVE domain
        # (RandBiasField multiplies -> it blows up on signed z-scored data)
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        # --- MODALITY GAP: SynthSeg-style contrast randomization (the #1 lever) ---
        RandBiasFieldd(keys="image", prob=0.8, coeff_range=(0.0, a3["bias_field_coeff"])),
        RandHistogramShiftd(keys="image", prob=0.7, num_control_points=(8, 15)),  # random nonlinear remap
        RandAdjustContrastd(keys="image", prob=0.7, gamma=a3["gamma"]),
        RandScaleIntensityd(keys="image", prob=0.5, factors=0.3),
        RandShiftIntensityd(keys="image", prob=0.5, offsets=0.1),
        RandGaussianSmoothd(keys="image", prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandGaussianNoised(keys="image", prob=0.3, std=0.03),
        # --- L3 surgical tissue loss (carve cavity -> 0 = background in [0,1]) ---
        _make_resection(prob=0.4, max_frac=0.16),
        # re-standardize for the encoder (bounds the range whatever the contrast augs did)
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
