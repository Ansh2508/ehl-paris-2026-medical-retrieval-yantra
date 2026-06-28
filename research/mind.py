"""Branch B: MIND structural descriptor (Heinrich et al. 2012; 6-neighbourhood variant).

Training-free, modality-invariant local self-similarity. A real anchor for L1 & L3;
~useless on L2 (independent rigid+elastic destroys voxel correspondence) — fusion
down-weights B there per the oracle.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Sequence

# 6-connected neighbourhood offsets (the search region for self-similarity).
_OFFSETS6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean filter over a (2r+1)^3 window. x: [N,1,D,H,W] -> same shape (patch aggregation)."""
    k = 2 * radius + 1
    kernel = torch.ones((1, 1, k, k, k), device=x.device, dtype=x.dtype) / float(k ** 3)
    return F.conv3d(x, kernel, padding=radius)


@torch.no_grad()
def mind_descriptor(volume: torch.Tensor, radius: int = 1, eps: float = 1e-5) -> torch.Tensor:
    """volume [1,D,H,W] -> MIND field [6,D,H,W].

    For each voxel, the patch-SSD to each of its 6 neighbours, passed through
    exp(-d/var). Because it encodes *relative* local structure (not absolute
    intensity), it is invariant to the ceT1<->T2 contrast change by design."""
    if volume.dim() != 4:
        raise ValueError(f"expected [1,D,H,W], got {tuple(volume.shape)}")
    v = volume.unsqueeze(0).float()                      # [1,1,D,H,W]
    dp = []
    for (dx, dy, dz) in _OFFSETS6:
        shifted = torch.roll(v, shifts=(dx, dy, dz), dims=(2, 3, 4))
        dp.append(_box_filter((v - shifted) ** 2, radius))   # patch SSD to this neighbour
    dp = torch.cat(dp, dim=1)                            # [1,6,D,H,W]
    var = dp.mean(dim=1, keepdim=True).clamp_min(eps)    # local variance estimate
    mind = torch.exp(-dp / var)                          # [1,6,D,H,W]
    mind = mind / mind.amax(dim=1, keepdim=True).clamp_min(eps)   # Heinrich: normalize by max
    return mind[0]                                       # [6,D,H,W]


@torch.no_grad()
def mind_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Mean absolute difference between two MIND fields. Lower = more similar."""
    d = (a - b).abs().mean(dim=0)                        # [D,H,W]
    if mask is not None:
        m = mask > 0
        return float(d[m].mean()) if bool(m.any()) else float(d.mean())
    return float(d.mean())


@torch.no_grad()
def mind_score_matrix(query_vols: Sequence[torch.Tensor], gallery_vols: Sequence[torch.Tensor],
                      radius: int = 1) -> torch.Tensor:
    """Distance matrix [Q,G] (lower = more similar). Precomputes gallery descriptors once."""
    gds = [mind_descriptor(g, radius) for g in gallery_vols]
    out = torch.zeros(len(query_vols), len(gallery_vols))
    for i, qv in enumerate(query_vols):
        qd = mind_descriptor(qv, radius)
        for j, gd in enumerate(gds):
            out[i, j] = mind_distance(qd, gd)
    return out


@torch.no_grad()
def mind_rank_fn(query_vol: torch.Tensor, gallery_vols: Sequence[torch.Tensor],
                 radius: int = 1) -> list[int]:
    """RankFn (oracle/infer contract): gallery *indices* best->worst by MIND distance."""
    qd = mind_descriptor(query_vol, radius)
    dists = [mind_distance(qd, mind_descriptor(g, radius)) for g in gallery_vols]
    return sorted(range(len(dists)), key=lambda i: dists[i])


@torch.no_grad()
def mind_rank(query_vol: torch.Tensor, gallery: Sequence[tuple[str, torch.Tensor]],
              radius: int = 1) -> list[str]:
    """Design contract: rank gallery (id, vol) by MIND similarity, best->worst."""
    ids = [g[0] for g in gallery]
    order = mind_rank_fn(query_vol, [g[1] for g in gallery], radius)
    return [ids[i] for i in order]
