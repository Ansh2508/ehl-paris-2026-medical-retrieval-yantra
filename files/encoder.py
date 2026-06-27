"""Branch A: shared 3D contrastive encoder -> projection -> L2-normed embedding.

Both ceT1 and T2 map into ONE space so they're comparable by cosine. Default backbone is
a MONAI 3D ResNet (fast, reliable on ROCm) — our challenger to Wilfred's SwinUNETR (machine 2).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from config import CFG

_RESNETS = {"resnet10", "resnet18", "resnet34", "resnet50", "resnet3d"}


class Encoder(nn.Module):
    def __init__(self, cfg=CFG, backbone: str | None = None):
        super().__init__()
        self.cfg = cfg
        bb = (backbone or cfg.backbone)
        d = int(cfg.embedding_dim)

        import monai.networks.nets as nets
        if bb not in _RESNETS:
            # swin / other -> fall back to a reliable resnet18 (swin is Wilfred's track, machine 2)
            print(f"[encoder] backbone '{bb}' not built here; using resnet18.")
            bb = "resnet18"
        ctor = {"resnet10": nets.resnet10, "resnet18": nets.resnet18, "resnet34": nets.resnet34,
                "resnet50": nets.resnet50, "resnet3d": nets.resnet18}[bb]
        # InstanceNorm (not BatchNorm): batch-size-independent, better for small-batch 3D medical,
        # and avoids a BatchNorm kernel-compile failure on gfx942/ROCm. ResNet's fc = our head.
        kw = dict(spatial_dims=3, n_input_channels=1, num_classes=d)
        try:
            # affine=True so the norm layers have weight/bias (ResNet init sets them; InstanceNorm
            # defaults to affine=False -> weight is None -> init crashes).
            self.net = ctor(norm=("instance", {"affine": True}), **kw)
        except TypeError:
            self.net = ctor(**kw)
        self.kind = "resnet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B,1,R,R,R]
        return F.normalize(self.net(x), dim=1)             # [B, d], unit-norm

    @torch.no_grad()
    def encode(self, volume: torch.Tensor) -> torch.Tensor:
        """Single volume [1,R,R,R] -> L2-normed embedding [d]."""
        self.eval()
        dev = next(self.parameters()).device
        v = volume.as_tensor() if hasattr(volume, "as_tensor") else torch.as_tensor(volume)
        return self.forward(v.unsqueeze(0).to(dev).float().contiguous())[0]
