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

        if bb in _RESNETS:
            import monai.networks.nets as nets
            ctor = {"resnet10": nets.resnet10, "resnet18": nets.resnet18, "resnet34": nets.resnet34,
                    "resnet50": nets.resnet50, "resnet3d": nets.resnet18}[bb]
            # ResNet ends in fc(feat -> num_classes); we use it as the projection head -> embedding.
            self.net = ctor(spatial_dims=3, n_input_channels=1, num_classes=d)
            self.kind = "resnet"
        else:
            # swin / other -> fall back to a reliable resnet18 with a note (swin is Wilfred's track)
            print(f"[encoder] backbone '{bb}' not built here; using resnet18 (swin = machine 2).")
            import monai.networks.nets as nets
            self.net = nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=d)
            self.kind = "resnet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B,1,R,R,R]
        return F.normalize(self.net(x), dim=1)             # [B, d], unit-norm

    @torch.no_grad()
    def encode(self, volume: torch.Tensor) -> torch.Tensor:
        """Single volume [1,R,R,R] -> L2-normed embedding [d]."""
        self.eval()
        dev = next(self.parameters()).device
        return self.forward(volume.unsqueeze(0).to(dev).float())[0]
