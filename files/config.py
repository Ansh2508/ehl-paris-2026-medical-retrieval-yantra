"""Single source of truth for all knobs. Import `CFG` everywhere."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- paths (data lives on the MI300X) ---
    data_root: Path = Path("data")
    cache_dir: Path = Path(".cache")
    out_dir: Path = Path("outputs")

    # --- preprocessing ---
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    resolution: int = 96          # 96 to start, try 128. NEVER 256 (see CLAUDE.md DON'T).
    clip_pct: tuple[float, float] = (0.5, 99.5)
    skull_strip: bool = True      # use SynthStrip if the simple mask is unreliable on T2/L3

    # --- encoder / training (Branch A) ---
    backbone: str = "swinunetr_encoder"   # or "resnet3d", "brainmvp"
    embedding_dim: int = 256
    temperature: float = 0.07     # InfoNCE
    batch_size: int = 64          # the real use of 192GB: more negatives. Raise as memory allows.
    epochs: int = 200
    lr: float = 1e-4
    amp: bool = True
    use_torch_compile: bool = False   # OPTIONAL/risky on ROCm — keep off until proven

    # --- augmentation strengths (calibrated to the real d2/d3 example pairs) ---
    # Applied INDEPENDENTLY to each scan in a pair.
    l2_aug: dict = field(default_factory=lambda: {
        "rotate_deg": 25.0,        # independent rigid rotation (~15-25° seen in d2)
        "translate_vox": 10,
        "elastic_sigma": (5.0, 8.0),
        "elastic_magnitude": (50.0, 150.0),
        "prob": 0.9,
    })
    l3_aug: dict = field(default_factory=lambda: {
        "bias_field_coeff": 0.5,   # scanner/hospital shift
        "gamma": (0.7, 1.5),
        "dropout_holes": 6,        # surgical tissue loss
        "dropout_size": 20,
        "small_rotate_deg": 8.0,   # KEEP roughly aligned (no big rotation)
        "prob": 0.9,
    })

    # --- fusion (STARTING points — must be tuned per level on oracle.py) ---
    fusion_w: dict = field(default_factory=lambda: {"l1": 0.8, "l2": 0.95, "l3": 0.4})

    # --- inference ---
    tta_n: int = 8
    rerank_topk: int = 10         # retrieve-then-rerank shortlist size

    # --- oracle (the keystone) ---
    holdout_pairs: int = 70       # of the 350 d1 pairs, held out for offline MRR

    # --- misc ---
    seed: int = 1234
    device: str = "cuda"          # ROCm exposes the AMD GPU as 'cuda'
    dry_run: bool = False


CFG = Config()
