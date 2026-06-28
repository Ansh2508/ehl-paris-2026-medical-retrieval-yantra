# INTERFACES.md — shared contracts & ownership boundaries

> **Purpose:** keep 5 people working in parallel from stepping on each other.
> Read this before you start a module. Companion to **Design.md** (data flow) and **CLAUDE.md** (decisions).

---

## TL;DR — how we avoid collisions
- **SHARE (one copy each, owned):** preprocessing + **alignment**, `oracle`, `metrics`, `fuse`, `submit`.
- **PARALLELIZE (separate, no conflict):** the **encoders** — a 3D primary and a 2.5D challenger — behind **one** embedding interface.
- **VALIDATE on the oracle, never on the Kaggle LB** (public LB = 40/40/20 queries, noisy; 100 submits/day cap).

---

## Ownership map

| Track | Modules | Branch | Owner |
|---|---|---|---|
| **3D engine (primary)** | `encoder.py` (SwinUNETR **encoder** + pool), `train.py` | `feat/encoder-3d` | Box-2 teammate |
| **2.5D challenger + augmentation** | slice encoder, `augment.py` | `feat/encoder-2p5d` | Gowshigan |
| **Preprocess + alignment** ⚠️ SHARED | `preprocess.py` (incl. the `align` step below) | `feat/preprocess` | **needs ONE owner** |
| **⭐ Oracle (keystone)** | `oracle.py`, `metrics.py` | `feat/oracle` | TBD |
| **MIND + fusion + submit** | `mind.py`, `fuse.py`, `infer.py`, `submit.py` | `feat/mind-fuse` | TBD |
| **Veto + presentation** | `veto.py`, slides | floats | TBD |

> The encoders are **legitimately parallel** (3D primary vs 2.5D challenger). The **only real collision** is the alignment step — see below.

---

## The contracts (do not change without a heads-up)

**1. `scorer` — the glue (Design.md).** Anything that ranks a gallery for a query:
```python
scorer(query_id: str, gallery_ids: list[str]) -> list[str]   # best -> worst
```
`oracle.py` and `submit.py` both consume a scorer, so MIND-only / 3D-embedding / 2.5D-embedding / fused are all interchangeable and measured identically.

**2. `Encoder.encode` — BOTH encoders must satisfy this.**
```python
encoder.encode(volume) -> torch.Tensor   # L2-normed embedding [d], comparable by cosine
```
3D and 2.5D encoders live in different files/branches but expose the **same** `.encode()`. Fusion and the oracle then treat them identically, and **the oracle picks the winner per level** → both efforts become an ensemble, nothing is wasted.

**3. `RankFn` — what the oracle scores.**
```python
RankFn = Callable[[query_vol, Sequence[gallery_vols]], list[int]]   # gallery indices best->worst
```

---

## ⚠️ The alignment merge (`center_of_mass` ⊕ `rigid_register`)

Two of us were about to implement geometry-normalization separately (center-of-mass vs register-to-template). **Merge into one shared step in `preprocess.py`,** behind a config flag:

```python
# config.py  (append at END — see append-only rule)
align: str = "rigid_register"   # one of: "none" | "center_of_mass" | "rigid_register"
```

| Mode | Fixes | Cost | Notes |
|---|---|---|---|
| `none` | nothing | 0 | baseline |
| `center_of_mass` | **translation only** | trivial (mask centroid) | does **NOT** fix L2's 15–25° rotation |
| `rigid_register` | **rotation + translation** | out-of-box | **default** — EasyReg / SynthMorph (FreeSurfer, contrast-agnostic, zero-training) |

**Why `rigid_register` is the default:** Dataset-2's hard part is *rotation*, which center-of-mass leaves untouched. Registration to a canonical template removes the rigid scramble (and most of the affine), so MIND / similarity / embeddings become viable on L2 again. Center-of-mass is kept as the cheap fallback. One implementation, one cache — not two.

> Requires FreeSurfer (EasyReg/SynthMorph) installed on the box. `mri_synthmorph -o moved.nii moving.nii fixed.nii`.

---

## Non-collision rules
1. **`config.py` is append-only** — add new fields at the **end**, never reorder existing ones. Config changes ship as their **own tiny PR**, merged immediately.
2. **Separate `cache_dir` / `out_dir` per experiment** — the 3D and 2.5D routes produce different tensor shapes; never share a preprocessed cache.
3. **Feature branch per component → PR → `main`.** One owner per `feat/*`. Keep PRs small.
4. **THE ONE RULE:** run `oracle.py` before every Kaggle submit. Oracle MRR up → keep & submit. Down → revert.
5. **SwinUNETR = encoder features pooled**, not the segmentation head. Resolution ≤ 128³ (never 256³ — OOM + overfit). Watch batch size on 3D.

---

## Open coordination items
- [ ] Assign owners for Preprocess, Oracle, MIND/Fusion, Veto.
- [ ] Install **FreeSurfer / EasyReg** on both boxes (box 2 also lacks MONAI + nibabel).
- [ ] Confirm the gallery is a **bijection** (N queries ↔ N targets) on the real manifests → unlocks **Hungarian / Sinkhorn** assignment for a free MRR lift.
- [ ] Decide pretrained backbone: SwinUNETR-encoder (safe) vs a 2025/26 brain foundation model (Decipher-MR / BrainIAC / BrainMVP) if weights + license check out.
