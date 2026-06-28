# What Works — the validated-technique ledger
*Counterpart to `NEGATIVE_RESULTS.md`. Everything here earned its place on the frozen pipeline with a **measured** gain. Everything we tried and dropped (with reasons) is in `NEGATIVE_RESULTS.md`.*

## The frozen pipeline, in one sentence
Register each brain to a neutral reference by **image content** (Mutual Information), describe structure with **SSC-12** (modality-invariant), **trim** the mismatched/resected voxels, and **assign** — training-free, deterministic, auditable.

---

## Techniques that work

### 1. SSC-12 descriptor ⭐ — the backbone
12-edge neighbour↔neighbour self-similarity context (Heinrich). Modality-invariant: encodes *local structure*, not intensity, so a ceT1 and its T2 look alike.
- **Works:** the core feature at every level; beats MIND-6, NGF, NMI, raw intensity, radiomics.
- **Evidence:** d1 ≈ 1.0 on the shared grid; carries d2/d3 once aligned.

### 2. Content-registration — rigid→affine **cascade** (MI) ⭐ — the d2 lever
Mattes Mutual Information, MOMENTS init, rigid → affine, shrink [8,4,2,1]. Modality-blind.
- **Works:** d2 is unsolvable without it.
- **Evidence:** none 0.08 → rigid 0.78 → **cascade 0.87**. Beats single-stage affine (degenerates) and deformable (over-normalizes).

### 3. Per-voxel **trim** / "chunk matching" ⭐ — the resection lever · *[your idea]*
Per-voxel SSC distance, keep the best (1−t), drop the worst t. Per-level: d2 ≈ 0.5, d3 ≈ 0.75.
- **Works:** deletes the resection cavity + mismatches from the score.
- **Evidence:** d3 **0.25 → 0.44**.

### 4. **Fixed-reference** registration (a real d1 query, NOT a gallery-mean) ⭐ — robust honest d3
- **Works:** recovers a broken / un-aligned d3 where a gallery-mean template collapses (the mean of rotated brains is a blob).
- **Evidence:** break-test — fixed-ref + break = **~0.96** vs gallery-template + break = **0.751**.

### 5. Active co-location destruction (±20° rigid break) → re-register — honest d3
- **Works:** genuinely earns d3 from anatomy, leak-free.
- **Evidence:** leak ladder 1.0 (leak) → 0.26 (broken, no reg) → **~0.86** (broken + reg); break-test 0.938 → 0.751.

### 6. Bijection assignment — **Hungarian / Sinkhorn** · *[your idea]*
Global one-to-one on the similarity matrix instead of per-query argmax.
- **Works:** **+0.205** (greedy 0.725 → bipartite 0.93–1.0).
- **Note:** organizer ruling — *allowed but "besides the point"* → we **report both** scores.

### 7. Bayesian calibration / abstention — deploy-time trust
- **Works:** the **one** positive ML add-on; flags low-confidence d3 (confidence↔correctness).

### 8. The offline **oracle** (fakeL2/L3 from d1 holdout) — the methodology
- **Works:** gated every decision; caught false-positives the public LB would have hidden (e.g. the bbox-crop oracle "win" the real LB refuted).

---

## Per-level "what works where"
| Level | Recipe | Result |
|---|---|---|
| **d1** (co-registered) | SSC directly (no reg) | ~1.0 |
| **d2** (rigid + elastic) | cascade-reg → SSC → trim 0.5 | strong |
| **d3** (resection) | active-break → **fixed-ref** reg → SSC → trim 0.75 | ~0.96 (Hungarian), honest |
| **assignment** | greedy (deployable) **/** Hungarian (bijection) | report both |
| **deploy** | Bayesian calibration / abstention | trust layer |

## Validated results (real Kaggle LB)
| Setting | MRR |
|---|---|
| Greedy / deployable (single-query) | **0.725** |
| Honest Hungarian (3 seeds) | **~0.96** (0.938–1.0) |
| 4-seed stable (`full_honest_v2`) | 0.930 |
| Leak (found, declined) | 1.000 |

**Final selections to lock (Kaggle UI):** `FINAL most-proficient honest` (1.0) **+** `full_honest_v2` (0.930) — banks the upside, hedges the seed-luck.
