# CLAUDE.md — Paris EHL 2026 · MRI Cross-Modal Retrieval

> Context for Claude Code. Read this fully before writing any code.
> Hard-won decisions are baked in here. Some "obvious" choices are deliberately
> ruled out (see DO / DON'T). When in doubt, the **offline oracle number decides** — never gut feeling.

---

## 1. Goal (one line)
Given a **ceT1 (T1 post-contrast) query** brain MRI volume, rank a **gallery of T2 volumes** so the **same patient's** T2 ranks as high as possible. Win a Kaggle leaderboard **and** present a research-worthy method to Inria/MPI judges.

## 2. Task & metric (precise — this dictates everything)
- For each query, output a **full ranking of every gallery target** in that query's dataset+split.
- Score = **Mean Reciprocal Rank (MRR)**: per query, `RR = 1 / rank_of_true_target`. Missing/omitted query → RR 0.
- Final = `(MRR_dataset1 + MRR_dataset2 + MRR_dataset3) / 3` (equal weight).
- **Implication: rank-1 is everything.** rank1=1.0, rank2=0.5, rank3=0.33. A method that nails rank-1 half the time beats one that's always ~rank-5.
- **Implication: the 3 levels weigh equally.** Most teams ace L1 and tank L3. The competition is won on **L2/L3**.
- Random baseline MRR ≈ 0.05–0.06. Kaggle limit: **100 submissions/team/day**. Public LB = validation rows; private LB (final) = test rows.

## 3. Data (challenge repo: NicoStellwag/ehl-paris-2026-medical-retrieval)
- Format: 3D NIfTI `.nii.gz`, RAS orientation, **1.0mm isotropic**. **No** intensity norm, **no** skull-strip, **no** crop applied. **Shapes vary** (esp. d2/d3); query and target may differ in shape — never assume a fixed shape.
- Counts:
  - `dataset1`: **350 labelled train pairs**, val 40/40, test 100/100. **Registered to a common grid.** ← the only labels we get.
  - `dataset2`: no train pairs. val 40/40, test 100/100.
  - `dataset3`: no train pairs. val 20/20, test 77/77.
- Submission: one combined CSV, **377 rows** (all val+test, all datasets). Columns: `query_id,target_id_ranking` (space-separated full ranking). Partial submissions allowed (omitted datasets score 0; displayed score ×3 = MRR of the single submitted dataset).

## 4. The three levels — their TRUE nature (from inspecting the example pairs)
- **L1 (dataset1)** — ceT1 and T2 **perfectly co-registered** on a common grid. Voxel-level cross-modal similarity works directly.
- **L2 (dataset2)** — strong **independent rigid (rotation ~15–25° + translation) + non-linear elastic** deformation applied to query and target **separately**. They no longer share geometry. → **voxel correspondence is destroyed → MIND/MI BREAK here.** Lean on the learned embedding.
- **L3 (dataset3)** — preop→**intraoperative**. Target resampled into the preop physical space → **roughly co-located** (NOT heavily rotated like L2), but **not strictly registered**: tissue shift, **missing tissue from surgery**, and **different-scanner/contrast (hospital) shift**. → MIND/MI **still carry signal** (rough spatial overlap survives) → **MIND is a real L3 anchor.** The hard part is the appearance/scanner shift + local structural change, not gross misalignment.

> So per-level fusion direction: **MIND strong on L1 AND L3, ~zero on L2.** (Tune the exact weights on the simulated oracle — see §6 — never guess them.)

## 5. Architecture — 5 components
**Preprocess** (`preprocess.py`)
- Resample to **96³ (start) or 128³** isotropic. **NOT 256³** (see DON'T). Percentile clip 0.5–99.5 → z-score. Brain mask (use **SynthStrip** if the mask is unreliable on T2/L3). Cache preprocessed tensors (MONAI `PersistentDataset`).

**A — 3D contrastive encoder** (`encoder.py`, `train.py`) — primary engine
- **Shared** 3D backbone (one encoder, modality token or shared weights) + projection head. Pretrained backbone preferred: SwinUNETR **encoder** (extract + pool features — it's a *segmentation* net), or BrainMVP, or Dey et al. randomized-synthesis encoder.
- **InfoNCE** on patient identity, **large batch for many negatives** (this is the real use of 192GB, not resolution).
- **Heavy domain-randomization augmentation** (§7) — this is what generalizes to L2/L3 with zero target labels.

**B — MIND structural anchor** (`mind.py`) — training-free, modality-invariant. Real anchor for L1 and L3.

**C — TTA** (`infer.py`) — embed each query under ~8 augmentations, average embeddings, then rank.

**D — Late fusion** (`fuse.py`) — `score = w·A + (1−w)·B`, **w tuned per level on the simulated oracle** (§6). Use **rank fusion** (reciprocal-rank or normalized-rank) rather than raw-score mixing when scales differ.

**E — Referee** (`veto.py`) — prefer a **robust shape/size-consistency filter** (brain-volume ratio, FOV) + an **agreement re-rank** (boost candidates both branches rank high). A full **registration-residual veto is optional and risky on L3** (surgery breaks correspondence) → only keep it if it **raises MRR on the simulated L3 set**.
- Optional polish: **retrieve-then-rerank** — A produces a top-10 shortlist, reorder those 10 with MIND/registration to push the true match to rank-1 (cheap, helps MRR). Optional fusion meta-ranker: TabPFN v2 (Apache-2.0 weights) over the per-pair signals — debranchable, only if ≥2–3 complementary signals exist.

## 6. ⭐ THE KEYSTONE — simulated L2/L3 oracle (`oracle.py`) — BUILD THIS FIRST
The only labels we have are 350 L1 pairs. So fusion weights and the veto **cannot** be tuned/validated on real L2/L3. Fix:
1. **Hold out** a slice of the 350 train pairs (e.g. 70 pairs) as an offline retrieval set.
2. **Synthesize fake-L2**: apply independent affine (rot 15–25°, translation) + `Rand3DElastic` to query and target separately.
3. **Synthesize fake-L3**: apply `RandBiasField` + gamma/contrast remap + `RandCoarseDropout` (region masking), **keep rough alignment** (no big rotation).
4. Compute **offline MRR per simulated level** (L1 / fakeL2 / fakeL3).
5. **Tune w per level** and **gate the veto** on these numbers.
> No-double-dipping: train the encoder on one subset; tune w / thresholds on a *different* held-out subset. The public LB (val = 40/40/20 queries) is **tiny and noisy** — overfitting it = the "examen blanc" trap. **Trust local CV; use the LB as a sparse sanity-check.**

## 7. Augmentation spec (split by what it teaches — calibrated to the real example pairs)
Apply **independently to each scan in a pair** (model never sees a clean aligned pair).
- **For L2**: `RandAffine` (rotation ~15–25°, translation) + `Rand3DElastic` (visible cortical warp). Strong.
- **For L3**: `RandBiasField` + gamma/contrast remap (scanner/hospital) + `RandCoarseDropout` (surgical tissue loss). Keep alignment.
- Calibrate magnitudes from the actual d2/d3 example pairs — don't augment blind.
- Pedigree to cite: this is the **SynthSeg/SynthMorph domain-randomization** paradigm applied to retrieval (see §10).

## 8. THE ONE RULE
Before **every** Kaggle submit: run `oracle.py`. **Oracle MRR up → keep & submit. Down → revert, don't submit.** The number decides — not instinct, not a teammate's hunch.

## 9. MI300X / ROCm — setup facts (verified against AMD ROCm blogs, Oct 2025)
- **Use the AMD ROCm PyTorch Docker image** (e.g. `rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0`) — not a blind `pip install ... rocm6.1`. Avoids version mismatch pain.
- **The real speed win is MIOpen auto-tuning** (~3–5×), set env vars `MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=3`. **AMP/autocast** also helps. **`torch.compile` is OPTIONAL and can fail on ROCm+SwinUNETR** — get correctness first, add compile only if it compiles cleanly. Do NOT build the timeline on torch.compile.
- **Resolution 96–128³**, not 256³ — AMD's own SwinUNETR walkthrough uses ROI **96³**. Use the memory for **batch size (negatives)**, parallel runs, and TTA.
- Smoke-test FIRST: `python -c "import torch;print(torch.cuda.get_device_name(0))"` (should print AMD Instinct MI300X), load one NIfTI via MONAI, run one fwd/bwd of a small 3D conv on GPU. **Before** building anything.
- **SECRETS**: never commit `kaggle.json` / the real API key; never paste it into shared docs. (Simplest: upload the CSV to Kaggle manually, no key.)

## 10. Citations — CORRECTED (the plan had errors — do not repeat them)
- ❌→✅ **DSIR / DNS — arXiv:2402.18933** = *Modality-Agnostic Structural Image Representation Learning for **Deformable Registration*** by **Mok et al., Alibaba DAMO Academy**. It is **NOT Inria** and **NOT a retrieval paper**. Cite as: a learned, contrast-invariant structural descriptor (successor to MIND) that we **repurpose for retrieval**. Saying "Inria's paper" to an Inria judge = credibility loss.
- ➕ **ADD the domain-randomization pedigree** (this impresses *this* jury): **SynthSeg** (Billot et al., MedIA 2023), **SynthMorph** (Hoffmann et al., IEEE TMI 2022 — contrast-invariant matching without inter-modality similarity = our exact problem), review *Synthetic data in generalizable neuroimaging* (Imaging Neuroscience 2024), **SynthStrip** (skull-strip), Dey et al. ICLR 2025 (randomized-synthesis 3D representations — backbone option). Our novelty vs SynthMorph: contrast **+ geometry + resection** randomization for **retrieval** generalizing to unseen L2/L3 with **zero target labels**.
- ✅fix **CoMIR** = **Pielawski et al., NeurIPS 2020** (for registration); cross-modal **retrieval** extension = **Breznik et al. (2022, later journal version)**. Not "Sci Rep 2024 / CoMIR".
- ✅ **MIND** = Heinrich et al., 2012. **BrainMVP** = arXiv 2410.10604. On-task & recent: *Cross-Dataset Linkage of Brain MRI using Image Similarity Measures* (arXiv:2602.10043) — read it.
- ✅ **AMD blogs are real & correctly dated** (7 Oct 2025): *Announcing MONAI 1.0.0 for AMD ROCm* and *Medical Imaging on MI300X: Optimized SwinUNETR* — but the SwinUNETR walkthrough is **lung CT tumor (NSCLC-Radiomics), NOT brain MRI**. Cite accurately.

## 11. DO / DON'T (guardrails for Claude Code)
**DO**
- Build `oracle.py` (simulated L2/L3) before tuning anything.
- Keep one file per feature, docstrings, `config.py` for all knobs, a `--dry-run` mode, a `GLOSSARY.md` and `Design.md`.
- Get a valid (even weak) Kaggle CSV early to establish a floor.
- Tune fusion w **per level on the oracle**; gate the veto on oracle numbers.

**DON'T**
- ❌ Don't go to 256³. (Overfits 350 pairs, breaks pretraining, blows timing.)
- ❌ Don't cite DSIR as Inria/retrieval. Don't cite CoMIR as Sci Rep 2024.
- ❌ Don't ship a registration veto on L3 unless the simulated-L3 oracle says it helps.
- ❌ Don't optimize against the public LB (40/40/20 queries) — overfitting trap.
- ❌ Don't commit secrets. Don't assume a fixed image shape.
- ❌ Don't depend on torch.compile for the schedule.

## 12. Module layout (proposed)
```
config.py        # all hyperparameters, paths, resolution, aug strengths
preprocess.py    # load → resample 96/128³ → clip+zscore → mask → cache
mind.py          # MIND descriptor + cross-modal distance (Branch B)
encoder.py       # shared 3D backbone + projection head
augment.py       # MONAI domain-randomization transforms (L2-set + L3-set)
train.py         # InfoNCE training loop, AMP, big batch
oracle.py        # ⭐ holdout + synth fakeL2/fakeL3 + per-level MRR  (BUILD FIRST after preprocess)
infer.py         # embed gallery+query, TTA, rank
fuse.py          # per-level rank fusion A⊕B
veto.py          # shape/size consistency + agreement rerank (registration veto optional)
submit.py        # write 377-row Kaggle CSV
Design.md        # architecture rationale  ·  GLOSSARY.md  ·  CLAUDE.md (this file)
```

## 13. Build order (the step plan)
0. **Context** (this file) + scaffolding (`config.py`, empty modules, `Design.md`). ← we are here
1. **Smoke-test** ROCm/GPU + MONAI NIfTI load on the MI300X.
2. **`preprocess.py`** + cache. **`mind.py`** → MIND-only submission = the **floor** on the board.
3. **⭐ `oracle.py`** (simulated L2/L3) — the keystone. Wire MIND through it → get real per-level numbers.
4. **`encoder.py` + `train.py`** (InfoNCE + domain-randomization aug, big batch). Run 2–3 aug-strength configs in parallel, pick best by oracle.
5. **`fuse.py`** (per-level w from oracle) + **`infer.py` TTA**. Submit best.
6. **`veto.py`** + retrieve-then-rerank, oracle-gated. Optional TabPFN fusion. Submit best.
7. **Presentation** in parallel: lead with the **method/idea** (domain randomization → cross-modal retrieval generalizing to L2/L3 with zero target labels, MIND-anchored), MI300X/MONAI-ROCm as the *enabler* slide, not the headline.
