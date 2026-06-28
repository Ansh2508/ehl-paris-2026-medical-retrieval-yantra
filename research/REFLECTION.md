# REFLECTION.md — Team glossary & how we reasoned to the plan

> For teammates joining mid-stream. Part 1 is a plain-English **glossary**; Part 2 is our
> **reflexion process** — *why* the plan is what it is, with the evidence behind each call.
> Companion to `CLAUDE.md` (decisions/guardrails), `Design.md` (contracts), `INTERFACES.md`
> (who-owns-what), `GLOSSARY.md` (original terms). When in doubt, **the oracle number decides.**

---

## Part 1 — Glossary

- **The task** — given a **ceT1** (T1 post-contrast) query brain MRI, rank a gallery of **T2** volumes so the *same patient's* T2 is rank-1. Cross-modal retrieval / re-identification.
- **MRR (Mean Reciprocal Rank)** — score = mean of `1/rank_of_true_target`. rank1=1.0, rank2=0.5. **Rank-1 is everything.** Final = mean over the 3 datasets (equal weight).
- **The 3 levels** — **L1** (dataset1): ceT1/T2 perfectly co-registered. **L2** (dataset2): each scan *independently* rotated (~15–25°) + elastically warped → voxel correspondence destroyed. **L3** (dataset3): preop→intraop, roughly co-located but tissue removed/shifted + scanner shift. **The competition is won on L2/L3** (everyone aces L1).
- **Modality gap** — ceT1 and T2 look totally different for the same brain; the *anatomy* matches, the *pixels* don't.
- **MIND** (Heinrich 2012) — training-free local self-similarity descriptor; contrast-invariant by construction. Our **Branch B** anchor. Strong on L1/L3, dies on L2.
- **InfoNCE / contrastive learning** — train a shared encoder so same-patient ceT1 & T2 embed close, others far. More in-batch negatives = stronger signal (the real use of the MI300X's 205 GB).
- **Registration** — estimating a transform that aligns one volume to another/to a template. **EasyReg / SynthMorph** (FreeSurfer) do it contrast-agnostically with **zero training**. Our lever for L2.
- **Oracle** (`oracle.py`) — our offline practice exam: hold out 70 of the 350 L1 pairs, **synthesize fake-L2 and fake-L3** from them, measure MRR per level. The only way to tune L2/L3 without real labels. **THE ONE RULE: run it before every Kaggle submit.**
- **Bijection / Hungarian assignment** — each split is N queries ↔ N targets, one-to-one. Solving it as a *global* assignment (Hungarian/Sinkhorn) beats greedy per-query argmax. **Our unique edge** — no competitor doc exploits it.
- **Domain randomization** — train on heavily distorted synthetic copies (warped, contrast-shifted, masked) so the model learns what *stays the same* about a brain. SynthSeg/SynthMorph pedigree.
- **TTA** — embed ~8 augmented views of a query, average, then rank. Cheap free gain.
- **Late fusion** (`fuse.py`) — combine Branch A (embedding) + Branch B (MIND) rankings; mix weight `w` tuned **per level on the oracle**.
- **Veto** (`veto.py`) — deterministic referee that demotes impossible matches (shape/FOV). Kept only if it raises oracle MRR.
- **2.5D vs 3D** — 3D = whole-volume conv encoder (Wilfred's primary). 2.5D = encode many slices → pool (our challenger; cheaper, can borrow 2D pretrained backbones).
- **Entire / W&B / MIOpen** — Entire = the hackathon's session-tracker (judges the codebase). W&B = our experiment dashboard. MIOpen = AMD's kernel library; `MIOPEN_FIND_MODE=1` avoids slow first-run tuning.

---

## Part 2 — Our reflexion process (why the plan is the plan)

**0. Framing.** Metric is mean MRR over 3 levels, equally weighted; the organizers said out loud the win is on **L2/L3**. Floor to beat = **0.42 combined** (the provided 2D slice baseline). Random ≈ 0.05.

**1. We grounded the approach in the literature** (see `research-findings`). Key findings:
- Our task is essentially a *published* one — cross-dataset brain-MRI linkage (Sharma 2026) hits ~perfect with skull-strip + **register-to-template** + simple similarity (NMI/GradSim) — *but* same-modality. So a classical pipeline is a real per-level contender, and **NMI/GradSim** are modality-robust baselines worth keeping.
- **Registration can defeat L2.** L2 is independent *rigid+elastic*. Register every volume into one canonical frame and the rotation is undone → MIND/embeddings work again. Tooling (EasyReg/SynthMorph) is zero-training, out-of-box.
- **The gallery is a bijection** → Hungarian/Sinkhorn assignment = free MRR. Ours alone.
- Better backbones than SwinUNETR-seg exist (Decipher-MR, BrainIAC, BrainMVP) — use if weights/licence OK.

**2. The strategic reframe.** The original plan said "learned embedding is the engine, MIND is a side anchor." The research sharpened it to:
> **Registration *removes* the variance it can (geometry → L2); augmentation *teaches* invariance to the rest (modality gap + L3 tissue change); MIND anchors L1/L3; the contrastive encoder is the engine; fuse per-level; exploit the bijection. The oracle gates every step.**

**3. The keystone first.** We can't tune L2/L3 on real labels (we have none), so we built `oracle.py` to synthesize fake-L2/L3 from the 350 L1 pairs and measure offline. Everything is decided by that number, not instinct.

**4. We tested it empirically.** First real run (MIND, 96³, 70 holdout):

| Level | MIND MRR | Meaning |
|---|---|---|
| L1 | **0.993** | MIND *solves* the registered case |
| L2 | **0.108** | ≈ random — MIND dies under independent deformation (as predicted) |
| L3 | **0.367** | a real anchor through scanner shift + tissue loss |

This *confirmed the thesis* and pinpointed L2 (0.11) as the battleground. **Next experiment (running now):** add rigid registration to a canonical frame and re-measure L2 — if it jumps, finding ② is proven.

**5. Honest calls we made (and why).**
- **Drop "geometric data depth" pre-alignment.** It's tabular anomaly detection (Mozharovskyi), exponential in dimension, intractable on 10⁵–10⁷-voxel images with n=350, and it produces *no spatial transform*. "Center-of-mass" only fixes translation, not L2's rotation. **Registration supersedes it.** (Its only legit niche: an embedding-space outlier score for the veto.)
- **pHash** is weak cross-modal (encodes appearance, which differs across modalities) — at best a fast L1 filter.
- **No 256³, no torch.compile on the schedule** — overfits 350 pairs / unreliable on ROCm. Use 96–128³ and spend the 192 GB on batch size + TTA + parallel runs.
- **Don't optimize the public LB** (40/40/20, noisy) — trust the oracle.

**6. Division of labor.** Wilfred → 3D SwinUNETR + InfoNCE (+ his own ideas) on **machine 2**. Us → shared **infra** (preprocess/oracle/mind/fuse/submit), the **2.5D challenger**, and **augmentation** (modality gap + L3), on **box 3**. Both encoders expose the same `Encoder.encode` so the oracle picks the winner per level → an ensemble, nothing wasted.

**7. For the pitch.** Judges want **methodology over flash**, value **creativity**, and a likely on-task judge works on **neurosurgical AI** → lead with: *domain-randomized, registration-anchored cross-modal retrieval that generalizes to surgical L3 with zero target labels, validated on a simulated oracle.* That story **is** this document.

---

## Data-integrity audit (and why we don't exploit the L3 leak)

We audited the NIfTI headers for non-content shortcuts (the SETI/PetFinder class of leak). Finding:

- **dataset3 (L3) has a *perfect* header leak**: each query's NIfTI **affine / qoffset uniquely identifies its true target** (20/20 distinct, 20/20 exact-1 match). Matching L3 by affine alone → **MRR ≈ 1.0, no content, no training**. Cause: the challenge resamples each L3 target *into its query's physical space*, so the target inherits the query's affine. It's **intrinsic** (would hold on the private split too).
- **dataset1 (L1)**: query & target share the affine, but it's a *common grid* shared by the whole gallery → not discriminative (no exploit).
- **dataset2 (L2)**: one affine across the whole gallery → no leak.

**Decision: we deliberately do NOT exploit it.** Our pipeline resamples to 96³ (discarding the original affine), so it is content-honest by construction. Rationale: (1) the challenge is judged on **methodology** by Inria/MPI researchers who explicitly warned against shortcuts; (2) **`entire`** runs cheating-detection on the codebase; (3) a metadata match doesn't *solve* the cross-modal retrieval problem; (4) identifying-and-avoiding the shortcut is a stronger, shake-up-proof research story than gaming it. We **report** the leak as a data-integrity finding (audit: `leak_audit.py`).

## Where we stand
- Infra built + validated; data on box 3; **MIND oracle baseline `L1=0.99 / L2=0.11 / L3=0.37`** (W&B `mind-oracle-v1`).
- Running: rigid-registration A/B on L2. Then: contrastive encoder, per-level fusion, bijection assignment, submission.
- Branch `dev/gowshigan`; coordination in `INTERFACES.md`; the rule in `CLAUDE.md §8`.
