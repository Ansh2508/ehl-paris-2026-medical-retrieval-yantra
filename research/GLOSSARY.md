# GLOSSARY.md — plain-English terms

- **Modality gap** — ceT1 and T2 look completely different for the same brain (colour photo vs X-ray of the same face). The pixels barely overlap; the *anatomy* does.
- **The three levels** — L1: scans perfectly aligned. L2: each scan independently rotated + warped (voxel correspondence destroyed). L3: target taken after surgery + different scanner, roughly co-located (tissue removed/shifted, contrast differs).
- **MIND descriptor** (Heinrich 2012) — training-free matcher. Encodes how each spot relates to its neighbours; that local pattern is the same across scan types. Our Branch B. Strong on L1 & L3, useless on L2.
- **Contrastive learning / InfoNCE** — train by showing pairs: "these two = same patient, those = different". Pulls same-patient fingerprints together, pushes others apart. More negatives in a batch = better signal (why big batch matters).
- **Fingerprint / embedding** — each scan → a short vector. Same patient → close vectors. Matching = nearest vector.
- **Domain randomization** — train on heavily distorted synthetic copies (warped, contrast-shifted, regions masked) so the model learns what *stays the same* about a brain. It does NOT add patients. This is the SynthSeg/SynthMorph paradigm (Iglesias/Fischl/Dalca), here applied to retrieval.
- **TTA (test-time augmentation)** — at inference, embed ~8 augmented versions of the query, average, then rank. Small free gain.
- **Late fusion** — Branch A and Branch B score independently; combine at the end. Mix ratio `w` tuned **per level** on the oracle.
- **Oracle / offline MRR** — our practice test built from the 350 known L1 pairs, plus **synthesized fake-L2 and fake-L3** copies so we can measure all three levels offline. Checked before every Kaggle submit. MRR: rank1=1.0, rank2=0.5, rank3=0.33…
- **Deterministic veto** — a fixed-rule referee that overrides the learned model when geometry/shape says its top pick is impossible. Kept only if it raises oracle MRR.
- **Neurosymbolic** — learned model (Branch A) + fixed rulebook (the veto).
- **ROCm / MIOpen** — ROCm is AMD's CUDA equivalent. MIOpen is its kernel library; enabling its **auto-tuning** is the real ~3–5× speed win on MI300X (env vars), not torch.compile.
