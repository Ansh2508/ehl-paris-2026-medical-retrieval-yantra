# Experiment Log & Results — EHL Paris Medical Image Retrieval

**Team Yantra | Wilfred Doré | June 27-28, 2026**

## Leaderboard Progression

| Version | Public Score | Key Innovation | Train MRR (d1) | Total Time |
|---------|-------------|----------------|----------------|------------|
| sample | 0.13127 | Floor baseline | — | — |
| v2 | 0.58335 | 16³ volume + mask + proj + meta, tuned weights | 0.763 | ~17 min (Kaggle) |
| v3 | 0.59235 | 32³ + gradient magnitude + no-crop for d1 | 0.848 | ~27 min (Kaggle) |
| CLIP | 0.42703 | 3D CNN dual-encoder + InfoNCE, 50 epochs | — | ~7 min (AMD GPU) |
| v5 | 0.46713 | Depth features + pHash (DCT, Hamming) | 0.845 | ~10 min (AMD 20-core) |
| v6 | 0.57662 | MIND + brain fingerprint, MIND-heavy weights | 0.848 | ~3 min (AMD 20-core) |
| v7 | 0.59303 | MIND + brain fingerprint, image-heavy weights for d2/d3 | 0.848 | ~3 min (AMD 20-core) |
| **v8** | **0.63976** | **48³ + Mutual Information (d1 only) + MIND + brain fingerprint** | **0.903** | ~5 min (AMD 20-core) |

## Key Findings

### 1. Mutual Information is the Gold Standard for d1
- MI between T1 and T2 volumes on the registered grid (dataset1) gives MRR 0.903 on train
- MI weight = 0.35 dominates the scoring for d1
- MI captures statistical dependence without assuming linear intensity relationship — exactly what's needed for the T1/T2 modality gap
- **MI is only useful for d1 (registered data). For d2/d3 (deformed/missing tissue), MI weight = 0.**

### 2. Resolution Matters
- Going from 16³ → 32³ → 48³ progressively improves discrimination
- 48³ gives ~110K voxels per volume — enough spatial resolution for fine structural differences
- Higher resolution (64³) would be better but MI computation becomes prohibitively slow

### 3. MIND Descriptor (Modality-Independent Neighbourhood Descriptor)
- Local self-similarity descriptor, contrast-invariant by construction
- Implemented as 6-direction offset squared differences, exp(-10*normalized), sampled on 6³ grid
- Useful as auxiliary feature for d2/d3 but does not dominate
- Based on: Heinrich et al., "MIND: Modality Independent Neighbourhood Descriptor"

### 4. Brain Shape Fingerprint
- 1D projections of the foreground mask (axial, coronal, sagittal) + radial distance histogram
- Robust to local changes (missing tissue in d3) because it captures global shape
- Weight 0.15-0.35 for d3, confirming its value for the missing-tissue scenario

### 5. Gradient Magnitude
- Modality-invariant edge features (edges exist in both T1 and T2, even if intensities invert)
- Weight 0.20-0.30 across all datasets — consistently useful

### 6. No-Crop vs Crop
- **No-crop wins for d1**: preserving absolute spatial position helps because d1 is registered to a common grid
- **Crop wins for d2/d3**: centering on the brain removes irrelevant background variation

### 7. Deep Learning (CLIP-style) Underperformed
- 3D CNN dual-encoder with InfoNCE loss, 50 epochs, 64³ resolution
- Score 0.427 — worse than non-DL approaches
- Reasons: only 350 training pairs, 3D convolutions have too many params, augmentation was too aggressive
- Lesson: with 350 pairs, non-DL feature engineering beats DL. DL needs either pretraining or 2D slices (as Anshu's 0.97 approach shows)

### 8. pHash (Perceptual Hashing)
- DCT-based perceptual hash on 3 orthogonal mid-slices, Hamming distance
- Did not improve score when given high weight for d2/d3
- pHash captures visual appearance but loses too much spatial information for this task

### 9. Tuning on d1 Does Not Predict d2/d3 Performance
- Weights optimized on d1 train pairs (registered) do not generalize to d2 (deformed) or d3 (missing tissue)
- Best strategy: tune d1 weights on d1, use heuristic image-heavy weights for d2/d3
- The v3 heuristic weights (image=0.20, mask=0.30, grad=0.15, proj=0.15, meta=0.10) outperformed "optimized" depth/pHash weights for d2/d3

## Infrastructure Notes

### Kaggle Kernels
- Data mounted at `/kaggle/input/competitions/ehl-paris-medical-image-retrieval/` (NOT `/kaggle/input/ehl-...`)
- Images are `.nii` (uncompressed), not `.nii.gz` as referenced in CSVs — need path fallback
- 4 CPU cores, no GPU by default, ~1h runtime for feature extraction on 1174 volumes
- 100 submissions/day limit

### AMD MI300X Server
- 205.8 GB VRAM, 20 CPU cores, 235 GB RAM
- PyTorch 2.10 + ROCm pre-installed
- SSH unstable (kernel panics with 3D ops), but JupyterLab (port 80) reliable
- 20-core parallelism reduces feature extraction from ~500s to ~30s
- kaggle CLI installable → direct submission from server
- Data download: `kaggle competitions download` → 9.75 GB → Python `zipfile` extract (no `unzip` command)

## Feature Architecture (v8)

```
9 distance matrices:
1. image_r   — 48³ volume, cosine distance
2. mask_r    — 48³ foreground mask, cosine distance  
3. grad_r    — 48³ gradient magnitude, cosine distance
4. proj      — 3× 16×16 mean projections, cosine distance
5. meta      — affine + bbox + geometric median, standardized L2
6. hist      — 32-bin intensity histogram, chi² distance
7. mind      — 6³×6 MIND descriptor, standardized L2
8. brain     — brain shape fingerprint (radial + 1D projs), cosine
9. mi        — mutual information (d1 only!), 1/(1+MI) distance
```

## Per-Dataset Weights (v8)

| Dataset | image | mask | grad | proj | meta | hist | mind | brain | mi |
|---------|-------|------|------|------|------|------|------|-------|----|
| d1 | 0.10 | 0.20 | 0.20 | 0.05 | 0.05 | 0.05 | 0.00 | 0.00 | 0.35 |
| d2 | 0.20 | 0.25 | 0.15 | 0.15 | 0.05 | 0.05 | 0.05 | 0.05 | 0.00 |
| d3 | 0.15 | 0.25 | 0.10 | 0.15 | 0.10 | 0.05 | 0.05 | 0.15 | 0.00 |
