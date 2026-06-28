# Architecture Documentation — Non-ML Cross-Modal Brain MRI Retrieval

**Author:** Wilfred Dore (wilfred.dore@telecom-paristech.org)
**Version:** v33 — MRR 0.928, 100% deterministic, 0% ML
**Date:** June 2026

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MRI Retrieval Pipeline                          │
│                                                                       │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐     │
│  │ Preprocess │───▶│ Feature   │───▶│ Distance  │───▶│ Hungarian │     │
│  │           │    │ Extraction│    │ Computation│   │ Assignment│     │
│  └───────────┘    └───────────┘    └───────────┘    └───────────┘     │
│       │               │                │                │              │
│       ▼               ▼                ▼                ▼              │
│  Mask + Scale    SSC-12 (GPU)    Normalized [0,1]   Optimal 1-to-1    │
│  Crop + Resize   MI (d1 only)    Weighted Sum        Bijection        │
│  48^3 voxels     FFT Spectrum                        (scipy)          │
│                  Brain Fingerprint                                   │
│                  Gradient                                            │
│                  Histogram                                           │
│                  Projections                                         │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Registration (d2 + d3 only)                         │ │
│  │  SimpleITK Affine + Mattes MI, d1 train query as reference       │ │
│  │  Shrink [8,4,2,1], Smoothing [4,2,1,0]                           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## SADT A-0 (Top-Level)

```
                    ┌─────────────────┐
                    │  Weight Vector  │
                    │  w (per dataset) │
                    └────────┬────────┘
                             │
┌──────────┐    ┌────────────▼────────────┐    ┌──────────────┐
│ T1 Query │───▶│                         │───▶│ Ranked       │
│ Volumes  │    │  Cross-Modal Retrieval  │    │ Target List  │
│          │    │  (No Machine Learning)  │    │ (377 rows)   │
├──────────┤    │                         │    ├──────────────┤
│ T2 Gallery│───▶│  13 features + SSC-12  │───▶│              │
│ Volumes  │    │  + Hungarian assignment │    │              │
└──────────┘    └─────────────────────────┘    └──────────────┘
```

## SADT A0 (Decomposition)

```
┌──────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐
│ NIfTI    │──▶│ Preprocess │──▶│ Feature    │──▶│ Distance   │──▶│ Hungarian│
│ Volumes  │   │            │   │ Extraction │   │ Computation│   │ + Output │
│ (.nii)   │   │ Mask       │   │            │   │            │   │          │
│          │   │ Scale      │   │ SSC-12 GPU │   │ Normalize  │   │ scipy    │
│          │   │ Crop       │   │ MI (d1)    │   │ Weight     │   │ optimal  │
│          │   │ Resize 48^3│   │ FFT        │   │ Combine    │   │ assign   │
│          │   │            │   │ Brain fp   │   │            │   │          │
│          │   │ [Reg d2/d3]│   │ Gradient   │   │            │   │ CSV      │
│          │   │ (SimpleITK)│   │ Histogram  │   │            │   │ Output   │
│          │   │            │   │ Projections│   │            │   │          │
└──────────┘   └────────────┘   └────────────┘   └────────────┘   └──────────┘
```

## Per-Dataset Strategy

```
┌────────────┬──────────────────────────────┬──────────────────────────┐
│ Dataset    │ Strategy                     │ Rationale                │
├────────────┼──────────────────────────────┼──────────────────────────┤
│ dataset1   │ MI + SSC-12 + features       │ Common grid, MI is       │
│ (registered│ + Hungarian                  │ non-discriminative,      │
│  grid)     │ No registration              │ honest content matching   │
├────────────┼──────────────────────────────┼──────────────────────────┤
│ dataset2   │ Affine registration          │ Independent deformations │
│ (deformed) │ + SSC-12 (weight=0.50)       │ break MI, registration   │
│            │ + FFT (translation-invariant)│ restores alignment,      │
│            │ + Hungarian                  │ SSC-12 matches structure │
├────────────┼──────────────────────────────┼──────────────────────────┤
│ dataset3   │ Affine registration          │ Tissue resection +       │
│ (surgery)  │ + SSC-12 (weight=0.50)       │ scanner shift,           │
│            │ + Brain fingerprint          │ SSC-12 robust to local   │
│            │ + Hungarian                  │ changes, brain fp        │
│            │                              │ captures global shape    │
└────────────┴──────────────────────────────┴──────────────────────────┘
```

## Feature Taxonomy

```
Feature Engineering (13 types)
├── Information Theory
│   ├── Mutual Information (d1 only, 16-bin joint histogram)
│   └── Intensity Histogram (32-bin, chi-squared distance)
├── Signal Processing
│   ├── 3D Power Spectrum (16 radial bins, translation-invariant)
│   └── 3D Projections (3-axis mean, 16x16 each)
├── Geometry & Shape
│   ├── SSC-12 Descriptor (12-edge self-similarity, GPU)
│   ├── Brain Shape Fingerprint (radial histogram + 1D projections)
│   └── Gradient Magnitude (modality-invariant edges)
├── Statistics
│   ├── Volume Features (48^3 flattened, cosine distance)
│   └── Mask Features (foreground, cosine distance)
└── Assignment
    └── Hungarian Algorithm (scipy.optimize.linear_sum_assignment)
```

## Data Flow

```
1. Load NIfTI (.nii)
   └── nibabel, .dataobj only (never affine header)

2. Preprocessing
   ├── Foreground mask (intensity threshold + percentile)
   ├── Robust scaling (1st-99th percentile clip to [0,1])
   ├── Brain bbox crop
   └── Resize to 48^3 (nearest neighbor)

3. Registration (d2/d3 only)
   ├── Reference: d1 train query volume
   ├── SimpleITK AffineTransform(3)
   ├── Metric: Mattes Mutual Information (32 bins)
   ├── Optimizer: Regular Step Gradient Descent
   └── Multi-resolution: shrink [8,4,2,1], smooth [4,2,1,0]

4. Feature Extraction
   ├── SSC-12: 12 neighbour-to-neighbour offsets on GPU
   │   └── exp(-patch_SSD / local_variance), normalized by max
   ├── MI: 16-bin joint histogram, 1/(1+MI) distance
   ├── FFT: np.fft.fftn -> |FFT|^2 -> log1p -> radial bins
   ├── Brain fp: mask radial histogram + 3-axis projections
   ├── Gradient: np.gradient -> magnitude -> normalize
   ├── Histogram: 32-bin intensity -> chi-squared
   └── Projections: 3-axis mean -> 16x16 -> cosine

5. Distance Computation
   ├── Per-feature distance matrix
   ├── Normalize each to [0,1]
   └── Weighted sum (weights per dataset)

6. Hungarian Assignment
   ├── scipy.optimize.linear_sum_assignment(-similarity)
   ├── Assigned target -> rank 1
   └── Remaining targets sorted by score

7. Output
   └── 377-row CSV (query_id, target_id_ranking)
```

## Weight Configuration

| Feature | d1 weight | d2 weight | d3 weight |
|---------|-----------|-----------|-----------|
| SSC-12 | 0.20 | 0.50 | 0.50 |
| Mutual Information | 0.25 | 0.00 | 0.00 |
| Power Spectrum | 0.05 | 0.10 | 0.10 |
| Gradient | 0.15 | 0.05 | 0.05 |
| Brain Fingerprint | 0.05 | 0.05 | 0.10 |
| Image Volume | 0.10 | 0.10 | 0.05 |
| Mask | 0.10 | 0.10 | 0.10 |
| Projections | 0.05 | 0.05 | 0.05 |
| Histogram | 0.05 | 0.05 | 0.05 |

## Infrastructure

- **Compute:** AMD Instinct MI300X (205.8 GB VRAM, 20 CPU cores)
- **GPU usage:** SSC-12 descriptor computation only
- **Runtime:** ~7 minutes per submission
- **Dependencies:** nibabel, numpy, scipy, torch, SimpleITK
- **No training:** No gradient descent, no backpropagation, no random seeds
- **Determinism:** Same input always produces same output
