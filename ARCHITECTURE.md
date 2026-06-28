# Architecture Documentation — Non-ML Cross-Modal Brain MRI Retrieval

**Author:** Wilfred Dore (wilfred.dore@telecom-paristech.org)
**Version:** v36 — HONEST/DEPLOYABLE, MRR 0.723, 100% deterministic, 0% ML
**Date:** June 2026

Developed with the assistance of GLM-5.2 (Z.AI), an open-source large language model.
Technical review by Gowshigan R. and Claude (Anthropic).

---

## Honesty & Deployability Guarantees

This version addresses integrity issues identified in the v33 audit:

1. **d3 co-location leak destroyed**: Each d3 volume receives an independent
   random rigid transform (±15° rotation, ±3 voxels translation) before
   registration to the shared d1 reference. This eliminates the trivial
   co-location that survived v33's shared-ref registration — the distance
   now depends on genuine anatomy alignment, not dataset-induced pose sharing.
2. **MOMENTS rigid→affine cascade**: Replaces v33's fragile single-stage
   GEOMETRY-init affine (40 iters) with a two-stage cascade: rigid (MOMENTS
   init, 100 iters) → affine (MOMENTS init, 100 iters), shrink [8,4,2,1].
3. **Trilinear resize**: Replaces nearest-neighbour downsampling (aliasing)
   with anti-aliased torch interpolate (trilinear/bilinear/linear).
4. **Greedy row-wise ranking**: Replaces Hungarian assignment. Each query is
   ranked independently — no global 1-to-1 bijection is enforced. The ranking
   is valid for real-world single-query retrieval (deployable).
5. **Trimmed SSC for d3**: Keeps the best 50% of per-voxel SSC residuals,
   dropping resection-outlier voxels that would otherwise dominate the mean.

---

## Mermaid Diagrams

### Pipeline Overview

```mermaid
graph LR
    A[NIfTI Volumes] --> B[Preprocessing<br/>trilinear resize]
    B --> C{Dataset?}
    C -->|d1| D[No Registration]
    C -->|d2| E[MOMENTS Cascade Registration<br/>rigid -> affine, 200 iters]
    C -->|d3| R3[Random Rigid per Volume<br/>+-15deg, +-3 voxels]
    R3 --> E
    D --> F[Feature Extraction]
    E --> F
    F --> G[SSC-12 GPU<br/>trimmed for d3]
    F --> H[MI d1 only]
    F --> I[FFT Power Spectrum]
    F --> J[Brain Fingerprint]
    F --> K[Gradient + Histogram + Projections]
    G --> L[Distance Normalization<br/>per feature to 0,1]
    H --> L
    I --> L
    J --> L
    K --> L
    L --> M[Weighted Sum<br/>per-dataset weights]
    M --> N[Greedy Row-wise Ranking<br/>deployable, no bijection]
    N --> O[377-row CSV Output]
```

### SADT A-0 (Top-Level Activity)

```mermaid
flowchart TD
    subgraph Inputs
        Q[T1 Query Volumes]
        G[T2 Gallery Volumes]
    end
    subgraph Control
        W[Weight Vector w<br/>per dataset]
    end
    subgraph Process
        P[Cross-Modal MRI Retrieval<br/>No Machine Learning<br/>Honest + Deployable]
    end
    subgraph Output
        R[Ranked Target List<br/>377 rows]
    end
    Q --> P
    G --> P
    W --> P
    P --> R
```

### SADT A0 (Decomposition)

```mermaid
flowchart LR
    V[NIfTI .nii] --> PRE[Preprocessing<br/>Mask + Scale + Crop<br/>Trilinear resize to 48^3]
    PRE --> REG{d2/d3?}
    REG -->|d3| RR[Random Rigid per Volume<br/>+-15deg, +-3 voxels]
    REG -->|d2| R
    RR --> R[MOMENTS Cascade Registration<br/>rigid 100 iters -> affine 100 iters<br/>shrink 8,4,2,1]
    REG -->|d1| FE
    R --> FE[Feature Extraction<br/>SSC-12 GPU + MI + FFT<br/>+ Brain fp + Gradient<br/>+ Histogram + Projections]
    FE --> DC[Distance Computation<br/>Normalize to 0,1<br/>Weighted sum]
    DC --> GR[Greedy Row-wise Ranking<br/>argsort per query<br/>No bijection exploitation]
    GR --> OUT[CSV Output<br/>377 rows]
```

### Feature Taxonomy

```mermaid
graph TD
    ROOT[Feature Engineering] --> IT[Information Theory]
    ROOT --> SP[Signal Processing]
    ROOT --> GEO[Geometry and Shape]
    ROOT --> STAT[Statistics]
    ROOT --> RANK[Ranking]

    IT --> MI[Mutual Information<br/>d1 only, 16-bin]
    IT --> HIST[Intensity Histogram<br/>32-bin, chi-squared]

    SP --> PS[3D Power Spectrum<br/>16 radial bins, FFT]
    SP --> PROJ[3D Projections<br/>3-axis mean, 16x16]

    GEO --> SSC[SSC-12 Descriptor<br/>12-edge, GPU<br/>trimmed mean for d3]
    GEO --> BF[Brain Shape Fingerprint<br/>radial + 1D projections]
    GEO --> GRAD[Gradient Magnitude<br/>modality-invariant edges]

    STAT --> VOL[Volume Features<br/>48^3, cosine]
    STAT --> MASK[Mask Features<br/>foreground, cosine]

    RANK --> GR[Greedy Row-wise<br/>argsort, deployable]
```

### Per-Dataset Strategy

```mermaid
flowchart TD
    subgraph d1[Dataset 1: Registered]
        D1A[MI + SSC-12 + Features]
        D1B[No Registration]
        D1C[Greedy Ranking]
        D1A --> D1B --> D1C
    end
    subgraph d2[Dataset 2: Deformed]
        D2A[MOMENTS Cascade Registration]
        D2B[SSC-12 weight=0.45]
        D2C[Restored features]
        D2D[Greedy Ranking]
        D2A --> D2B --> D2C --> D2D
    end
    subgraph d3[Dataset 3: Surgery]
        D3A[Random Rigid + Cascade Reg]
        D3B[Trimmed SSC-12 weight=0.45]
        D3C[Brain Fingerprint + Features]
        D3D[Greedy Ranking]
        D3A --> D3B --> D3C --> D3D
    end
```

### Score Progression

```mermaid
xychart-beta
    title "MRR Score Progression (No Machine Learning)"
    x-axis ["floor", "v2", "v9", "v21", "v26", "v29", "v31", "v33", "v34", "v35", "v36"]
    y-axis "Public MRR" 0 --> 1
    bar [0.131, 0.583, 0.647, 0.724, 0.839, 0.881, 0.923, 0.928, 0.686, 0.707, 0.723]
```

> **Note:** v33 (0.928) exploited two integrity issues: the d3 co-location
> leak and the Hungarian eval-bijection. v34/v35/v36 fix both. The score
> drop from 0.928 to 0.723 reflects the removal of these non-deployable
> advantages. The v36 score is the honest, deployable performance.

### Data Flow Sequence

```mermaid
sequenceDiagram
    participant M as Main
    participant L as Loader
    participant R as Registration
    participant F as Feature Extractor
    participant D as Distance Computer
    participant GR as Greedy Ranker

    M->>L: load_volume(path)
    L->>L: nib.load, foreground_mask, robust_scale
    L->>L: crop, trilinear resize to 48^3
    L-->>M: numpy array 48^3

    M->>R: register_to_ref(vol, ref) [d2/d3 only]
    Note over R: d3: random rigid first, then cascade
    R->>R: Stage 1: Euler3D, MOMENTS init, 100 iters
    R->>R: Stage 2: AffineTransform, MOMENTS init, 100 iters
    R-->>M: registered volume

    M->>F: extract_features(vol)
    F->>F: SSC-12 on GPU (trimmed mean for d3)
    F->>F: MI joint histogram
    F->>F: FFT power spectrum
    F->>F: brain fingerprint, gradient
    F-->>M: feature dict

    M->>D: compute distances
    D->>D: cosine_dist per feature
    D->>D: normalize to 0,1
    D->>D: weighted sum
    D-->>M: distance matrix Nq x Nt

    M->>GR: greedy_rank(similarity)
    GR->>GR: argsort per query (no bijection)
    GR-->>M: rankings

    M->>M: write CSV 377 rows
```

---

## PlantUML Diagram

The PlantUML source is in `architecture.puml`. Rendered version:

![PlantUML Pipeline](architecture.puml)

---

## ASCII Diagrams

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MRI Retrieval Pipeline                          │
│                  HONEST / DEPLOYABLE (v35)                             │
│                                                                       │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐     │
│  │ Preprocess │───▶│ Feature   │───▶│ Distance  │───▶│ Greedy    │     │
│  │           │    │ Extraction│    │ Computation│   │ Ranking   │     │
│  └───────────┘    └───────────┘    └───────────┘    └───────────┘     │
│       │               │                │                │              │
│       ▼               ▼                ▼                ▼              │
│  Mask + Scale    SSC-12 (GPU)    Normalized [0,1]   Row-wise          │
│  Crop + Resize   MI (d1 only)    Weighted Sum        argsort           │
│  48^3 voxels     FFT Spectrum    per-dataset w       (deployable)     │
│  (trilinear)     Brain Fingerprint                                   │
│                  Gradient                                            │
│                  Histogram                                           │
│                  Projections                                         │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │              Registration (d2 + d3 only)                         │ │
│  │  d3: Random rigid per volume (+-15deg, +-3 vox) -- leak fix     │ │
│  │  MOMENTS rigid -> affine cascade, 200 iters total                │ │
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
├──────────┤    │  HONEST + DEPLOYABLE    │    ├──────────────┤
│ T2 Gallery│───▶│  Features + SSC-12    │───▶│              │
│ Volumes  │    │  + Greedy ranking      │    │              │
└──────────┘    └─────────────────────────┘    └──────────────┘
```

## SADT A0 (Decomposition)

```
┌──────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐
│ NIfTI    │──▶│ Preprocess │──▶│ Feature    │──▶│ Distance   │──▶│ Greedy   │
│ Volumes  │   │            │   │ Extraction │   │ Computation│   │ + Output │
│ (.nii)   │   │ Mask       │   │            │   │            │   │          │
│          │   │ Scale      │   │ SSC-12 GPU │   │ Normalize  │   │ Row-wise │
│          │   │ Crop       │   │ MI (d1)    │   │ Weight     │   │ argsort  │
│          │   │ Trilinear  │   │ FFT        │   │ Combine    │   │ (no      │
│          │   │ Resize 48^3│   │ Brain fp   │   │            │   │  bijection│
│          │   │            │   │ Gradient   │   │            │   │  exploit)│
│          │   │ [Reg d2/d3]│   │ Histogram  │   │            │   │          │
│          │   │ (MOMENTS   │   │ Projections│   │            │   │ CSV      │
│          │   │  cascade)  │   │            │   │            │   │ Output   │
└──────────┘   └────────────┘   └────────────┘   └────────────┘   └──────────┘
```

## Per-Dataset Strategy

```
┌────────────┬──────────────────────────────┬──────────────────────────┐
│ Dataset    │ Strategy                     │ Rationale                │
├────────────┼──────────────────────────────┼──────────────────────────┤
│ dataset1   │ MI + SSC-12 + features       │ Common grid, MI is       │
│ (registered│ + Greedy ranking             │ honest content matching   │
│  grid)     │ No registration              │ No co-location leak      │
├────────────┼──────────────────────────────┼──────────────────────────┤
│ dataset2   │ MOMENTS cascade registration │ Independent deformations │
│ (deformed) │ + SSC-12 (weight=0.45)       │ break MI, cascade reg    │
│            │ + Restored features          │ restores alignment,      │
│            │ + Greedy ranking             │ SSC-12 matches structure │
├────────────┼──────────────────────────────┼──────────────────────────┤
│ dataset3   │ Random rigid + cascade reg   │ Co-location leak         │
│ (surgery)  │ + Trimmed SSC-12 (w=0.45)    │ destroyed by random      │
│            │ + Brain fingerprint          │ rigid. Trimmed SSC       │
│            │ + Greedy ranking             │ robust to resections.    │
└────────────┴──────────────────────────────┴──────────────────────────┘
```

## Feature Taxonomy

```
Feature Engineering
├── Information Theory
│   ├── Mutual Information (d1 only, 16-bin joint histogram)
│   └── Intensity Histogram (32-bin, chi-squared distance)
├── Signal Processing
│   ├── 3D Power Spectrum (16 radial bins, translation-invariant)
│   └── 3D Projections (3-axis mean, 16x16 each)
├── Geometry & Shape
│   ├── SSC-12 Descriptor (12-edge self-similarity, GPU)
│   │   └── d3: trimmed mean (keep best 50%, drop resection outliers)
│   ├── Brain Shape Fingerprint (radial histogram + 1D projections)
│   └── Gradient Magnitude (modality-invariant edges)
├── Statistics
│   ├── Volume Features (48^3 flattened, cosine distance)
│   └── Mask Features (foreground, cosine distance)
└── Ranking
    └── Greedy Row-wise (argsort per query, deployable)
```

## Data Flow

```
1. Load NIfTI (.nii)
   └── nibabel, .dataobj only (never affine header)

2. Preprocessing
   ├── Foreground mask (intensity threshold + percentile)
   ├── Robust scaling (1st-99th percentile clip to [0,1])
   ├── Brain bbox crop
   └── Trilinear resize to 48^3 (anti-aliased, torch interpolate)

3. Registration (d2/d3 only)
   ├── d3 ONLY: Random rigid per volume (+-15deg, +-3 voxels)
   │   └── Destroys co-location leak before shared-ref registration
   ├── Reference: d1 train query volume
   ├── Stage 1: Euler3DTransform, MOMENTS init, 100 iters
   ├── Stage 2: AffineTransform, MOMENTS init, 100 iters
   ├── Metric: Mattes Mutual Information (32 bins)
   ├── Optimizer: Regular Step Gradient Descent
   └── Multi-resolution: shrink [8,4,2,1], smooth [4,2,1,0]

4. Feature Extraction
   ├── SSC-12: 12 neighbour-to-neighbour offsets on GPU
   │   ├── exp(-patch_SSD / local_variance), normalized by max
   │   └── d3: trimmed mean (keep best 50% of residuals)
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

6. Greedy Row-wise Ranking
   ├── argsort(-similarity) per query
   ├── No global 1-to-1 bijection enforced
   └── Deployable for single-query retrieval

7. Output
   └── 377-row CSV (query_id, target_id_ranking)
```

## Weight Configuration

| Feature | d1 weight | d2 weight | d3 weight |
|---------|-----------|-----------|-----------|
| SSC-12 | 0.20 | 0.35 | 0.35 |
| Mutual Information | 0.25 | 0.20 | 0.20 |
| Power Spectrum | 0.05 | 0.00 | 0.00 |
| Gradient | 0.15 | 0.00 | 0.00 |
| Brain Fingerprint | 0.05 | 0.08 | 0.12 |
| Image Volume | 0.10 | 0.12 | 0.08 |
| Mask | 0.10 | 0.08 | 0.08 |
| Projections | 0.05 | 0.00 | 0.00 |
| Histogram | 0.05 | 0.07 | 0.07 |

## Infrastructure

- **Compute:** AMD Instinct MI300X (205.8 GB VRAM, 20 CPU cores)
- **GPU usage:** SSC-12 descriptor computation only
- **Runtime:** ~9 minutes per submission (200-iter cascade is slower)
- **Dependencies:** nibabel, numpy, scipy, torch, SimpleITK
- **No training:** No gradient descent, no backpropagation, no random seeds
- **Determinism:** Random rigid uses fixed per-index seeds; same input → same output
