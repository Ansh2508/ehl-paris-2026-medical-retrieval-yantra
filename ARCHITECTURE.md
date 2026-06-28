# Architecture Documentation — Non-ML Cross-Modal Brain MRI Retrieval

**Author:** Wilfred Dore (wilfred.dore@telecom-paristech.org)
**Version:** v33 — MRR 0.928, 100% deterministic, 0% ML
**Date:** June 2026

Developed with the assistance of GLM-5.2 (Z.AI), an open-source large language model.

---

## Mermaid Diagrams

### Pipeline Overview

```mermaid
graph LR
    A[NIfTI Volumes] --> B[Preprocessing]
    B --> C{Dataset?}
    C -->|d1| D[No Registration]
    C -->|d2/d3| E[Affine Registration<br/>SimpleITK Mattes MI]
    D --> F[Feature Extraction]
    E --> F
    F --> G[SSC-12 GPU]
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
    M --> N[Hungarian Assignment<br/>scipy.optimize]
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
        P[Cross-Modal MRI Retrieval<br/>No Machine Learning<br/>13 features + SSC-12 + Hungarian]
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
    V[NIfTI .nii] --> PRE[Preprocessing<br/>Mask + Scale + Crop<br/>Resize to 48^3]
    PRE --> REG{d2/d3?}
    REG -->|Yes| R[Affine Registration<br/>SimpleITK + Mattes MI<br/>shrink 8,4,2,1]
    REG -->|No| FE
    R --> FE[Feature Extraction<br/>SSC-12 GPU + MI + FFT<br/>+ Brain fp + Gradient<br/>+ Histogram + Projections]
    FE --> DC[Distance Computation<br/>Normalize to 0,1<br/>Weighted sum]
    DC --> HA[Hungarian Assignment<br/>scipy.optimize<br/>Exact 1-to-1 bijection]
    HA --> OUT[CSV Output<br/>377 rows]
```

### Feature Taxonomy

```mermaid
graph TD
    ROOT[Feature Engineering<br/>13 types] --> IT[Information Theory]
    ROOT --> SP[Signal Processing]
    ROOT --> GEO[Geometry and Shape]
    ROOT --> STAT[Statistics]
    ROOT --> ASG[Assignment]

    IT --> MI[Mutual Information<br/>d1 only, 16-bin]
    IT --> HIST[Intensity Histogram<br/>32-bin, chi-squared]

    SP --> PS[3D Power Spectrum<br/>16 radial bins, FFT]
    SP --> PROJ[3D Projections<br/>3-axis mean, 16x16]

    GEO --> SSC[SSC-12 Descriptor<br/>12-edge, GPU]
    GEO --> BF[Brain Shape Fingerprint<br/>radial + 1D projections]
    GEO --> GRAD[Gradient Magnitude<br/>modality-invariant edges]

    STAT --> VOL[Volume Features<br/>48^3, cosine]
    STAT --> MASK[Mask Features<br/>foreground, cosine]

    ASG --> HUN[Hungarian Algorithm<br/>scipy.optimize]
```

### Per-Dataset Strategy

```mermaid
flowchart TD
    subgraph d1[Dataset 1: Registered]
        D1A[MI + SSC-12 + Features]
        D1B[No Registration]
        D1C[Hungarian Assignment]
        D1A --> D1B --> D1C
    end
    subgraph d2[Dataset 2: Deformed]
        D2A[Affine Registration]
        D2B[SSC-12 weight=0.50]
        D2C[FFT translation-invariant]
        D2D[Hungarian Assignment]
        D2A --> D2B --> D2C --> D2D
    end
    subgraph d3[Dataset 3: Surgery]
        D3A[Affine Registration]
        D3B[SSC-12 weight=0.50]
        D3C[Brain Fingerprint]
        D3D[Hungarian Assignment]
        D3A --> D3B --> D3C --> D3D
    end
```

### Score Progression

```mermaid
xychart-beta
    title "MRR Score Progression (No Machine Learning)"
    x-axis ["floor", "v2", "v9", "v21", "v26", "v29", "v31", "v33"]
    y-axis "Public MRR" 0 --> 1
    bar [0.131, 0.583, 0.647, 0.724, 0.839, 0.881, 0.923, 0.928]
```

### Data Flow Sequence

```mermaid
sequenceDiagram
    participant M as Main
    participant L as Loader
    participant R as Registration
    participant F as Feature Extractor
    participant D as Distance Computer
    participant H as Hungarian

    M->>L: load_volume(path)
    L->>L: nib.load, foreground_mask, robust_scale
    L->>L: crop, resize to 48^3
    L-->>M: numpy array 48^3

    M->>R: register_to_ref(vol, ref) [d2/d3 only]
    R->>R: SimpleITK AffineTransform
    R->>R: Mattes MI, shrink 8,4,2,1
    R-->>M: registered volume

    M->>F: extract_features(vol)
    F->>F: SSC-12 on GPU
    F->>F: MI joint histogram
    F->>F: FFT power spectrum
    F->>F: brain fingerprint, gradient
    F-->>M: feature dict

    M->>D: compute distances
    D->>D: cosine_dist per feature
    D->>D: normalize to 0,1
    D->>D: weighted sum
    D-->>M: distance matrix Nq x Nt

    M->>H: hungarian_rank(similarity)
    H->>H: scipy.optimize.linear_sum_assignment
    H-->>M: rankings 1-to-1

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
