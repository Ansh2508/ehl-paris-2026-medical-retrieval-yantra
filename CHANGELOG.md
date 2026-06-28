# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/).

Developed with the assistance of GLM-5.2 (Z.AI), an open-source large language model.

## [v36] — 2026-06-28 — Honest/Deployable, Optimized (MRR 0.723)

### Added
- MI after registration for d2/d3 (weight=0.20). Once volumes are aligned
  to the d1 reference via the MOMENTS cascade, they share a common grid,
  making MI a legitimate intensity-correspondence measure.

### Changed (tuned vs v34/v35)
- Reduced d3 random rigid amplitude (+-5deg, +-1 voxel vs v35's +-15deg, +-3 voxels)
- Less aggressive d3 trim (keep 75% vs v35's 50%)
- Retuned d2/d3 weights: SSC=0.35, MI=0.20, features share 0.45

### Score History
| Version | Score | Key Change |
|---------|-------|------------|
| sample | 0.131 | Floor baseline |
| v2 | 0.583 | 16^3 volume + mask + proj |
| v3 | 0.592 | 32^3 + gradient + no-crop d1 |
| v9 | 0.647 | + MI (d1) + power spectrum (FFT) |
| v21 | 0.724 | + Sinkhorn (tau=20) + registration d2 |
| v26 | 0.839 | + SSC-12 + distance normalization |
| v29 | 0.881 | + higher SSC weight (0.50 for d2/d3) |
| v31 | 0.923 | + d1 query as reg ref + shrink [8,4,2,1] |
| v33 | 0.928 | + Hungarian + d3 registration (LEAKY) |
| v34 | 0.686 | Honest: random rigid + trim + cascade + greedy + SSC-only |
| v35 | 0.707 | + restored features + tuned d3 params |
| **v36** | **0.723** | **+ MI post-reg d2/d3 + reduced rigid + less trim** |

## [v35] — 2026-06-28 — Honest (MRR 0.707)

## [v34] — 2026-06-28 — Honest/Deployable (MRR 0.686)

### Integrity Fixes
- All 6 audit items applied: random rigid, trimmed SSC, MOMENTS cascade,
  trilinear resize, SSC-only fusion, greedy ranking
- Score drop reflects removal of d3 leak + Hungarian bijection exploitation

## [v33] — 2026-06-28 — MRR 0.928 (LEAKY — superseded by v35)

### Added
- Hungarian optimal assignment (replaces Sinkhorn, exact 1-to-1 bijection)
- Registration for d3 (in addition to d2)
- d1 train query as registration reference (Gowshigan's approach)
- Registration shrink [8,4,2,1] for better multi-resolution convergence
- Architecture documentation (ARCHITECTURE.md, architecture.puml)
- PlantUML diagram of the full pipeline
- Research paper (paper_draft.pdf) with FDA/CE clinical certification argument
- Presentation slides (slides_beamer.pdf) with MRI slice images
- GLM-5.2 (Z.AI) acknowledgment in all deliverables

### Changed
- SSC-12 weight increased to 0.50 for d2/d3 (from 0.30)
- Distance normalization: each feature normalized to [0,1] before weighting
- Paper updated with 0.928 score, Hungarian algorithm, Optuna methodology

### Score History
| Version | Score | Key Change |
|---------|-------|------------|
| sample | 0.131 | Floor baseline |
| v2 | 0.583 | 16^3 volume + mask + proj |
| v3 | 0.592 | 32^3 + gradient + no-crop d1 |
| v9 | 0.647 | + MI (d1) + power spectrum (FFT) |
| v21 | 0.724 | + Sinkhorn (tau=20) + registration d2 |
| v26 | 0.839 | + SSC-12 + distance normalization |
| v29 | 0.881 | + higher SSC weight (0.50 for d2/d3) |
| v31 | 0.923 | + d1 query as reg ref + shrink [8,4,2,1] |
| **v33** | **0.928** | **+ Hungarian + d3 registration (LEAKY — not deployable)** |

## [v31] — 2026-06-28 — MRR 0.923

### Added
- d1 train query as registration reference
- Registration shrink factors [8,4,2,1] with smoothing [4,2,1,0]
- Optuna Bayesian optimization for weight tuning

## [v26] — 2026-06-28 — MRR 0.839

### Added
- SSC-12 descriptor (12-edge neighbour-to-neighbour, GPU)
- Per-feature distance normalization to [0,1]
- Spectral flux 3D feature (Samsung-inspired onset detection)

## [v21] — 2026-06-28 — MRR 0.724

### Added
- Sinkhorn optimal transport reranking (tau=20)
- Affine registration for d2 (SimpleITK Mattes MI)

## [v9] — 2026-06-28 — MRR 0.647

### Added
- Mutual Information (d1 only, 16-bin joint histogram)
- 3D power spectrum via FFT (16 radial bins, translation-invariant)
- MIND descriptor (6^3 grid sampling)

## [v3] — 2026-06-27 — MRR 0.592

### Added
- 32^3 resolution (from 16^3)
- Gradient magnitude (modality-invariant edges)
- No-crop mode for d1 (preserves absolute position)

## [v2] — 2026-06-27 — MRR 0.583

### Added
- Initial feature extraction pipeline
- Foreground mask + robust percentile scaling
- Volume + mask + projections + metadata features
- Grid search weight optimization
