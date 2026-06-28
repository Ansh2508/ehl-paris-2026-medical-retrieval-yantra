# Non-ML Inspiration: Cross-Domain Ideas for Brain MRI Retrieval

**Wilfred Doré | Team Yantra | Paris Research Hackathon 2026**

This document catalogs creative ideas drawn from diverse engineering and scientific fields, with assessments of their applicability to our non-ML feature engineering approach for brain MRI cross-modal retrieval.

## Score Progression (Non-ML Only)

| Version | Score | Key Feature Added | Resolution |
|---------|-------|-------------------|------------|
| v2 | 0.583 | 16³ volume + mask + proj | 16³ |
| v3 | 0.592 | Gradient magnitude + no-crop d1 | 32³ |
| v7 | 0.593 | MIND + brain fingerprint | 32³ |
| v8 | 0.640 | **Mutual Information** (d1 only) | 48³ |
| v9 | **0.647** | **Power Spectrum (FFT)** | 48³ |
| v10 | 0.645 | NMI + wavelet + SSIM (NMI worse) | 48³ |
| v11 | 0.640 | 64³ (too high res, worse) | 64³ |

**Best: v9 = 0.64658 (no ML, pure feature engineering + information theory)**

## Cross-Domain Inspiration Catalog

### 1. LIDAR & Remote Sensing
**Origin:** Autonomous driving, topographic mapping, forest canopy analysis

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Point cloud registration (ICP)** | Treat brain voxels as 3D point cloud, align query/target via ICP before comparison | Not implemented — registration for d2/d3 |
| **Echo decomposition** | Separate "returns" at different depths = separate intensity layers in MRI | Experimental |
| **Ground/non-ground classification** | Skull (ground) vs brain tissue (non-ground) → mask-based features | Already done (foreground_mask) |
| **Canopy height model** | Maximum intensity projection = "height" of signal at each (x,y) | Already done (projection_features) |
| **Voxel cloud features** | Treat foreground voxels as point cloud → DTM, density, convex hull | Partially done (brain_shape_fingerprint) |

**Key insight from LIDAR:** ICP (Iterative Closest Point) could align d2 volumes before MI computation. This is essentially what Gowshigan's "affine-cascade registration" does.

### 2. Video Games & Computer Graphics
**Origin:** Game engines, real-time rendering

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Level-of-Detail (LOD)** | Multi-resolution features: 16³ + 32³ + 48³ concatenated | **Not yet implemented** — high potential |
| **Keyframe extraction** | Select most informative slices (max variance) instead of fixed positions | Not implemented |
| **Frustum culling** | Ignore background voxels (already done via foreground_mask) | Done |
| **Texture mipmaps** | Gaussian pyramid of the volume at multiple scales | Wavelet approximates this |
| **Normal mapping** | Surface normals of the brain boundary = shape descriptor | Not implemented |
| **Ambient occlusion** | Local "accessibility" of each voxel = how deep in tissue | Experimental |

**Key insight from games:** LOD (multi-resolution concatenation) is the most promising untested idea. Each resolution captures structures at different scales: 16³ = global shape, 32³ = lobe-level, 48³ = fine detail.

### 3. Signal Processing & Telecommunications
**Origin:** Radio, WiFi, 5G, radar

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Power spectrum (FFT magnitude)** | Translation-invariant frequency fingerprint | **Implemented (v9)** — +0.007 gain |
| **Matched filter / cross-correlation** | NCC between aligned volumes (d1) | MI covers this |
| **OFDM subcarrier allocation** | Different frequency bands = different tissue types | Experimental |
| **Channel impulse response** | Volume response to a "delta" = deconvolution of acquisition | Not applicable |
| **Beamforming** | Spatial filtering → focus on specific brain regions | Not implemented |
| **CDMA spreading codes** | Pseudo-random projections of volume for compact hashing | Related to pHash (tested, not effective) |

### 4. Thermodynamics & Statistical Physics
**Origin:** Heat transfer, entropy, phase transitions

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Local entropy** | Shannon entropy of intensity in local neighborhoods = "disorder" map | Not implemented |
| **Free energy** | MI = negative free energy in statistical mechanics | MI already used |
| **Phase transition detection** | Sudden changes in intensity distribution = pathology boundaries | Experimental |
| **Heat equation diffusion** | Apply diffusion filter to volume → smooths noise, preserves edges | Gaussian filter in MIND |

### 5. Genetics & Molecular Biology
**Origin:** DNA sequencing, population genetics

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Brain fingerprinting** | Each brain's morphology is unique (like DNA) → global shape features | **Implemented** (brain_shape_fingerprint) |
| **Allele frequency** | Intensity histogram = "allele frequency" of tissue types | **Implemented** (intensity_histogram) |
| **Phylogenetic distance** | Distance between brain "genotypes" = our feature distance | Conceptual framework |
| **Gene expression heatmap** | 3D intensity map = "expression" of tissue properties | The MRI itself |

### 6. Neuroscience & Biology
**Origin:** Brain research, animal models

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Mouse brain transfer learning** | Pretrain on mouse MRI → transfer to human | Future work (presentation material) |
| **Cortical hierarchy (V1→V2→V4→IT)** | Edge → texture → shape → identity features | Our feature stack mirrors this |
| **Hemispheric asymmetry** | Left-right brain asymmetry = individual trait | **Not implemented** — quick to add |
| **Connectome topology** | Graph of brain regions → topology features | Future work (GNN) |
| **Neural code** | Population coding → our feature vector = "neural code" of the volume | Conceptual |

### 7. Music Theory & Acoustics
**Origin:** Audio processing, harmonic analysis

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Fundamental + harmonics** | FFT peaks = "fundamental frequency" of brain structure | **Implemented** (power_spectrum radial bins) |
| **Timbre** | Spectral envelope = tissue texture signature | **Implemented** (power_spectrum angular bins) |
| **Rhythm** | Periodic patterns in volume = structural regularity | Not implemented |
| **Counterpoint** | Multiple independent "voices" = multiple tissue classes | Histogram captures this |

### 8. Cryptography & Security
**Origin:** Hash functions, digital fingerprints

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Robust hashing (pHash)** | DCT-based perceptual hash, Hamming distance | **Tested** — not effective alone |
| **Locality-sensitive hashing (LSH)** | Compact binary code for fast retrieval | Future work |
| **Merkle tree** | Hierarchical hash of volume chunks | Conceptual |
| **Digital watermarking** | Embed invisible features → not applicable | N/A |

### 9. Crystallography & Solid State Physics
**Origin:** X-ray diffraction, symmetry analysis

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Symmetry plane detection** | Brain's left-right symmetry → measure asymmetry | **Not implemented** — high potential |
| **Bragg peaks** | Periodic structures in brain = FFT peaks | **Implemented** (power_spectrum) |
| **Space group classification** | Brain "symmetry class" as feature | Experimental |
| **Packing density** | Foreground fraction = tissue density | **Implemented** (fg_frac in meta) |

### 10. Operations Research
**Origin:** Assignment problems, optimization

| Concept | MRI Application | Status |
|---------|----------------|--------|
| **Hungarian assignment** | Optimal query-target matching per split | Used by Gowshigan (teammate) |
| **Linear programming** | Optimize feature weights as LP | Grid search used instead |
| **Network flow** | Bipartite matching = retrieval formulation | Conceptual |

## Ideas Inspired by Teammate Approaches

### Gowshigan's SSC + Affine-Cascade (score: 1.0)
Key techniques we can borrow:
1. **Affine registration cascade** — progressively align d2/d3 volumes before comparison
2. **SSC (Self-Similarity Context)** — similar to our MIND but with spatial context
3. **Volume chunks** — split volume into regions, compare per-region
4. **Hungarian assignment** — optimal matching (not directly applicable to our retrieval setup)

### Anshu's CLIP 2D (score: 0.97)
Key insights:
1. **2D slices >> 3D volumes** for 350 training pairs
2. **500 epochs** needed (we tried 50 → 300)
3. **Pretrained backbone** helps (we tested ResNet18)
4. **Batch 128** for stable contrastive learning

## Most Promising Untested Ideas (Ranked by ROI)

| # | Idea | Source Domain | Est. Time | Expected Gain |
|---|------|---------------|-----------|---------------|
| 1 | **Multi-resolution LOD** (16³+32³+48³ concat) | Video games | 30 min | +0.01-0.02 |
| 2 | **Hemispheric asymmetry** feature | Neuroscience | 15 min | +0.005-0.01 |
| 3 | **Local entropy map** | Thermodynamics | 20 min | +0.005 |
| 4 | **Affine registration** before MI for d2 | LIDAR/ICP | 1h | +0.02-0.05 |
| 5 | **Keyframe slice selection** (max variance) | Video games | 30 min | +0.005 |
| 6 | **Normal map** of brain boundary | Computer graphics | 45 min | +0.005 |
