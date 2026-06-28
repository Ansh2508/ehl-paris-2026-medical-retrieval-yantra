# Research Ideas & Cross-Domain Inspirations

**Wilfred Doré | Team Yantra | Paris Research Hackathon 2026**

This document captures brainstorming ideas spanning multiple engineering and scientific disciplines, with assessments of their applicability to the brain MRI cross-modal retrieval challenge.

## 1. Signal Processing & Spectroscopy

### 1.1 Power Spectrum (FFT Magnitude) — Translation Invariant
**Origin:** Radio frequency engineering, spectroscopy, audio processing

**Key theorem:** The magnitude of the Fourier transform is invariant to translation. A shift in spatial domain only affects the phase, not the magnitude of the frequency spectrum.

**Application to d2 (geometric deformations):**
- Dataset2 applies random rigid transformations (rotation + translation) to query and target independently
- The power spectrum `|FFT(volume)|²` is invariant to translation by construction
- For rotation: a rotation in spatial domain = rotation in frequency domain → take radial mean of power spectrum → rotation-invariant feature
- This gives a **compact, transformation-invariant fingerprint** of each volume

**Implementation:** `np.fft.fftn(volume)` → `np.abs()` → `np.log1p()` → radial binning → 16-32 values per volume

**ROI:** Very high for d2. Fast to compute (O(N log N)). ~20 min to implement.

### 1.2 Wavelet 3D Decomposition — Multi-Scale Structure
**Origin:** Image compression (JPEG 2000), telecom (OFDM), seismic analysis

**Key idea:** Decompose the 3D volume into frequency bands at multiple scales. Low-frequency bands capture global brain shape, high-frequency bands capture fine details (lesions, edges).

**Application:**
- d1: wavelet low-frequency band ≈ coarse structure → robust cross-modal comparison
- d2: wavelet is naturally multi-scale → robust to non-linear deformations at different scales
- d3: missing tissue affects mainly certain frequency bands → other bands still discriminative

**Implementation:** `pywt.wavedec3(volume, 'db4', level=3)` → extract approximation + detail coefficients → statistics per band

**ROI:** High. Wavelets are the natural multi-scale extension of our current gradient features.

### 1.3 Cepstral Features
**Origin:** Speech recognition, audio processing

**Key idea:** The cepstrum is the inverse FFT of the log power spectrum. It separates source (excitation) from filter (vocal tract) in speech. Analogously for MRI: separate tissue structure (filter) from acquisition artifacts (source).

**Application:** Could decorrelate scanner-specific artifacts from anatomical structure, making features more robust across acquisitions.

**ROI:** Experimental. Novel but unproven for MRI retrieval. Worth a quick test.

### 1.4 Phase Correlation
**Origin:** Image registration, stereo vision

**Key idea:** Cross-power spectrum between two images → inverse FFT → peak location gives translation offset. 

**Application for d2:** Estimate the rigid translation between query and target, then align before comparison. But this requires per-pair computation (350×100 = 35K pairs for d2 test) — may be too slow.

**ROI:** Medium. Could improve d2 if combined with MI after alignment.

## 2. Graph Neural Networks & Topological Data Analysis

### 2.1 Brain Connectivity Graph
**Origin:** Network science, computational neuroscience

**Key idea:** Model the brain as a graph where nodes are anatomical regions and edges are structural/functional connections. Two scans from the same subject should have similar graph topology.

**Implementation:**
1. Parcellate the brain into regions (using intensity clustering or AAL atlas)
2. Build adjacency graph based on spatial proximity + intensity correlation
3. Extract graph features: degree distribution, clustering coefficient, spectral gap
4. Compare graphs between query and target

**ROI:** Medium-high for d3 (missing tissue changes graph locally but global topology preserved). Complex to implement in remaining time.

### 2.2 Persistent Homology (Topological Data Analysis)
**Origin:** Algebraic topology, computational geometry

**Key idea:** Compute the "shape" of the brain volume at multiple scales using persistence diagrams. The Betti numbers (number of connected components, holes, voids) at different thresholds capture topological features that are invariant to deformation.

**Application:** Brain topology is subject-specific (ventricle shape, sulcus pattern). Missing tissue changes local topology but persistent features remain.

**ROI:** High creativity score for presentation. Medium practical ROI. Libraries: `gudhi`, `ripser`.

### 2.3 Graph Neural Networks (GNN)
**Origin:** Deep learning on graphs

**Key idea:** Build a spatial graph from the 3D volume (superpixels or voxels as nodes), then use message passing to learn node embeddings. Pool into a graph-level embedding for retrieval.

**Challenge:** 48³ = 110K nodes → too large for standard GNN. Would need superpixel reduction first.

**ROI:** Low for remaining time. Better as future work / paper material.

## 3. Mixture of Experts (MoE)

### 3.1 Dataset-Specific Expert Routing
**Origin:** Large language models (Mixtral, GShard)

**Key idea:** Train separate "expert" models for each dataset difficulty level:
- Expert 1: MI-based comparison (d1 registered)
- Expert 2: MIND + power spectrum (d2 deformed)
- Expert 3: brain fingerprint + topological features (d3 missing tissue)
- Router: learned or heuristic (based on dataset column)

**Application:** Our current approach is already a soft MoE (different weights per dataset). A hard MoE with explicit routing could be more interpretable and performant.

**ROI:** Medium. We already do this manually. Formalizing it helps the presentation narrative.

### 3.2 Feature-Level MoE
**Origin:** Multi-task learning

**Key idea:** Instead of fixed weights per dataset, learn a gating function that dynamically selects which features to emphasize based on the input volume's properties (e.g., how deformed it appears).

**Implementation:** Small MLP that takes volume statistics as input and outputs feature weights.

**ROI:** Low practical (we only have 3 datasets). High for paper novelty.

## 4. Chimiometrics & Multivariate Analysis

### 4.1 PCA on Feature Matrix
**Origin:** Chemometrics, spectral analysis

**Key idea:** Our 9 feature types produce vectors of different lengths. Concatenating them and applying PCA decorrelates the features and reduces dimensionality.

**Application:** Could improve the L2/cosine distance computation by working in a decorrelated space.

**ROI:** Low. Our features are already somewhat independent by design. Quick to test though.

### 4.2 Partial Least Squares (PLS)
**Origin:** Chemometrics

**Key idea:** PLS finds directions of maximum covariance between two sets of variables. Could be used to find the optimal linear combination of features that maximizes retrieval accuracy.

**ROI:** Low. We already do grid search over weight combinations.

## 5. Biomimetics & Biological Inspiration

### 5.1 Visual Cortex Hierarchy
**Origin:** Neuroscience

**Key idea:** The visual cortex processes information in a hierarchy: V1 (edges) → V2 (textures) → V4 (shapes) → IT (objects). Our features mirror this:
- Gradient magnitude ≈ V1 (edge detection)
- MIND ≈ V2 (local texture)
- Brain shape fingerprint ≈ V4/IT (global shape)

**Insight:** Adding a "V2-like" texture feature (e.g., Gabor filters 3D, local binary patterns 3D) could fill the gap between edges and global shape.

**ROI:** Medium. Gabor filters 3D are implementable but computationally expensive.

### 5.2 Auditory Cortex Spectral Processing
**Origin:** Auditory neuroscience

**Key idea:** The cochlea performs a wavelet-like decomposition of sound. The auditory cortex then computes spectral templates for recognition. Analogously: compute "spectral templates" of MRI volumes in frequency domain.

**Connection to 1.1:** The power spectrum approach is exactly this — a spectral template of the brain volume.

## 6. Mechanical Engineering & Continuum Mechanics

### 6.1 Strain Tensor Analysis
**Origin:** Solid mechanics, elasticity theory

**Key idea:** If we estimate the deformation field between a query and target, the strain tensor captures local stretching/compression. Volumes from the same subject (even deformed) should have low strain in most regions.

**Application for d2:** Estimate affine deformation between volumes, compute strain, use as similarity measure.

**Challenge:** Requires registration, which is slow. Better as future work.

**ROI:** Low for remaining time. High for paper.

### 6.2 Modal Analysis (Vibration Modes)
**Origin:** Structural engineering, acoustics

**Key idea:** Just as structures have natural vibration modes, the intensity distribution of a brain volume can be decomposed into "modes" (eigenvectors of the Laplacian). These modes are shape descriptors.

**Application:** Spectral clustering / Laplacian eigenmaps of the volume → compact shape descriptor robust to deformation.

**ROI:** Medium. Novel and creative. Implementable via `scipy.sparse.linalg.eigsh` on the volume Laplacian.

## 7. Information Theory Beyond MI

### 7.1 Jensen-Shannon Divergence
**Origin:** Information theory

**Key idea:** JSD is the symmetric, smoothed version of KL divergence. It's bounded and always defined (unlike KL). Could replace or complement MI.

**ROI:** Low. MI already works well. JSD would give similar results.

### 7.2 Transfer Entropy
**Origin:** Causal inference, neuroscience

**Key idea:** Measures directional information flow. Not directly applicable (we don't have time series), but the concept of "information flow" between spatial regions could inspire spatially-weighted MI.

**ROI:** Theoretical. No direct implementation.

## 8. Future Work & Paper Directions

### 8.1 Self-Supervised Pretraining (M3Ret-style)
- Train MAE + SimDINO on all 1174 volumes (unlabeled) → learn modality-agnostic representations
- Fine-tune with InfoNCE on 350 d1 pairs
- Reference: M3Ret (arXiv:2509.01360, Sep 2025)

### 8.2 Cross-Modal Synthesis + Retrieval
- Train a GAN/diffusion model to synthesize T2 from T1
- Compare synthesized T2 with gallery T2 using within-modal metrics (SSIM, NCC)
- Reference: CMS-MCSR frameworks with deformable convolutions

### 8.3 Geometry-Aware CLIP with 2D Slices
- Anshu's approach (0.97 score): 2D slice CLIP with 500 epochs, batch 128
- 2D is more parameter-efficient than 3D for 350 samples
- Could ensemble with our non-DL features

### 8.4 Mixture of Experts Framework
- Formalize the per-dataset weight selection as an MoE
- Each expert is a feature family (MI, MIND, brain shape, power spectrum)
- Router selects based on dataset characteristics
- Novel contribution for paper

## Bibliography (Key References)

1. Heinrich et al., "MIND: Modality Independent Neighbourhood Descriptor for multi-modal deformable registration", Medical Image Analysis, 2012
2. Maes et al., "Multimodality image registration by maximization of mutual information", IEEE TMI, 1997
3. Liu et al., "M3Ret: Unleashing Zero-shot Multimodal Medical Image Retrieval via Self-Supervision", arXiv:2509.01360, 2025
4. Demir et al., "Multigradicon: A foundation model for multimodal medical image registration", MICCAI 2024
5. Zhou et al., "macJNet: weakly-supervised multimodal image deformable registration using joint learning framework and multi-sampling cascaded MIND", BioMedical Engineering OnLine, 2023
6. Singh et al., "Cross-Modal Brain MRI Synthesis Via Contrastive Disentanglement and Conditional Diffusion", IEEE JBHI, 2026
