# Non-ML Cross-Modal Brain MRI Retrieval

**Author:** Wilfred Doré (wilfred.dore@telecom-paristech.org), Independent Researcher

**Score:** MRR 0.735 on public Kaggle leaderboard, without any machine learning training.

## Citation

If you use this code, please cite:

```bibtex
@misc{dore2026nonml,
  title={Pure Mathematical Feature Engineering for Cross-Modal Brain MRI Retrieval},
  author={Dor\'e, Wilfred},
  year={2026},
  note={Developed with assistance from GLM-5.2 (Z.AI)}
}
```

## Acknowledgments

Developed with the assistance of **GLM-5.2** (Z.AI), an open-source large language model.

## Method

Pure mathematical feature engineering combining:
- **Mutual Information** (information theory) for registered datasets
- **3D Power Spectrum** via FFT (signal processing, translation-invariant)
- **MIND descriptor** (modality-independent self-similarity)
- **Brain shape fingerprint** (global morphology, robust to missing tissue)
- **Hemispheric asymmetry** (individual neuroanatomical trait)
- **Multi-resolution LOD** (16/32/48 voxel representations)
- **Gradient magnitude** (modality-invariant edges)

13 feature types, weighted distance, grid search optimization.
Runs in 5 minutes on 20-core CPU, 0 GPU, 0 training.

See `paper_draft.pdf` for the full paper and `slides_beamer.pdf` for the presentation.
