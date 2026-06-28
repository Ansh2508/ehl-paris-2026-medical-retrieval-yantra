# Non-ML Cross-Modal Brain MRI Retrieval

**Author:** Wilfred Dore (wilfred.dore@telecom-paristech.org), Independent Researcher
**Score:** MRR 0.928 on public Kaggle leaderboard, without any machine learning training.
**Version:** v33 (final)

## Quick Start

```bash
# On a machine with GPU (AMD MI300X or NVIDIA) and the competition data:
python v33_final.py --data-root /path/to/data --out submission.csv
```

## Citation

```bibtex
@misc{dore2026nonml,
  title={Pure Mathematical Feature Engineering for Cross-Modal Brain MRI Retrieval},
  author={Dor\'e, Wilfred},
  year={2026},
  note={Developed with the assistance of GLM-5.2 (Z.AI)}
}
```

## Acknowledgments

Developed with the assistance of **GLM-5.2** (Z.AI), an open-source large language model.

## Method

Pure mathematical feature engineering combining:
- **SSC-12** (12-edge self-similarity context descriptor, GPU)
- **Mutual Information** (information theory, d1 only)
- **3D Power Spectrum** via FFT (signal processing, translation-invariant)
- **Brain shape fingerprint** (global morphology, robust to missing tissue)
- **Normalized distance combination** (key innovation: per-feature [0,1] scaling)
- **Hungarian algorithm** (optimal 1-to-1 bijection assignment)
- **Affine registration** for d2/d3 (SimpleITK, Mattes MI)

13 feature types, normalized weighted distance, Hungarian assignment.
Runs in ~7 minutes on 20-core CPU with GPU-accelerated SSC-12.
Zero training, zero gradient descent, zero random seeds.

## Why No ML?

With only 350 training pairs, deep learning approaches (CLIP, ResNet, BrainIAC) all scored 0.24-0.43. Mathematical feature engineering scored 0.928. For clinical deployment (FDA/CE certification), each feature has a physical justification and the system is fully deterministic and auditable.

## Documentation

| File | Description |
|------|-------------|
| `v33_final.py` | Final source code (MRR 0.928) |
| `paper_draft.pdf` | Research paper |
| `slides_beamer.pdf` | Presentation slides |
| `ARCHITECTURE.md` | Architecture documentation (SADT, data flow) |
| `architecture.puml` | PlantUML pipeline diagram |
| `CHANGELOG.md` | Version history and score progression |
| `Experiment_Log.md` | Detailed experiment log |
| `Research_Ideas_CrossDomain.md` | Cross-domain inspiration catalog |
| `Non_ML_Cross_Domain_Ideas.md` | Non-ML ideas from LIDAR, spectroscopy, etc. |
| `Infrastructure_Notes.md` | AMD MI300X and Kaggle setup notes |

## Score Progression

```
0.131 → 0.583 → 0.647 → 0.724 → 0.839 → 0.881 → 0.923 → 0.928
 floor    v2      v9      v21     v26     v29     v31     v33
```

## License

This work is shared for research purposes. Please cite the author if used.
