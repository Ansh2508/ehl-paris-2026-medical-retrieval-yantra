# Yantra — Cross-Modal 3D Brain-MRI Patient Re-Identification
**EHL Paris 2026 · Team Yantra**

Training-free, deterministic retrieval: match a contrast-enhanced T1 (ceT1) query to the **same patient's** T2 in a gallery, across three datasets of increasing geometric difficulty (co-registered → independent rigid+elastic → preop→intraop with resection). We found, measured, and **refused** a planted geometric leak in dataset3 — and re-earn the score from anatomy alone.

## Results (real Kaggle leaderboard)
| Setting | MRR |
|---|---|
| Honest, no leak — Hungarian / bipartite | **~0.96** |
| Honest, no leak — greedy (single-query, deployable) | **0.725** |
| With the d3 leak (found, **declined**) | 1.000 |

d3 leak ladder, measured: **1.0** (leak) → **0.26** (co-location destroyed, no re-registration) → **~0.96** (re-earned by content registration). Both greedy and bipartite scores are reported per the organizer's ruling.

## Method (`method/`)
Register each brain to a neutral reference by **image content** (Mutual Information) → describe structure with **SSC-12** (modality-invariant self-similarity) → **trim** the mismatched/resected voxels → assign. No training, no labels, fully deterministic.

> For the leak-prone d3, the genuinely leak-free pipeline first **destroys** the planted co-location and re-registers to a fixed reference. `method/d3_earned.py` is the resize-from-array variant; the break-test that proves the co-location's worth and the honest recipe are in `docs/NEGATIVE_RESULTS.md` (§14) and `docs/WHAT_WORKS.md`.

## Layout
| Path | Contents |
|---|---|
| `method/` | frozen per-level pipeline: `d1_ssc.py`, `d2_trim_hungarian.py`, `d3_earned.py` |
| `experiments/` | controls, ablations & alternates (d3 leak controls, slice-CLIP baseline, …) |
| `research/` | the oracle-gated framework: synthetic oracle, learned-encoder track, descriptors, fusion, `leak_audit.py` |
| `docs/` | `REPORT.md` (full writeup) · `WHAT_WORKS.md` (validated techniques) · `NEGATIVE_RESULTS.md` (~23 ruled out, with reasons) · `PITCH_FINDINGS.md` · `CHALLENGE.md` (task spec) |
| `slides/` | pitch deck (`deck.pdf`, `deck.tex`) + retrieval visual |
| `submissions/` | Kaggle submission CSVs |
| `assets/` | example query–target pairs |
| `Wilfred_No_ML/` | teammate's self-contained non-ML pipeline (v15–v36) + its own docs & slides |

## Reproduce
```bash
pip install -r research/requirements.txt        # SimpleITK, NumPy/SciPy, nibabel, …
python method/d1_ssc.py            --data-root <DATA>
python method/d2_trim_hungarian.py --data-root <DATA>
python method/d3_earned.py         --data-root <DATA>
```

## Integrity
The complete AI-agent session history is recorded with [Entire](https://entire.io) on the **`entire/checkpoints/v1`** branch (checkpoint `8f0fa7bcb3e2`) — auditable for cheating-detection review.
