# Pitch Findings — Team Yantra (for the deck)
*Consolidated for the teammate building the pitch. Depth: `REPORT.md` (method+results), `NEGATIVE_RESULTS.md` (the ~22-technique frontier map).*

---

## The one-paragraph story
Training-free, deterministic cross-modal MRI retrieval: **register every brain to a neutral template by image content (Mutual Information), describe structure with SSC-12 (modality-invariant self-similarity), trim the mismatched/resected voxels, and assign**. We found a **planted geometric leak** in dataset3, measured its exact worth, refused it, and re-earned the score from anatomy. Every dead end is documented with a measured reason — we didn't stumble onto the answer, we eliminated everything else.

## Per-level "what works where" (the method map)
| Level | What works | Why |
|---|---|---|
| **d1** (common grid) | SSC directly → ~1.0 | already co-registered; content matches |
| **d2** (indep rigid+elastic) | **registration is the lever** (none 0.08 → rigid 0.78 → affine-cascade **0.87**) + SSC + light trim | voxel correspondence destroyed → must re-align |
| **d3** (preop→intraop + resection) | content-registration + SSC + **heavy trim** (drops resected voxels) | tissue removed; match the intact anatomy |
| **descriptor** | **SSC-12 > MIND-6** | neighbour↔neighbour self-similarity, modality-invariant |
| **assignment** | **Hungarian (0.930)** / Sinkhorn — exploits the eval 1-to-1 bijection | **allowed but "besides the point"** (organizer) → report both; **content/greedy 0.725 = the task** |
| **deploy trust** | **Bayesian calibration + abstention** | flags uncertain d3 cases (conf↔correct +0.68) |

## Numbers — LB-VALIDATED (real Kaggle) ✅
| Claim | LB number | Status |
|---|---|---|
| Full submission, leak d3 | **1.000** | ✅ on board — *not selected* (it's the leak) |
| Full submission, honest d3 (Hungarian) | **0.914 / 0.930** | ✅ `full_honest` / `full_honest_v2` |
| **Honest, NO leak, Hungarian (3 seeds)** | **~0.96** (0.938 / 0.953 / **1.000**) | ✅ `bbox_nocrop`+repro — *essentially matches the leak honestly*; the 1.0 is the lucky-seed tail, reliable honest ≈ 0.96 |
| **d3 leak** (resize / co-location) | **1.000** | ✅ |
| **d3 co-location BROKEN, no re-reg** | **0.259** | ✅ *the leak is worth 0.74 — proves the 1.0 is the grid* |
| **d3 honest** (co-location destroyed + content-registration) | **0.741** | ✅ `d3_strong` — earned by anatomy |
| Teammate v31 (Sinkhorn, deformable d3) | 0.923 | ✅ |

## Numbers — greedy/deployable now LB-VALIDATED ✅
| Claim | LB number | Status |
|---|---|---|
| **Greedy (single-query, deployable) macro** | **0.725** | ✅ `full_greedy` (honest, d3 leak-destroyed) — the *real* deployable number (oracle's ~0.80 was optimistic) |
| Hungarian − greedy gap (transductive boost) | **+0.205** | ✅ 0.930 (Hungarian) − 0.725 (greedy) — confirms the assignment is a real but *non-deployable* lift |

## The three pitch assets
1. **The method** — SSC + content-registration + trim. Training-free, deterministic, auditable (FDA/CE-credible). Per-level map above.
2. **Integrity** — found the planted d3 leak and **measured its worth two independent ways**: the no-reg control (1.0 → 0.26) *and* a break-test (any resize-from-array recipe drops **0.938 → 0.751** when the co-location is destroyed). We refused it and **re-earn ~0.96 honestly** via active-break + fixed-reference re-registration. Provable, not asserted.
3. **Rigor (the frontier map)** — `NEGATIVE_RESULTS.md`: ~22 techniques (8 classifiers incl. TabPFN, fusion, learned encoder, Bayesian, matrix-methods/Markov, GA, DP/DTW, NGF/NMI/shape/power-spectrum/radiomics) **all tested, all documented with why-we-tried-it → result → reason.** The bottleneck is the *content signal*, not the model. + **Bayesian calibration/abstention** as the deploy-time trust layer.

## Honest caveats (say them first)
- **Organizer ruling (Nico): bipartite (Hungarian) matching is *allowed* but "besides the point of the challenge" → report BOTH scores.** ✅ Done: **content / single-query 0.725** (the task itself) and **+ bipartite 0.930** (disclosed, not hidden). The leaderboard is saturated, so **judging now weights problem-understanding + how cool the approach is** — our leak-discovery, per-level map, and negative-results rigor are the differentiators.
- **Public LB is 27% (40/40/20 pools)** → private will derate; the assignment boost shrinks on larger pools. Because every level is *earned*, the derate is graceful.
- **Oracle is a proxy** — synthetic d3 resection is harsher than real, so some oracle numbers (e.g. optimal trim) don't transfer; we cross-check large claims on the real LB.
- **Public LB is compressed near the ceiling** — leak = a *reliable* 1.0; honest single-seed Hungarian = **0.96 ± 0.03 (0.938–1.0 over 3 seeds)** → we *essentially match the leak honestly*, but the exact 1.0 is seed-luck, not reliable. **Private LB is the real test.** (bbox-crop tested → *hurt* 1.0→0.764, not adopted; 4-seed fusion 0.930 is dragged but more stable.)
