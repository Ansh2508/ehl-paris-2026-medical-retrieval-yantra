# Negative-Results Ledger — EHL Paris 2026 Cross-Modal MRI Retrieval

> **Why this document exists.** A perfect-looking score is suspicious unless you can show you *eliminated the alternatives*. For every technique we tried that did **not** improve the honest, deployable score, we record: **(1) why we tried it** (the hypothesis), **(2) the measured result**, **(3) why it failed / what it told us.** This is the scientific-rigor moat — we didn't stumble onto the method, we ruled everything else out first. All numbers are oracle-gated (synthetic L2/L3) or real-Kaggle, greedy = per-query/deployable unless noted.

**The surviving method (what passed):** SSC-12 descriptor + content-registration (rigid→affine cascade) + trimmed matching, per-level, 96³. Honest deployable (greedy) macro ≈ 0.80 (d1 ~1.0, d2 ~0.81, d3 ~0.58); leak-free.

---

## 1. Learned / ML approaches
| Technique | Why we tried it (hypothesis) | Result | Why it failed / insight |
|---|---|---|---|
| **InfoNCE 3D contrastive encoder** (GPU, W&B) | A trained, modality-invariant embedding *should* beat a hand-crafted descriptor — especially on d3 — if heavy augmentation teaches the L2/L3 invariances. The "real" use of the 192 GB MI300X. | best oracle mean **0.4157** vs training-free **0.80** | **Tiny data (~180 pairs)** → the encoder overfits; loss kept dropping (4.16→1.7) while retrieval MRR plateaued low. **Confirms no-ML is correct here** — a zero-parameter descriptor wins. |
| TabPFN / Logistic / GBM **meta-ranker** | A tiny-data-friendly model could learn the optimal *combination* of per-pair signals without overfitting (TabPFN is built for small n). | logistic ≈ TabPFN ≈ baseline on oracle; **TabPFN *hurt* on real-d3 transfer (0.928 < 1.0 control)** | The bottleneck is the **features, not the model** — there's no complementary signal beyond SSC to combine. Also needs labels we lack for real d3 → transfer fails. |
| DINOv2 / pretrained encoders (teammate-measured) | Foundation-model embeddings might transfer zero-shot. | 0.24–0.44 | Not tuned for 3D cross-modal MRI; far below training-free. |

## 2. Fusion / multi-signal (the most-repeated negative)
| Technique | Why we tried it | Result | Why it failed / insight |
|---|---|---|---|
| **5-way rank fusion** (MIND/SSC/NGF/low-res/MI), stochastic weight-tuned, held-out | Classic "combine complementary descriptors" — each captures a different facet. | **lost at every level** | The tuner, free to weight any signal, drove the secondaries to **≈0** — when a tuner declines a signal, the signal isn't there. |
| 0.60·MIND + 0.30·SSC + 0.10·NGF (hand-weighted) | Literature-style ensemble. | 0.920 < SSC-alone 0.968 (d2) | Blending *adds variance, not information*, when signals are correlated where right + noisy where wrong. |
| **General law** | — | per-level *selection* beats every *blend* (replicated ~7×) | The recipe is "pick the best single signal per level," not fuse. |

## 3. Alternative descriptors / features
| Technique | Why we tried it | Result | Why it failed / insight |
|---|---|---|---|
| **NGF** (normalized gradient fields) | Edge-direction is intensity-free → modality-invariant, complementary to SSC. | alone d2 0.88<0.91; fused drags d3 0.42→0.34 | Gradient info is a *subset* of what SSC's self-similarity already captures. |
| **NMI / Mutual Information** on d2/d3 | The gold-standard cross-modal *registration* metric — maybe also a good *matching* score. | fusion **hurt** (d2 0.87→0.82, d3 0.33→0.27) | MI is an *alignment* metric → redundant after registration; v31 independently sets MI=0 for d2/d3. (We *do* use MI **inside** registration — that's its right job.) |
| MIND-SSC vs MIND-6 (early, z-score pipeline) | Richer 12-edge context. | initially worse (d2 0.58→0.52) — **later flipped to win** in the [0,1]+cascade pipeline | Verdict was *pipeline-dependent*; re-testing after the pipeline improved flipped it. (Lesson: a benched result can be wrong if the implementation was weak.) |
| **Multi-scale SSC** (σ=1 + σ=2) | Fine + coarse self-similarity → robust to residual misalignment. | loses to single-scale at every trim | Coarse scale adds noise, not robustness, on this data. |
| Low-res shape / brain-size / Dice | Global anatomy as a cheap discriminator. | drags d2/d3 | Too global → low patient-specificity. |
| **Alignment-invariant vectorization** (3D power-spectrum FFT, radiomics-lite) | A *registration-free* per-query vector would be cheap + deployable. | ≈ **0.12–0.16 (≈ random)** on d2/d3 | Global frequency/texture stats carry almost no *patient identity* without spatial correspondence. |
| 2D CV (montage + HOG/SIFT/ORB/LBP/SSIM) | "Turn it into an image and use CV." | not pursued (predicted-dead) | HOG/edges = NGF family (lost); SIFT/ORB/SSIM are **not modality-invariant** (the whole point of MIND); 2D discards 3D context. The running NGF result already covers the family. |

## 4. Registration variants
| Technique | Why we tried it | Result | Why it failed / insight |
|---|---|---|---|
| Single-stage affine (cold/GEOMETRY init) | Simpler than a cascade. | **0.36 — degenerates** | A cold affine over-normalizes toward the reference → kills discriminability. (→ cascade with MOMENTS init is the fix, and it won at 0.87.) |
| Deformable / BSpline as the d2 scorer | More DOF = better alignment. | 0.23 | Too flexible → over-normalizes; *and* a deformable retrieval **gain is a red flag** (warping erodes inter-subject identity → can re-create co-location). |
| **Pairwise registration as the full scorer** | Per-pair optimal alignment. | 0.50 < common-ref 0.62, 20× slower | Registering the query to *every* candidate aligns the *wrong* pairs too → less discriminative. |

## 5. Hyperparameters / preprocessing
| Technique | Why we tried it | Result | Why it failed / insight |
|---|---|---|---|
| **128³ resolution** | More voxels = more anatomical detail. | apparent +0.30 d2 — **but a synthesis artifact** | The oracle's voxel-defined distortions get *relatively milder* at higher res → inflated score that won't transfer. **96³ is the real sweet spot** (resolution-fair grid: 64→96 real, 96→128 flat). A caught self-deception. |

## 6. Re-ranking / assignment (deployability-flagged)
| Technique | Why we tried it | Result | Why it failed / insight |
|---|---|---|---|
| k-reciprocal re-ranking (Zhong) | Neighborhood-consistency re-rank. | benched | Superseded by optimal assignment on the bijection. |
| **Hungarian / Sinkhorn (assignment)** | The eval is a known N↔N bijection → exact 1-to-1 resolves collisions. | greedy d2 0.81 → 1.0 | **Works, but it's *transductive* benchmark-structure, not anatomy** — non-deployable (a real OR has no bijection). We report it *only* as a disclosed leaderboard optimization; the **honest deployable number is greedy.** |

## 7. The d3 leak (declined on principle)
| Technique | Why it "works" | Decision |
|---|---|---|
| Affine/qoffset/co-location match on d3 | Organizer resampled targets into the query's grid → true pair voxel-aligned → MRR 1.0 with zero anatomy. Audit: 20/20. | **Refused.** Co-location-destroyed control → 0.205–0.26 (proves it's the grid). We re-earn d3 by content-registration. |

---

## 8. Trained encoder — precise per-level evidence (W&B `jw4t3jnf`)
*Why:* a learned modality-invariant embedding *should* beat a hand-crafted descriptor with enough augmentation. *Result (greedy MRR over training):* d1 0.25→**0.78**, d2 0.08→**0.28**, d3 0.12→**0.27**, mean→**0.44** — *worse at every level* than training-free SSC (1.0/0.81/0.58), still climbing slowly, d3 barely moved. *Why it failed:* ~180 pairs → the encoder can't out-learn a zero-parameter descriptor + explicit registration. **Insight: explicit registration ≫ learned invariance on tiny data.**

## 9. Retrieve-then-rerank (MEASURED, honest-d3 greedy & Hungarian, n=20×2 seeds)
| Arm | Why we tried it | Result (greedy / Hung) | Verdict |
|---|---|---|---|
| baseline SSC | — | 0.291 / 0.399 | reference |
| **feature-fusion rerank** (SSC + residual-p95 + NMI) | residual-distribution shape + NMI might add signal the trimmed-mean misses | **0.258 / 0.324 — WORSE** | ❌ NMI + residual-tail carry no complementary info; adding them drags. Confirms the feature family is empty. |
| **pairwise re-registration rerank** (re-register SSC top-k to the query, finer SSC) | sharper per-pair alignment on the few good candidates could lift rank-1 | 0.319 / 0.440 (**+0.03 / +0.04**) | ⚠️ small positive **but within noise** — the only thing that's nudged greedy-d3 up leak-free; needs a multi-seed confirm before trusting. |

## 10. Bayesian methods
| Form | Why | Result | Verdict |
|---|---|---|---|
| **Bayesian classification** (Gaussian Naive Bayes, GP classifier) over per-pair features | a *Bayesian* classifier might find structure the logistic/GBM/TabPFN rankers missed, and it's better-calibrated | greedy ties baseline (0.21≈0.21), **Hungarian worse** (0.31–0.35 < 0.41) | ❌ same wash — the prior improves calibration, not accuracy; no signal in the features to classify on. |
| **Bayesian search** (Optuna TPE) | principled HPO of the honest-d3 knobs | running (affine-only, after a deformable-hang fix) | TBD — expected to confirm near-optimality (landscape is flat/signal-thin). |
| **Bayesian calibration / uncertainty** (per-query posterior + abstention) | *not* for MRR — for deployable trust: flag low-confidence queries | **BUILT — works.** confidence↔correctness corr **+0.68**; abstain on uncertain half → answered-set MRR 0.22→0.49 (rank-1 12%→42%) | ✅ **the one Bayesian win** — clinical/abstention value, not score. |
| **SVM (RBF), GMM-Bayes, voting ensemble** | binary match-classifier / mixture / mélange-de-modèles | greedy 0.15–0.26 (≈ or < baseline 0.21), Hungarian all worse | ❌ same wash. |

> **★ CLASSIFIER-FAMILY CLOSEOUT (definitive):** **8 models** now tested as the match-classifier / meta-ranker — logistic, GBM, **TabPFN**, Gaussian-NB, GP, **SVM-RBF, GMM-Bayes, voting ensemble** — **every one ties-or-loses to plain SSC.** The bottleneck is *proven* to be the **features** (no signal beyond SSC self-similarity), **not the model.** No classifier / mixture / ensemble / Bayesian / learned model changes this. Direction exhausted.

## 11. Matrix manipulation (on the score matrix S[Q,G])
| Method | Why | Result (greedy/Hung) | Verdict |
|---|---|---|---|
| **column (or Sinkhorn) normalization** | demote "popular" gallery targets that match every query | **0.64 / 0.65** vs base 0.22/0.56 | ✅ **real & legitimate (no leak), BUT transductive (batch)** — = what the team's Sinkhorn already does; more-deployable-than-Hungarian (offline gallery calibration). Not a *new* lever, validates Sinkhorn. |
| row normalization (pure per-query) | — | 0.224 / 0.578 | ❌ no greedy gain — all the lift is the column step (the batch part). |
| low-rank SVD denoise (k=5,10) | remove noisy pairwise scores | 0.11–0.14 | ❌ worse — overfits the small gallery (as predicted). |
| multi-score matrix fusion | combine S_mind+S_nmi+... | — | ❌ = the fusion family, proven empty (§2). |
| graph diffusion / Personalized PageRank | propagate on Q-Q/G-G/Q-G | not run (needs Q-Q/G-G recompute) | ⏸️ same transductive family as Sinkhorn/k-reciprocal; prior says ≈ Sinkhorn. |
| Mahalanobis / whitening | — | n/a | the *embedding* path; encoder lost (§1,§8). |

> **Matrix takeaway:** the *only* big matrix lever is column/Sinkhorn normalization — and it's **transductive (batch)**, already in the team's method, not a single-query-deployable gain. Per-query (deployable) matrix ops give nothing.

## 12. DTW slice-alignment & Optuna (final two — both confirm the frontier)
| Technique | Why | Result | Verdict |
|---|---|---|---|
| **DTW axial-slice-alignment rerank** | sequence alignment with gap-penalties → skip resected slices (different paradigm than voxel overlap) | greedy 0.221 < base 0.269; Hungarian 0.221 ≪ 0.445 | ❌ **worse** — the voxel-**trim already does resection-robustness**; 2-D slice descriptors are weaker than 3-D SSC; 1-D DTW can't undo 3-D rotation (registration does). New paradigm, no gain. |
| **Optuna (Bayesian search), 15 trials** | tune reg-iters/samp/shrink, SSC radius/σ, trim on the strict honest-d3 objective | best **0.446** at `{trim:0.9, σ:0.8}` vs 0.27 default | ⚠️ **confirms near-optimality** — the +0.15 is **fitting the oracle's too-aggressive 6-hole synthetic resection** (keep best 10%); real-d3 trim≈0.75 is validated, so it's an oracle-calibration artifact, *not* a transferable HPO gain. |

> **EXPLORATION COMPLETE (~22 techniques).** The honest deployable frontier is fixed: **SSC-12 + content-registration (rigid→affine cascade) + per-level trim + greedy (deployable) / Sinkhorn-or-Hungarian (batch) + Bayesian calibration (trust).** Everything else — 8 classifiers, fusion, learned encoder, NGF/NMI/shape/power-spectrum/radiomics, multi-scale, deformable-as-scorer, low-rank, DTW, GA(redundant) — is documented above with *why we tried it → result → reason*. The bottleneck is proven to be the **content signal**, not the model/search/matrix method.

## 13. Crop-to-brain-bbox (from Wilfred's branch) + the public-LB-saturation insight (MEASURED on real LB)
*Why:* Wilfred's `Experiment_Log.md` **and** our own oracle A/B both said crop-to-brain-bbox lifts d2/d3 (oracle: d2 Hungarian 0.921→**1.000**, d3 greedy 0.318→0.404). The one genuine untested lever from his branch.

| Real-LB A/B (matched pipeline, single-seed honest d3, Hungarian) | Public MRR | Verdict |
|---|---|---|
| no-crop baseline | **1.00000** | — |
| + bbox-crop (d2/d3) | **0.76373** | ❌ **bbox HURTS −0.236** |

*Why it failed:* (i) the no-crop baseline **already saturates** the public pool (1.0) → zero headroom; (ii) our crop (`>0.05` mask bbox) likely clips/misaligns on real volumes. **Reaffirms the keystone lesson (REPORT §2): the oracle is a proxy — validate every large lever on the real LB.**

> **METHODOLOGICAL INSIGHT — public LB compressed near the ceiling; honest matches the leak.** The **leak** (`full_clean`) scores a *reliable* **1.0**; the **honest** pipeline (d3 co-location destroyed, verified by greedy=0.725) scores **0.96 ± 0.03 across 3 break-seeds (1.000 / 0.953 / 0.938)**. Corollaries: **(a)** our honest method *essentially matches the leak* — **we do not need it**; **(b)** the exact 1.0 (`bbox_nocrop`) is the **lucky-seed tail, NOT reliable** — single-seed mean 0.96 > 4-seed fusion 0.930 (fusion drags slightly per §6.2, but is more stable across seeds); **(c)** the **private LB is the real arbiter**; defensible honest estimate **~0.96 (Hungarian) / 0.725 (greedy)**. *Final selection: hedge `bbox_nocrop` (1.0 tail) + `full_honest_v2` (0.930 stable).*

## 14. d3 break-test on the real LB — "resize-from-array + template-reg" is NOT leak-free
A common d3 recipe (load from own array + register to a gallery template, **no active break**) was believed leak-free ("co-location not inherited"). The real-LB break-test refutes it (d1+d2 held constant, Hungarian):

| d3 recipe | macro | reading |
|---|---|---|
| resize-from-array + gallery-template-reg, **NO break** | **0.938** | rides the co-location |
| + active ±20° rigid break before reg | **0.751** | **−0.19 macro (≈ −0.56 in d3)** once alignment is destroyed |

*Why:* the co-location is baked into the **array** (not the affine header), so resize-from-array preserves it, and registering both volumes to a common template preserves it → **leak-assisted, not leak-free**. Worse, the **gallery template itself depends on co-location** — the mean of independently-rotated brains is a near-spherical blob, so once broken, template-registration collapses (honest d3 ≈ 0.35).

*The honest recipe that actually works (most proficient):* resize-from-array → **active break** → register to a **FIXED real reference** (a d1 query, not a gallery-mean) → SSC + trim75 + Hungarian = **~0.96** (`bbox_nocrop` 1.0 / repro 0.95 / 0.94). Locked final candidate = `FINAL most-proficient honest`.

---
*Discipline going forward: every experiment is logged here as hypothesis → result → reason, whether it wins or loses.*
