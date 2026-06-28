# Design.md — Architecture & module contracts

> Read alongside `CLAUDE.md` (decisions/guardrails) and `GLOSSARY.md` (terms).
> This file = the data flow and the contract each module must honour, so modules
> can be implemented independently and still snap together.

## Pipeline
```
NIfTI volumes (.nii.gz, RAS, 1mm, variable shape)
        │
        ▼  preprocess.py        resample 96/128³ · clip 0.5–99.5% · z-score · brain mask · cache
   preprocessed tensors [1,D,H,W]
        │
        ├─────────────► Branch A: encoder.py + train.py (InfoNCE, domain-randomization aug)
        │                       → embeddings  → infer.py (TTA) → ranking_A
        │
        └─────────────► Branch B: mind.py (training-free) → ranking_B
                                 (strong on L1 & L3, ~useless on L2)
        │
        ▼  fuse.py              per-level rank fusion: w·A ⊕ (1−w)·B   (w from oracle)
        ▼  veto.py              shape/size consistency + agreement rerank (registration veto optional, oracle-gated)
        ▼  submit.py            write 377-row Kaggle CSV
```
Everything is validated through `oracle.py` (simulated L1/L2/L3) BEFORE any Kaggle submit.

## The `scorer` contract (the glue)
A **scorer** is any callable that ranks a gallery for a query:
```python
scorer(query_id: str, gallery_ids: list[str]) -> list[str]   # gallery_ids ranked best→worst
```
`oracle.py` and `submit.py` both consume a scorer, so MIND-only / embedding-only / fused are interchangeable and all measured the same way. Build scorers by composing: embedding ranking (infer) ⊕ MIND ranking (mind) via fuse, optionally vetoed.

## Module contracts
| Module | Key functions | In → Out |
|---|---|---|
| `config.py` | `Config` dataclass | all knobs, one source of truth |
| `preprocess.py` | `preprocess_volume(path,cfg)`, `build_cache(manifest,cfg)` | path → tensor [1,D,H,W]; cache |
| `augment.py` | `train_transforms(cfg)`, `l2_transforms(cfg)`, `l3_transforms(cfg)` | cfg → MONAI `Compose` |
| `encoder.py` | `Encoder(cfg)`, `.encode(vol)` | tensor → L2-normed embedding [d] |
| `train.py` | `info_nce_loss(...)`, `train(cfg)` | trains encoder, saves checkpoint |
| `mind.py` | `mind_rank(q_vol, gallery)`, `mind_score_matrix(...)` | volumes → ranking / scores |
| `infer.py` | `embed_images(enc,ds,tta_n)`, `rank_by_embedding(q,gallery)` | → dict id→emb / ranking |
| `fuse.py` | `rank_fusion(rankings,weights)`, `fuse_per_level(...)` | rankings → fused ranking |
| `veto.py` | `apply(ranking,query,gallery,cfg)` | ranking → vetoed ranking |
| `metrics.py` | `mrr(rankings,truth)`, `reciprocal_rank(...)` | → float |
| `oracle.py` | `evaluate(scorer,cfg)`, `synth_fake_l2/l3`, `make_holdout` | scorer → {l1,l2,l3,mean} |
| `submit.py` | `build(prediction_sets,scorer)`, `write_submission(rows,path)` | → 377-row CSV |

## Why these choices (short)
- **96/128³, not 256³**: 350 pairs → capacity isn't the bottleneck, generalization is; matches SwinUNETR pretraining; fits timing. Use the 192GB for **batch (InfoNCE negatives)**, parallel runs, TTA.
- **Domain randomization is the engine**: synthesizes L2/L3 conditions from L1 so the model generalizes with zero target labels. Pedigree: SynthSeg/SynthMorph.
- **MIND anchor**: training-free, modality-invariant; voxel correspondence survives on L1 & (roughly) L3, so it's a real anchor exactly where a learned model may wobble. Dies on L2 → fusion handles that per level.
- **Simulated oracle is non-negotiable**: it's the only way to measure L2/L3 offline and to gate the veto. Without it, half the system is faith.
