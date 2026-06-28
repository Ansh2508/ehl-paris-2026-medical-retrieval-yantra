# Brain MRI Cross-Modal Retrieval Challenge

Your task is to build a cross-modal medical image retrieval system. For each query brain MRI volume, rank all candidate target MRI volumes from the matching gallery so that the true same-subject target appears as high as possible.

## Kaggle Challenge

https://www.kaggle.com/t/b33ec3e76c3d4e16a6b56852470b3ebf

## Our Solution (v36 — Honest/Deployable, MRR 0.723)

**Team Yantra | Wilfred Doré | June 2026**

A 100% non-ML, 0-training, deterministic pipeline for cross-modal (T1→T2)
brain MRI retrieval. The canonical submission script is
[`v36_final.py`](v36_final.py).

### Pipeline

1. **Preprocessing**: foreground mask → robust percentile scaling → brain
   bbox crop → trilinear resize to 48³.
2. **Registration** (d2/d3 only): d3 volumes first receive an independent
   random rigid transform (±15°, ±3 voxels) to destroy the dataset's
   co-location leak, then a MOMENTS-init rigid→affine cascade (200 iters
   total, shrink [8,4,2,1]) aligns each volume to a shared d1 reference.
3. **Feature extraction**: SSC-12 self-similarity descriptor (GPU), mutual
   information (d1 only), FFT power spectrum, brain shape fingerprint,
   gradient magnitude, intensity histogram, 3D projections.
4. **Distance fusion**: per-feature normalization to [0,1] + weighted sum
   with per-dataset weights (SSC-dominant for d2/d3).
5. **Trimmed SSC** (d3 only): keep best 50% of per-voxel residuals for
   resection robustness.
6. **Ranking**: greedy row-wise argsort (deployable, no Hungarian bijection
   exploitation).

### Honesty Guarantees

- **No co-location leak**: d3 random rigid per volume before registration.
- **No eval-bijection exploitation**: greedy ranking instead of Hungarian.
- **Deployable**: each query ranked independently — valid for single-query
  retrieval in clinical deployment.

### Results

| Version | MRR | Notes |
|---------|-----|-------|
| v33 | 0.928 | Leaky (d3 co-location + Hungarian bijection) |
| v34 | 0.686 | Honest but under-optimized (SSC-only d2/d3) |
| v35 | 0.707 | + restored features + tuned d3 params |
| **v36** | **0.723** | **+ MI post-reg d2/d3 + reduced rigid + less trim** |

### Acknowledgments

Developed with the assistance of GLM-5.2 (Z.AI), an open-source large
language model. Technical review by Gowshigan R. and Claude (Anthropic).

### Key Files

- [`v36_final.py`](v36_final.py) — canonical submission script
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full architecture documentation
- [`architecture.puml`](architecture.puml) — PlantUML pipeline diagram
- [`Experiment_Log.md`](Experiment_Log.md) — experiment history and findings
- [`CHANGELOG.md`](CHANGELOG.md) — version history
- [`paper_draft.pdf`](paper_draft.pdf) — research paper draft
- [`slides_beamer.pdf`](slides_beamer.pdf) — presentation slides

---## Modalities

- Query: T1 post-contrast MRI
- Target: T2 MRI

All images are 3D NIfTI `.nii.gz` volumes converted to RAS orientation.

## Data Layout

The data is split into three independent datasets:

```text
dataset1/
  train_pairs.csv
  val_queries.csv
  val_gallery.csv
  test_queries.csv
  test_gallery.csv
  images/
    train/
    val/
    test/

dataset2/
  val_queries.csv
  val_gallery.csv
  test_queries.csv
  test_gallery.csv
  images/
    val/
    test/

dataset3/
  val_queries.csv
  val_gallery.csv
  test_queries.csv
  test_gallery.csv
  images/
    val/
    test/

sample_submission.csv
```

`dataset1` includes labelled training pairs. `dataset2` and `dataset3` have no labelled training pairs and are intended to evaluate generalization.

## Dataset Descriptions

### Dataset 1

`dataset1` contains preoperative MRI pairs only. It provides the labelled training set for the challenge: each row in `dataset1/train_pairs.csv` gives a matching T1 post-contrast query image and T2 target image from the same subject.

All `dataset1` pairs, including training, validation, and test pairs, are registered to a common image grid. You may use that fact when training on the labelled pairs and when developing methods on this dataset.

`dataset1` validation and test data are provided as query/gallery retrieval pools. The correct matches are hidden and are used for leaderboard scoring.

### Dataset 2

`dataset2` contains preoperative MRI pairs from the same source setting as `dataset1`, but the validation and test images have random rigid rotation/translation and non-linear deformations applied. Query and target images in a correct pair are deformed independently, so they no longer share one common geometry.

No labelled training pairs are provided for `dataset2`. It is intended to test whether a method can generalize from the registered development data to a setting with synthetic geometric variation.

The example below shows one correct query-target pair from `dataset2` using one representative slice from each volume:

![Dataset 2 correct pair example](assets/dataset2_example_pair.png)

### Dataset 3

`dataset3` contains preoperative-to-intraoperative MRI pairs. No labelled training pairs are provided for `dataset3`; it is intended to evaluate generalization to a more structurally different setting.

In `dataset3`, each intraoperative target image has been resampled into the same geometric space as its matching preoperative query image using the source image physical coordinates. This does not mean the images are registered in the strict sense. The candidates are intraoperative images, so the anatomy can be structurally different from the preoperative query: tissue may have shifted, parts of the brain may be missing, and local structures can look different because of the intervention. The goal is still to retrieve the matching subject, but exact local alignment is not guaranteed.

The example below shows one correct query-target pair from `dataset3` using one representative slice from each volume:

![Dataset 3 correct pair example](assets/dataset3_example_pair.png)

### Preprocessing Notes

All images have been converted to NIfTI, RAS orientation, and 1.0 x 1.0 x 1.0 mm voxel spacing. No intensity normalization, histogram matching, skull stripping, deformable registration, or cropping has been applied as part of this release.

Your code should not assume one fixed image shape for the whole challenge. Matching query and target volumes may also differ in shape, especially in `dataset2` and `dataset3`.

## Files

### Training

`dataset1/train_pairs.csv` contains labelled query-target pairs:

```text
pair_id,query_id,target_id,query_image,target_image,query_modality,target_modality,dataset
```

### Query Manifests

Validation and test query files contain:

```text
query_id,query_image,query_modality,dataset
```

### Gallery Manifests

Validation and test gallery files contain:

```text
target_id,target_image,target_modality,dataset
```

## Retrieval Pools

The three datasets are independent retrieval pools. Always rank a query only against the gallery from the same dataset and same split:

- `dataset1/val_queries.csv` uses `dataset1/val_gallery.csv`
- `dataset1/test_queries.csv` uses `dataset1/test_gallery.csv`
- `dataset2/val_queries.csv` uses `dataset2/val_gallery.csv`
- `dataset2/test_queries.csv` uses `dataset2/test_gallery.csv`
- `dataset3/val_queries.csv` uses `dataset3/val_gallery.csv`
- `dataset3/test_queries.csv` uses `dataset3/test_gallery.csv`

Do not rank queries from one dataset against another dataset's gallery, and do not mix validation and test galleries.

## Counts

```text
dataset1:
  train pairs: 350
  validation queries/gallery: 40 / 40
  test queries/gallery: 100 / 100

dataset2:
  validation queries/gallery: 40 / 40
  test queries/gallery: 100 / 100

dataset3:
  validation queries/gallery: 20 / 20
  test queries/gallery: 77 / 77
```

## Evaluation

The score is mean reciprocal rank (MRR), computed separately for `dataset1`, `dataset2`, and `dataset3`, then averaged:

```text
score = (dataset1_MRR + dataset2_MRR + dataset3_MRR) / 3
```

For each query, reciprocal rank is `1 / rank` of the true matching target in the submitted ranking. If the true target is absent, or if the query row is omitted from the submission, that query receives reciprocal rank `0`.

Kaggle uses the hidden solution file to decide which rows are public and private:

- validation query rows are scored on the public leaderboard during the competition
- test query rows are scored on the private leaderboard for final ranking

Participants do not include a split column. Kaggle aligns rows by `query_id`, then scores the public and private subsets internally.

## Submission Format

**Attention: Kaggle comes with a limitation of 100 submissions per team per day!**

Kaggle expects one submission file per attempt. Submit one combined CSV with the same columns as the root-level `sample_submission.csv`; do not submit separate files for individual datasets.

```text
query_id,target_id_ranking
q_...,g_... g_... g_... ...
```

All `query_id` and `target_id` values are globally unique across all three datasets, so the combined file can include rows from `dataset1`, `dataset2`, and `dataset3` without an extra dataset column.

Each submitted row must contain a full ranking of every target ID from that query's corresponding same-dataset, same-split gallery. Rankings are space-separated and ordered from most likely match to least likely match.

For example, a query from `dataset2/test_queries.csv` must rank all target IDs from `dataset2/test_gallery.csv`, and no target IDs from any validation gallery or from another dataset:

```text
query_id,target_id_ranking
q_example_dataset2_test,g_first_choice g_second_choice g_third_choice ...
```

Expected ranking lengths:

```text
dataset1 validation rows: 40 target IDs
dataset1 test rows: 100 target IDs
dataset2 validation rows: 40 target IDs
dataset2 test rows: 100 target IDs
dataset3 validation rows: 20 target IDs
dataset3 test rows: 77 target IDs
```

The complete submission template contains one row for every validation and test query from all three datasets, for `377` rows total. Partial submissions are allowed:

- For validation-only experiments, submit rows from `val_queries.csv` files.
- For full challenge submissions, submit both validation and test query rows in one file.
- To focus on one dataset first, submit only that dataset's rows and omit the other datasets. The omitted datasets receive zero credit. Multiplying the displayed score by `3` gives the MRR for the submitted dataset.

## Baseline Code

We provide a small MONAI + PyTorch baseline to help you get started with the challenge data format, preprocessing, training loop, and submission generation.

The baseline is intentionally simple. It is not meant to be a strong solution. Its purpose is to demonstrate:

* how to load 3D NIfTI volumes with MONAI
* how to apply basic medical image preprocessing and create a quick-to-load cache
* how to write a valid Kaggle submission file
* the task complexity of the different datasets

Install and run it with uv:

```sh
DATA_ROOT=/path/to/kaggle_dataset
uv run slice_clip_baseline.py \
  --data-root "$DATA_ROOT" \
  --train-pair-csv "$DATA_ROOT/dataset1/train_pairs.csv" \
  --query-csv "$DATA_ROOT/dataset1/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset1/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/test_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset2/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset2/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset2/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset2/test_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset3/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset3/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset3/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset3/test_gallery.csv" \
  --out slice_clip_submission.csv
```

This writes a combined submission file:

`slice_clip_submission.csv`

The file can be submitted directly to Kaggle.

You can also submit only one dataset to get intuition for the difficulty of each retrieval pool.
For example, to train on dataset1 and submit only dataset1 validation and test rows:

```sh
DATA_ROOT=/path/to/kaggle_dataset
uv run slice_cnn_baseline.py \
  --data-root "$DATA_ROOT" \
  --train-pair-csv "$DATA_ROOT/dataset1/train_pairs.csv" \
  --query-csv "$DATA_ROOT/dataset1/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset1/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/test_gallery.csv" \
  --out dataset1_submission.csv
```