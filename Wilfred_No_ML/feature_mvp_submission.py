from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


DATASETS = ("dataset1", "dataset2", "dataset3")
SPLITS = ("val", "test")


@dataclass(frozen=True)
class ImageFeature:
    image_id: str
    path: Path
    dataset: str
    meta: np.ndarray
    mask16: np.ndarray
    image16: np.ndarray
    proj: np.ndarray


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    candidates = [
        data_root / image_path,
        data_root / image_path.replace(".nii.gz", ".nii"),
        data_root / Path(image_path).name,
        data_root / Path(image_path.replace(".nii.gz", ".nii")).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resize_nearest(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if array.size == 0:
        return np.zeros(shape, dtype=np.float32)
    indices = [np.linspace(0, array.shape[axis] - 1, shape[axis]).round().astype(np.int64) for axis in range(array.ndim)]
    return array[np.ix_(*indices)].astype(np.float32, copy=False)


def robust_scale(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = volume[mask]
    if values.size < 16:
        values = volume[np.isfinite(volume)]
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mean = float(np.mean(values))
        std = float(np.std(values)) + 1e-6
        return np.clip((volume - mean) / (4.0 * std) + 0.5, 0.0, 1.0).astype(np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def foreground_mask(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    nonzero = finite & (np.abs(volume) > 1e-6)
    if np.count_nonzero(nonzero) > 128:
        values = np.abs(volume[nonzero])
        threshold = max(1e-6, float(np.percentile(values, 5.0)))
        mask = finite & (np.abs(volume) >= threshold)
        if np.count_nonzero(mask) > 128:
            return mask
    return nonzero if np.count_nonzero(nonzero) else finite


def bbox_features(mask: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    coords = np.argwhere(mask)
    shape = np.array(mask.shape, dtype=np.float32)
    if coords.size == 0:
        meta = np.zeros(10, dtype=np.float32)
        return meta, tuple(slice(0, int(size)) for size in mask.shape)  # type: ignore[return-value]

    mins = coords.min(axis=0).astype(np.float32)
    maxs = coords.max(axis=0).astype(np.float32)
    center = (mins + maxs) / 2.0
    extent = np.maximum(maxs - mins + 1.0, 1.0)
    com = coords.mean(axis=0).astype(np.float32)
    denom = np.maximum(shape - 1.0, 1.0)
    fg_frac = np.array([coords.shape[0] / max(float(np.prod(shape)), 1.0)], dtype=np.float32)
    meta = np.concatenate([center / denom, extent / np.maximum(shape, 1.0), com / denom, fg_frac]).astype(np.float32)
    crop = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(mins, maxs))
    return meta, crop  # type: ignore[return-value]


def projection_features(volume: np.ndarray) -> np.ndarray:
    projections = []
    for axis in range(3):
        proj = volume.mean(axis=axis)
        projections.append(resize_nearest(proj, (16, 16)).reshape(-1))
    return np.concatenate(projections).astype(np.float32)


def extract_feature(image_id: str, path: Path, dataset: str) -> ImageFeature:
    img = nib.load(str(path))
    volume = np.asanyarray(img.dataobj, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = foreground_mask(volume)
    bbox_meta, crop = bbox_features(mask)
    scaled = robust_scale(volume, mask)

    cropped_scaled = scaled[crop]
    cropped_mask = mask[crop].astype(np.float32)
    image16 = resize_nearest(cropped_scaled, (16, 16, 16)).reshape(-1)
    mask16 = resize_nearest(cropped_mask, (16, 16, 16)).reshape(-1)
    proj = projection_features(cropped_scaled)

    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    shape = np.array(volume.shape[:3], dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)
    affine_meta = np.concatenate(
        [
            np.log1p(shape) / 8.0,
            np.log1p(np.maximum(zooms, 1e-6)),
            affine[:3, :3].reshape(-1) / 256.0,
            affine[:3, 3] / 256.0,
        ]
    )
    meta = np.concatenate([affine_meta, bbox_meta]).astype(np.float32)
    return ImageFeature(image_id, path, dataset, meta, mask16.astype(np.float32), image16.astype(np.float32), proj)


def cosine_distance_matrix(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    query = query.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)
    query = query - query.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-6
    target /= np.linalg.norm(target, axis=1, keepdims=True) + 1e-6
    return 1.0 - query @ target.T


def standardized_l2_matrix(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    combined = np.vstack([query, target]).astype(np.float32, copy=False)
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True) + 1e-6
    q = (query - mean) / std
    t = (target - mean) / std
    return ((q[:, None, :] - t[None, :, :]) ** 2).mean(axis=2)


def score_matrix(
    query_features: list[ImageFeature],
    target_features: list[ImageFeature],
    weights: tuple[float, float, float, float],
) -> np.ndarray:
    q_image = np.stack([feature.image16 for feature in query_features])
    t_image = np.stack([feature.image16 for feature in target_features])
    q_mask = np.stack([feature.mask16 for feature in query_features])
    t_mask = np.stack([feature.mask16 for feature in target_features])
    q_proj = np.stack([feature.proj for feature in query_features])
    t_proj = np.stack([feature.proj for feature in target_features])
    q_meta = np.stack([feature.meta for feature in query_features])
    t_meta = np.stack([feature.meta for feature in target_features])

    image_w, mask_w, proj_w, meta_w = weights
    return (
        image_w * cosine_distance_matrix(q_image, t_image)
        + mask_w * cosine_distance_matrix(q_mask, t_mask)
        + proj_w * cosine_distance_matrix(q_proj, t_proj)
        + meta_w * standardized_l2_matrix(q_meta, t_meta)
    )


def rank_pool(
    query_features: list[ImageFeature],
    target_features: list[ImageFeature],
    weights: tuple[float, float, float, float],
) -> list[dict[str, str]]:
    distances = score_matrix(query_features, target_features, weights)
    target_ids = [feature.image_id for feature in target_features]
    rows = []
    for row_index, query_feature in enumerate(query_features):
        order = np.argsort(distances[row_index], kind="mergesort")
        rows.append({"query_id": query_feature.image_id, "target_id_ranking": " ".join(target_ids[index] for index in order)})
    return rows


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_features(data_root: Path, csv_path: Path, id_column: str, image_column: str) -> list[ImageFeature]:
    rows = read_csv(csv_path)
    dataset = csv_path.parts[-2] if len(csv_path.parts) >= 2 else "unknown"
    features = []
    for row in rows:
        image_id = row[id_column]
        path = resolve_image_path(data_root, row[image_column])
        features.append(extract_feature(image_id, path, dataset))
    return features


def load_train_features(data_root: Path) -> tuple[list[ImageFeature], list[ImageFeature], dict[str, str]]:
    train_pairs_path = data_root / "dataset1" / "train_pairs.csv"
    pairs = read_csv(train_pairs_path)
    query_features = []
    target_features = []
    truth = {}
    for row in pairs:
        query_id = row["query_id"]
        target_id = row["target_id"]
        truth[query_id] = target_id
        query_features.append(extract_feature(query_id, resolve_image_path(data_root, row["query_image"]), "dataset1"))
        target_features.append(extract_feature(target_id, resolve_image_path(data_root, row["target_image"]), "dataset1"))
    return query_features, target_features, truth


def mrr_for_weights(
    query_features: list[ImageFeature],
    target_features: list[ImageFeature],
    truth: dict[str, str],
    weights: tuple[float, float, float, float],
) -> float:
    distances = score_matrix(query_features, target_features, weights)
    target_ids = [feature.image_id for feature in target_features]
    reciprocal_ranks = []
    for row_index, query_feature in enumerate(query_features):
        expected = truth[query_feature.image_id]
        order = np.argsort(distances[row_index], kind="mergesort")
        rank = int(np.where(np.array(target_ids, dtype=object)[order] == expected)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
    return float(np.mean(reciprocal_ranks))


def tune_dataset1_weights(data_root: Path) -> tuple[float, float, float, float]:
    query_features, target_features, truth = load_train_features(data_root)
    candidates = [
        (0.70, 0.10, 0.15, 0.05),
        (0.55, 0.20, 0.20, 0.05),
        (0.40, 0.30, 0.20, 0.10),
        (0.30, 0.40, 0.15, 0.15),
        (0.20, 0.35, 0.20, 0.25),
        (0.10, 0.30, 0.20, 0.40),
    ]
    scores = [{"weights": weights, "train_mrr": mrr_for_weights(query_features, target_features, truth, weights)} for weights in candidates]
    scores.sort(key=lambda item: item["train_mrr"], reverse=True)
    print(json.dumps({"dataset1_train_weight_search": scores[:6]}, indent=2))
    return scores[0]["weights"]  # type: ignore[return-value]


def default_weights(dataset: str, dataset1_weights: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if dataset == "dataset1":
        return dataset1_weights
    if dataset == "dataset2":
        return (0.30, 0.35, 0.20, 0.15)
    return (0.20, 0.35, 0.20, 0.25)


def build_submission(data_root: Path, out: Path, tune: bool) -> None:
    dataset1_weights = tune_dataset1_weights(data_root) if tune else (0.55, 0.20, 0.20, 0.05)
    submission_rows: list[dict[str, str]] = []
    summary = []
    for dataset in DATASETS:
        for split in SPLITS:
            query_csv = data_root / dataset / f"{split}_queries.csv"
            gallery_csv = data_root / dataset / f"{split}_gallery.csv"
            query_features = load_manifest_features(data_root, query_csv, "query_id", "query_image")
            target_features = load_manifest_features(data_root, gallery_csv, "target_id", "target_image")
            weights = default_weights(dataset, dataset1_weights)
            submission_rows.extend(rank_pool(query_features, target_features, weights))
            summary.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "queries": len(query_features),
                    "targets": len(target_features),
                    "weights": weights,
                }
            )
    write_submission(out, submission_rows)
    print(json.dumps({"wrote": str(out), "rows": len(submission_rows), "summary": summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast non-DL MRI retrieval MVP using header, mask, and downsampled volume features.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("feature_mvp_submission.csv"))
    parser.add_argument("--no-tune", action="store_true", help="Skip dataset1 train-pair weight search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_submission(args.data_root.resolve(), args.out, tune=not args.no_tune)


if __name__ == "__main__":
    main()
