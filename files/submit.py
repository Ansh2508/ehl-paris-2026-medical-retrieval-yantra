"""Write the exact Kaggle CSV. Format-critical — fully implemented."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Callable, Sequence

# A scorer ranks a gallery for one query (see Design.md):
Scorer = Callable[[str, Sequence[str]], list[str]]

EXPECTED_ROWS = 377  # all val+test queries across the 3 datasets


def build(prediction_sets: list[dict], scorer: Scorer) -> list[tuple[str, list[str]]]:
    """prediction_sets: [{"queries": [qid...], "gallery": [tid...]}].
    Returns rows = [(query_id, full_ranking_of_that_gallery)]."""
    rows: list[tuple[str, list[str]]] = []
    for ps in prediction_sets:
        gallery = list(ps["gallery"])
        for qid in ps["queries"]:
            ranking = list(scorer(qid, gallery))
            assert sorted(ranking) == sorted(gallery), f"{qid}: ranking must permute its gallery"
            rows.append((qid, ranking))
    return rows


def write_submission(rows: list[tuple[str, list[str]]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "target_id_ranking"])
        for qid, ranking in rows:
            w.writerow([qid, " ".join(ranking)])


def validate(rows: list[tuple[str, list[str]]], strict: bool = False) -> None:
    """Sanity checks before submitting."""
    if strict and len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    for qid, ranking in rows:
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"{qid}: duplicate target ids in ranking")
