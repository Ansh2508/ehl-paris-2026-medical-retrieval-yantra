"""Embed gallery+query, optional TTA, rank by cosine. STUB (+ contracts)."""
from __future__ import annotations
import numpy as np
import torch
from typing import Sequence
from config import CFG


def embed_images(encoder, dataset, tta_n: int = CFG.tta_n) -> dict[str, np.ndarray]:
    """Return {id -> embedding}. If tta_n>1, embed tta_n augmented views and average
    (then re-normalize). TTA is ~free on 192GB and nudges rank-1."""
    raise NotImplementedError("TODO(claude-code): batched encode (+TTA averaging)")


def rank_by_embedding(q_emb: np.ndarray, gallery: Sequence[tuple[str, np.ndarray]]) -> list[str]:
    """Rank gallery ids by cosine similarity to q_emb, best->worst."""
    ids = [g[0] for g in gallery]
    mat = np.stack([g[1] for g in gallery])
    scores = mat @ q_emb
    return [ids[i] for i in np.argsort(-scores)]
