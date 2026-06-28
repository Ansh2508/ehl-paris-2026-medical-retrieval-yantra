"""Component E: deterministic referee. Prefer robust shape/size checks + agreement
rerank. Registration veto is OPTIONAL and must be oracle-gated on simulated L3.
STUB — keep every rule oracle-gated (only ship if it raises oracle MRR)."""
from __future__ import annotations
from typing import Sequence


def shape_consistency_filter(query_meta: dict, gallery_meta: dict,
                             ranking: Sequence[str]) -> list[str]:
    """Demote candidates whose brain-volume ratio / FOV is grossly incompatible with
    the query. A rule that CANNOT backfire (unlike registration). TODO(claude-code)."""
    raise NotImplementedError("TODO(claude-code): coarse volume/FOV consistency")


def agreement_rerank(ranking_a: Sequence[str], ranking_b: Sequence[str], topk: int) -> list[str]:
    """Boost candidates that BOTH branches rank high (consistency = confidence)."""
    raise NotImplementedError("TODO(claude-code): intersect top-k, promote agreed candidates")


def registration_veto(query_vol, gallery, ranking, enabled: bool = False) -> list[str]:
    """OPTIONAL. If A's top pick fails an affine-registration residual check, override.
    RISKY on L3 (surgery breaks correspondence) -> only enable if simulated-L3 oracle improves."""
    if not enabled:
        return list(ranking)
    raise NotImplementedError("TODO(claude-code): affine registration residual gate (guarded)")
