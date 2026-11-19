"""Leading residual-stream singular direction of a weight change and its agreement with peers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import isfinite, nan, sqrt
from typing import NamedTuple

import numpy as np

from llmscan.rank import canonical_kind

_ROWS_PER_BLOCK = 256


class Direction(NamedTuple):
    vector: np.ndarray | None
    energy_share: float
    delta_norm: float  # Frobenius norm before scaling
    reason: str = ""


def residual_axis(kind: str) -> int | None:
    """Axis of the residual stream in the reader's [out, in] or [vocabulary, embedding] layout."""
    kind = canonical_kind(kind)
    if kind in ("o_proj", "down_proj", "attn_output", "ffn_down"):
        return 0
    if kind in ("embed_tokens", "token_embd"):
        return 1
    return None


def analyze_direction(w_base: np.ndarray, w_suspect: np.ndarray, kind: str) -> Direction:
    """Analyze two same-shape floating matrices without modifying them. An absent vector is unusable."""
    if w_base.ndim != 2 or w_base.shape != w_suspect.shape:
        raise ValueError("analyze_direction needs two 2-D arrays of the same shape")
    if w_base.dtype.kind != "f" or w_suspect.dtype.kind != "f":
        raise ValueError("analyze_direction needs floating weights")
    axis = residual_axis(kind)
    if axis is None:
        return Direction(None, nan, nan, "unknown residual side")
    if w_base.size == 0:
        return Direction(None, nan, 0.0, "zero delta")

    # Work on the smaller Gram even when the residual axis is the larger one.
    gram_axis = 1 if w_base.shape[1] <= w_base.shape[0] else 0
    base, suspect = (w_base, w_suspect) if gram_axis == 1 else (w_base.T, w_suspect.T)

    def blocks():
        for start in range(0, len(base), _ROWS_PER_BLOCK):
            rows = slice(start, start + _ROWS_PER_BLOCK)
            yield rows, np.subtract(suspect[rows], base[rows], dtype=np.float64)

    gram = np.zeros((base.shape[1], base.shape[1]), dtype=np.float64)
    scale = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _, block in blocks():
            peak = max(-float(block.min()), float(block.max()))
            if not isfinite(peak):
                return Direction(None, nan, nan, "non-finite delta")
            if peak == 0.0:
                continue
            if peak > scale:
                gram *= (scale / peak) ** 2  # express previous blocks in the new scale
                scale = peak
            block /= scale
            gram += block.T @ block
    if scale == 0.0:
        return Direction(None, nan, 0.0, "zero delta")
    del block
    energy = float(gram.trace())
    values, eigenvectors = np.linalg.eigh(gram)
    top = float(values[-1])
    vector = eigenvectors[:, -1].copy()  # retain one vector after the pair is released
    del gram, eigenvectors
    if axis != gram_axis:
        # Recover the other singular vector as (ΔW/scale) v / (σ/scale), in the same row blocks.
        lifted = np.empty(len(base), dtype=np.float64)
        for rows, block in blocks():
            block /= scale
            lifted[rows] = block @ vector
        vector = lifted / sqrt(top)
    return Direction(vector, min(1.0, top / energy), scale * sqrt(energy))


def alignment_scores(vectors: Sequence[np.ndarray | None]) -> np.ndarray:
    """Leave-one-out alignment of unit residual vectors, pooled by width. Missing or lone vectors score NaN.

    A peer can represent the target only as well as it matches both the target and half the remaining
    peers (counting the representative once). Take the best such peer. All cosines are absolute.
    """
    groups = defaultdict(list)
    for i, vector in enumerate(vectors):
        if vector is not None:
            groups[len(vector)].append(i)

    scores = np.full(len(vectors), np.nan)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        unit = np.stack([vectors[i] for i in indices])
        cosine = np.clip(np.abs(unit @ unit.T), 0.0, 1.0)
        # Removing one vote moves the upper median to one of two adjacent order statistics.
        k = (len(indices) - 1) // 2
        middle = np.partition(cosine, (k, k + 1), axis=1)[:, k:k + 2]
        for target, index in enumerate(indices):
            vote = cosine[:, target]
            support = np.where(vote <= middle[:, 0], middle[:, 1], middle[:, 0])
            agreement = np.minimum(vote, support)
            agreement[target] = 0.0  # the target cannot represent itself
            scores[index] = agreement.max()
    return scores
