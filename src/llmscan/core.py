"""The ΔW core: weight delta, singular values, effective rank.

The pipeline in `llmscan.delta` calls `analyze_delta` once per changed 2-D tensor and
writes what it returns to `delta.csv`.

The spectrum comes from the Gram matrix of ΔW on its smaller side: its eigenvalues are the
squared singular values. That is one matrix product plus a small symmetric eigensolve, in
float64, accumulated in row blocks so a 151936 x 1024 embedding never needs a float64 copy.

Effective rank is the participation ratio of the energy spectrum, (Σσ²)² / Σσ⁴. It is 1 for a
rank-1 edit and r for r equal directions. Rounding noise from a bf16 re-save is spread thinly
over every direction and moves it very little, which a threshold count would not tolerate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_ROWS_PER_BLOCK = 4096


class DeltaStats(NamedTuple):
    singular_values: np.ndarray
    effective_rank: float


def analyze_delta(w_base: np.ndarray, w_suspect: np.ndarray, k: int) -> DeltaStats:
    """Spectrum of the change from `w_base` to `w_suspect` for one weight matrix.

    Inputs
        w_base, w_suspect: 2-D float32 arrays of identical shape (out_features, in_features),
            as stored in the checkpoint. Both are fresh copies; modifying them is fine.
        k: how many leading singular values to return, k >= 1.

    Returns DeltaStats
        singular_values: 1-D float64 array of the k largest singular values of
            ΔW = w_suspect − w_base, largest first. Fewer than k if min(shape) < k.
        effective_rank: participation ratio (Σσ²)² / Σσ⁴, 0.0 when ΔW is zero. It is NaN
            when ΔW holds a non-finite value; the ranking treats that tensor as unusable.

    The pipeline never calls this for bit-identical tensors.
    """
    if w_base.ndim != 2 or w_base.shape != w_suspect.shape:
        raise ValueError(f"analyze_delta needs two 2-D arrays of one shape, got {w_base.shape} and {w_suspect.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    np.subtract(w_suspect, w_base, out=w_suspect)  # ΔW, in place: the caller's copy is ours
    dw = w_suspect if w_suspect.shape[0] >= w_suspect.shape[1] else w_suspect.T
    if not np.isfinite(dw).all():
        return DeltaStats(np.full(min(k, dw.shape[1]), np.nan), float("nan"))

    gram = np.zeros((dw.shape[1], dw.shape[1]))
    for start in range(0, dw.shape[0], _ROWS_PER_BLOCK):
        block = dw[start:start + _ROWS_PER_BLOCK].astype(np.float64)
        gram += block.T @ block
    energy = np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None)  # σ², largest first

    total = float(energy.sum())
    effective_rank = total * total / float(np.square(energy).sum()) if total > 0 else 0.0
    return DeltaStats(np.sqrt(energy[:k]), effective_rank)
