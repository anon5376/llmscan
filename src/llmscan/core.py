"""The ΔW core: weight delta, singular values, effective rank.

The pipeline in `llmscan.delta` calls `analyze_delta` once per changed 2-D tensor and
writes what it returns to `delta.csv`.

The spectrum comes from the Gram matrix of ΔW on its smaller side: its eigenvalues are the
squared singular values. That is one matrix product plus a small symmetric eigensolve, in
float64, accumulated in row blocks so a 151936 x 1024 embedding never needs a float64 copy.
ΔW is divided by its largest entry first, so neither tiny nor huge F64 deltas under- or overflow.

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
    frobenius: float | None = None
    column_fraction: float | None = None
    second_tail_share: float | None = None
    direction: np.ndarray | None = None
    projection_gain: float | None = None
    projection_error: float | None = None
    column_spike: float | None = None


def _projection_fit(right: np.ndarray, projected: np.ndarray, gain: float | None) -> tuple[float, float]:
    """Fit the signed coefficients of a partial removal, in units of σ₁."""
    residual = right.copy()
    if gain is not None and gain > 0:
        # A partial or full removal lies on the segment from zero to
        # -Wᵀu₁/σ₁. Here projected is its unit base direction.
        length = np.clip(-(right @ projected), 0.0, 1.0 / gain)
        residual += length * projected
    error = min(1.0, float(np.linalg.norm(residual)))
    # Large coefficients already explained by the base row have zero error.
    # Use both the RMS of a unit coefficient vector and a vocabulary-size
    # adjusted Gaussian maximum, with MAD estimating the background scale.
    deviations = np.abs(residual - np.median(residual))
    n = len(right)
    robust_sigma = 1.4826 * np.median(deviations)
    bound = max(1 / np.sqrt(n), robust_sigma * np.sqrt(2 * np.log(2 * n / 0.01)))
    return error, float(deviations.max() / bound)


def analyze_delta(w_base: np.ndarray, w_suspect: np.ndarray, k: int) -> DeltaStats:
    """Spectrum of the change from `w_base` to `w_suspect` for one weight matrix.

    Inputs
        w_base, w_suspect: 2-D arrays of identical shape (out_features, in_features) and one
            dtype, as stored in the checkpoint: float32, or float64 when either side is F64.
            Both are fresh copies; modifying them is fine.
        k: how many leading singular values to return, k >= 1.

    Returns DeltaStats
        singular_values: 1-D float64 array of the k largest singular values of
            ΔW = w_suspect − w_base, largest first. Fewer than k if min(shape) < k.
        effective_rank: participation ratio (Σσ²)² / Σσ⁴, 0.0 when ΔW is zero. It is NaN
            when ΔW holds a non-finite value; the ranking treats that tensor as unusable.
        frobenius: norm of the full delta, including modes beyond k.
        column_fraction: effective number of energized columns divided by the column count.
        second_tail_share: fraction of non-leading energy in the second singular mode.
        direction: leading left singular vector when the leading mode is separated.
        projection_gain: σ₁ / ‖u₁ᵀ W_base‖₂ when the denominator is nonzero.
        projection_error: relative error of the best partial removal along u₁.
        column_spike: largest centered coefficient error relative to its robust noise bound.

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

    scale = max(float(dw.max()), -float(dw.min()))  # largest |ΔW| entry, without an abs() copy
    if scale == 0.0:
        return DeltaStats(np.zeros(min(k, dw.shape[1])), 0.0)

    # Orient the Gram on the small side, but report the direction in the
    # original output space. The caller puts embeddings in residual x token order.
    tall = w_suspect.shape[0] >= w_suspect.shape[1]
    gram = np.zeros((dw.shape[1], dw.shape[1]))
    column_energy = np.empty(w_suspect.shape[1])
    for start in range(0, dw.shape[0], _ROWS_PER_BLOCK):
        block = dw[start:start + _ROWS_PER_BLOCK].astype(np.float64) / scale  # entries in [-1, 1]
        gram += block.T @ block
        if not tall:
            column_energy[start:start + len(block)] = np.einsum("ij,ij->i", block, block)
    if tall:
        column_energy = gram.diagonal()
    values, vectors = np.linalg.eigh(gram)
    energy = np.maximum(values[::-1], 0.0)
    total = energy.sum()
    effective_rank = float(total ** 2 / (energy @ energy))
    fraction = float(column_energy.sum() ** 2 / (len(column_energy) * (column_energy @ column_energy)))
    tail = energy[1:].sum()
    second_share = float(energy[1] / tail) if tail else 0.0

    # A repeated or nearly repeated leading singular value has no stable axis.
    direction = gain = projection_error = column_spike = None
    if len(energy) == 1 or energy[0] > 4 * energy[1]:
        small = vectors[:, -1].copy()
        other = np.empty(dw.shape[0])
        direction, right = (other, small) if tall else (small, other)
        base = w_base if tall else w_base.T
        base_scale = max(float(w_base.max()), -float(w_base.min()))
        projected = np.zeros(w_suspect.shape[1])
        for start in range(0, dw.shape[0], _ROWS_PER_BLOCK):
            sl = slice(start, start + _ROWS_PER_BLOCK)
            other[sl] = ((dw[sl].astype(np.float64) / scale) @ small) / np.sqrt(energy[0])
            if base_scale:
                block = base[sl].astype(np.float64) / base_scale
                if tall:
                    projected += block.T @ direction[sl]
                else:
                    projected[sl] = block @ direction
        # For the orthogonal projection W' = W - uuᵀW this gain equals one,
        # regardless of how much weight this layer has in direction u.
        projected_norm = np.linalg.norm(projected)
        if projected_norm:
            gain = float((np.sqrt(energy[0]) / projected_norm) * (scale / base_scale))
            projected /= projected_norm
        projection_error, column_spike = _projection_fit(right, projected, gain)

    return DeltaStats(np.sqrt(energy[:k]) * scale, effective_rank, float(np.sqrt(total) * scale),
                      fraction, second_share, direction, gain, projection_error, column_spike)
