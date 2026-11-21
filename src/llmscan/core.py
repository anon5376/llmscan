"""The ΔW core: weight delta, singular values, effective rank.

The pipeline in `llmscan.delta` calls `analyze_delta` once per changed 2-D tensor and
writes what it returns to `delta.csv`.

The spectrum comes from the Gram matrix of ΔW on its smaller side: its eigenvalues are the
squared singular values. That is one matrix product plus a small symmetric eigensolve, in
float64, accumulated in row blocks so a 151936 x 1024 embedding never needs a float64 copy.
ΔW is divided by its largest entry first, so neither tiny nor huge F64 deltas under- or overflow.

Only the eigenvalues are needed unless the leading mode is separated, and then only one vector.
So the eigensolve is values-only, followed by a power iteration for that vector. Values-only and
full solves differ by rounding of a few ulp of the largest eigenvalue. When the spectrum beyond
the leading mode is small enough for that to show, the full solve is used, as it always was.

Effective rank is the participation ratio of the energy spectrum, (Σσ²)² / Σσ⁴. It is 1 for a
rank-1 edit and r for r equal directions. Rounding noise from a bf16 re-save is spread thinly
over every direction and moves it very little, which a threshold count would not tolerate.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

_ROWS_PER_BLOCK = 4096
# The values-only eigensolve is used when it reproduces the full solve to this relative accuracy.
_AGREEMENT = 1e-9
_EPS = float(np.finfo(np.float64).eps)
# A reported singular value under 1e-3 of the first is an eigenvalue under 1e-6 of the largest,
# where a few ulp of the largest approach _AGREEMENT of it.
_REPORTED = 1e-6


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
    tail_norm_fraction: float | None = None
    base_energy: float | None = None


# Spacing at one and at zero for the supported checkpoint storage formats.
_SPACING = {"BF16": (2.0**-7, 2.0**-133), "F16": (2.0**-10, 2.0**-24),
            "F32": (2.0**-23, 2.0**-149), "F64": (2.0**-52, 2.0**-1074)}


def rank_one_excess(base, stats: DeltaStats, base_dtype: str, suspect_dtype: str) -> float | None:
    """Lower bound on the fraction of ΔW unexplained by one removal and a floating-point re-save.

    `base` is what `analyze_delta` was given as `w_base`, and `stats` is what it returned.
    """
    total = stats.frobenius
    if not total or not math.isfinite(total) or stats.tail_norm_fraction is None:
        return None
    # Round the denominator down when bounding the relative rounding budget.
    # At the F64 subnormal floor, even one ULP can be a large fraction of ΔW.
    total = math.nextafter(total, 0.)
    if not total or not stats.tail_norm_fraction:
        return 0.

    # A partial orthogonal removal cannot increase the base Frobenius norm.
    # Allow a full ULP at the coarser precision, including a subnormal floor.
    # Accumulate a scaled norm in blocks to avoid squaring extreme F64 values.
    relative, absolute = map(max, zip(_SPACING[base_dtype], _SPACING[suspect_dtype]))
    scale = _largest(base)
    budget = absolute * math.sqrt(base.shape[0] * base.shape[1])
    if scale:
        energy = stats.base_energy  # already summed if the projection pass read the base
        if energy is None:
            rows = base if base.shape[0] >= base.shape[1] else base.T
            energy = 0.
            for sl, block in _scaled_blocks(_row_reader(rows), rows.shape[0], scale):
                energy += float(np.einsum("ij,ij->", block, block))
        budget += (relative * math.sqrt(energy)) * scale
    return max(0., stats.tail_norm_fraction - budget / total)


def _largest(base) -> float:
    """Largest |entry|, without an abs() copy."""
    return max(float(base.max()), -float(base.min())) if isinstance(base, np.ndarray) else base.largest()


def _row_reader(matrix):
    """`reader(slice)` returns those rows of an array, or of a matrix read from its checkpoint in pieces."""
    return matrix.__getitem__ if isinstance(matrix, np.ndarray) else matrix.rows


def _delta_reader(w_base, w_suspect):
    """`reader(slice)` returns those rows of ΔW, oriented with at least as many rows as columns."""
    tall = w_suspect.shape[0] >= w_suspect.shape[1]
    if isinstance(w_suspect, np.ndarray):
        whole = w_base if isinstance(w_base, np.ndarray) else w_base.rows(slice(None))
        np.subtract(w_suspect, whole, out=w_suspect)  # ΔW, in place: the caller's copy is ours
        return _row_reader(w_suspect if tall else w_suspect.T)
    base, suspect = (_row_reader(m if tall else m.T) for m in (w_base, w_suspect))

    def rows(sl: slice) -> np.ndarray:
        out = suspect(sl)  # a fresh copy
        return np.subtract(out, base(sl), out=out)
    return rows


def _scaled_blocks(rows, count: int, scale: float):
    """Row blocks of a matrix divided by `scale`, in float64. `rows(slice)` reads the matrix.

    Each block is valid until the next one: they share one buffer. It is laid out in memory as
    the rows are, so that BLAS sums in the same order as it would over a copy of those rows.
    """
    buffer = layout = None
    for start in range(0, count, _ROWS_PER_BLOCK):
        sl = slice(start, min(start + _ROWS_PER_BLOCK, count))
        raw = rows(sl)
        fortran = abs(raw.strides[0]) < abs(raw.strides[1])
        if layout != (raw.shape, fortran):
            layout = raw.shape, fortran
            buffer = None  # free the old one first
            buffer = np.empty(raw.shape, order="F" if fortran else "C")
        np.divide(raw, scale, out=buffer, dtype=np.float64)  # divided as float64, not as float32
        raw = None  # may be a fresh copy as large as the buffer
        yield sl, buffer


def _leading_vector(gram: np.ndarray, lead: float) -> np.ndarray | None:
    """Unit eigenvector of the largest eigenvalue `lead`, which is over four times the next one.

    Power iteration: each step shrinks the error at least fourfold. None if it does not settle
    on `lead`, which a starting vector orthogonal to the answer would cause.
    """
    x = gram[int(np.argmax(gram.diagonal()))].copy()
    x /= np.linalg.norm(x)
    polish = None
    for _ in range(96):
        y = gram @ x
        quotient = float(x @ y)
        residual = float(np.linalg.norm(y - quotient * x))
        x = y / np.linalg.norm(y)
        if polish is None and residual <= 1e-12 * lead and abs(quotient - lead) <= _AGREEMENT * lead:
            polish = 8  # down to rounding error
        elif polish is not None:
            polish -= 1
            if not polish:
                return x
    return None


def _weakest_reported(gram: np.ndarray, reported: int) -> float:
    """The `reported`-th largest eigenvalue as a share of the largest, estimated from below by
    the Ritz values of two steps of subspace iteration. 1.0 if a solve would cost no more."""
    n = len(gram)
    if 4 * (reported + 8) > n:
        return 1.0
    q = np.random.default_rng(0).standard_normal((n, reported + 8))
    for _ in range(2):
        q = np.linalg.qr(gram @ q)[0]
    ritz = np.linalg.eigvalsh(q.T @ gram @ q)
    return float(ritz[-reported] / ritz[-1])


def _eigenvalues(gram: np.ndarray, reported: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Ascending eigenvalues of the Gram matrix, and its eigenvectors if the full solver ran.

    The values-only solver is several times cheaper but rounds differently, by a few ulp of the
    largest eigenvalue. It runs only where that cannot show in any reported number.
    """
    n = len(gram)
    # 1 − Σλ²/(Σλ)² is at most twice the share of energy beyond the leading mode. The second
    # eigenvalue is at least that share / n, and solver rounding is a few ulp of the first.
    trace = float(gram.trace())
    beyond = 1.0 - float(np.einsum("ij,ij->", gram, gram)) / trace ** 2
    # Estimating the weakest reported value first avoids a values-only solve that is then discarded.
    if n > 1 and beyond * _AGREEMENT >= 4 * n * _EPS and _weakest_reported(gram, reported) >= _REPORTED:
        values = np.linalg.eigvalsh(gram)
        # Neither the separation test nor a reported singular value may depend on which solver ran.
        if abs(values[-1] - 4 * values[-2]) > 1e-6 * values[-1] and values[-reported] >= _REPORTED * values[-1]:
            return values, None
    return np.linalg.eigh(gram)


def _spectrum(gram: np.ndarray, reported: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Eigenvalues of the Gram matrix, largest first and clipped at zero, and the leading
    eigenvector when the leading value is separated. The caller reports the first `reported` values."""
    values, vectors = _eigenvalues(gram, reported)
    energy = np.maximum(values[::-1], 0.0)
    if len(gram) > 1 and not energy[0] > 4 * energy[1]:
        return energy, None
    if vectors is None:
        leading = _leading_vector(gram, float(energy[0]))
        if leading is not None:
            return energy, leading
        vectors = np.linalg.eigh(gram)[1]
    return energy, vectors[:, -1].copy()


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


def analyze_delta(w_base, w_suspect, k: int) -> DeltaStats:
    """Spectrum of the change from `w_base` to `w_suspect` for one weight matrix.

    Inputs
        w_base, w_suspect: 2-D arrays of identical shape (out_features, in_features) and one
            dtype, as stored in the checkpoint: float32, or float64 when either side is F64.
            `w_suspect` is a fresh copy and is overwritten with ΔW; `w_base` is not modified.
            Either may instead be a `safetensors_io.StoredMatrix`, which is read from its
            checkpoint a row block at a time. With two of those no whole matrix is ever held,
            and the results are the same to the last bit.
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
        tail_norm_fraction: Frobenius distance to the best rank-one delta divided by the delta norm.
        base_energy: Σ (W_base / max|W_base|)² when the projection pass read the base; `rank_one_excess` reuses it.

    The pipeline never calls this for bit-identical tensors.
    """
    if w_base.ndim != 2 or w_base.shape != w_suspect.shape:
        raise ValueError(f"analyze_delta needs two 2-D arrays of one shape, got {w_base.shape} and {w_suspect.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Orient the Gram on the small side, but report the direction in the
    # original output space. The caller puts embeddings in residual x token order.
    tall = w_suspect.shape[0] >= w_suspect.shape[1]
    rows, width = w_suspect.shape if tall else w_suspect.shape[::-1]
    columns = w_suspect.shape[1]
    delta_rows = _delta_reader(w_base, w_suspect)
    del w_suspect

    scale = 0.0  # largest |ΔW| entry, without an abs() copy
    for start in range(0, rows, _ROWS_PER_BLOCK):
        block = delta_rows(slice(start, start + _ROWS_PER_BLOCK))
        if not np.isfinite(block).all():
            return DeltaStats(np.full(min(k, width), np.nan), float("nan"))
        scale = max(scale, float(block.max()), -float(block.min()))
    del block
    if scale == 0.0:
        return DeltaStats(np.zeros(min(k, width)), 0.0)

    gram = product = None
    column_energy = np.empty(columns)
    for sl, block in _scaled_blocks(delta_rows, rows, scale):  # entries in [-1, 1]
        if gram is None:
            gram = np.matmul(block.T, block)
            gram += 0.0  # as if added to zeros: no negative zero
        else:
            product = np.matmul(block.T, block, out=product)
            gram += product
        if not tall:
            column_energy[sl] = np.einsum("ij,ij->i", block, block)
    if tall:
        column_energy = gram.diagonal()
    fraction = float(column_energy.sum() ** 2 / (len(column_energy) * (column_energy @ column_energy)))
    del block, product, column_energy  # the eigensolve is the memory peak

    energy, small = _spectrum(gram, min(k, width))
    del gram
    total = energy.sum()
    effective_rank = float(total ** 2 / (energy @ energy))
    tail = energy[1:].sum()
    # Eckart–Young, evaluated before restoring physical units: separate F64
    # subnormal norms can round differently and distort a ratio of the two.
    tail_norm_fraction = float(np.sqrt(tail / total))
    second_share = float(energy[1] / tail) if tail else 0.0

    # A repeated or nearly repeated leading singular value has no stable axis.
    direction = gain = projection_error = column_spike = base_energy = None
    if small is not None:
        other = np.empty(rows)
        direction, right = (other, small) if tall else (small, other)
        base_scale = _largest(w_base)
        projected = np.zeros(columns)
        for sl, block in _scaled_blocks(delta_rows, rows, scale):
            other[sl] = (block @ small) / np.sqrt(energy[0])
        del block
        if base_scale:
            base_energy = 0.
            for sl, block in _scaled_blocks(_row_reader(w_base if tall else w_base.T), rows, base_scale):
                if tall:
                    projected += block.T @ direction[sl]
                else:
                    projected[sl] = block @ direction
                base_energy += float(np.einsum("ij,ij->", block, block))
        # For the orthogonal projection W' = W - uuᵀW this gain equals one,
        # regardless of how much weight this layer has in direction u.
        projected_norm = np.linalg.norm(projected)
        if projected_norm:
            gain = float((np.sqrt(energy[0]) / projected_norm) * (scale / base_scale))
            projected /= projected_norm
        projection_error, column_spike = _projection_fit(right, projected, gain)

    return DeltaStats(np.sqrt(energy[:k]) * scale, effective_rank, float(np.sqrt(total) * scale),
                      fraction, second_share, direction, gain, projection_error, column_spike, tail_norm_fraction,
                      base_energy)
