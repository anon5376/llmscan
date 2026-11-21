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

import math
from typing import NamedTuple

import numpy as np

from .interval_fit import fit_excess
from .removal import joint_excess, promising_symmetric_response, skew_probe

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
    tail_norm_fraction: float | None = None
    fit_residual: np.ndarray | None = None
    base_probe: np.ndarray | None = None
    output_subspace: np.ndarray | None = None

    def tail_excess(self, rounding_budget: float) -> float | None:
        total = self.frobenius
        if not total or not math.isfinite(total) or self.tail_norm_fraction is None:
            return None
        # Round down at the F64 subnormal floor, where one ULP can be a
        # large fraction of the reported physical norm.
        total = math.nextafter(total, 0.)
        if not total or not self.tail_norm_fraction:
            return 0.
        return max(0., self.tail_norm_fraction - rounding_budget / total)


# Spacing at one and at zero for the supported checkpoint storage formats.
_SPACING = {"BF16": (2.0**-7, 2.0**-133), "F16": (2.0**-10, 2.0**-24),
            "F32": (2.0**-23, 2.0**-149), "F64": (2.0**-52, 2.0**-1074)}


def rank_one_excess(base: np.ndarray, stats: DeltaStats, base_dtype: str, suspect_dtype: str) -> float | None:
    """Lower bound on the fraction of ΔW unexplained by one removal and a floating-point re-save."""
    return stats.tail_excess(rounding_budget(base, base_dtype, suspect_dtype))


def rounding_budget(base: np.ndarray, base_dtype: str, suspect_dtype: str) -> float:
    """Frobenius allowance for a partial removal and floating-point re-save."""
    # A partial orthogonal removal cannot increase the base Frobenius norm.
    # Allow a full ULP at the coarser precision, including a subnormal floor.
    # Accumulate a scaled norm in blocks to avoid squaring extreme F64 values.
    relative, absolute = map(max, zip(_SPACING[base_dtype], _SPACING[suspect_dtype]))
    scale = max(float(base.max()), -float(base.min()))
    if not math.isfinite(scale):
        return math.inf
    budget = absolute * math.sqrt(base.size)
    if scale:
        rows = base if base.shape[0] >= base.shape[1] else base.T
        energy = 0.
        for start in range(0, len(rows), _ROWS_PER_BLOCK):
            block = rows[start:start + _ROWS_PER_BLOCK].astype(np.float64) / scale
            energy += float(np.einsum("ij,ij->", block, block))
        budget += (relative * math.sqrt(energy)) * scale
    return budget


def contraction_excess(base: np.ndarray, delta: np.ndarray, stats: DeltaStats,
                       base_dtype: str, suspect_dtype: str) -> float | None:
    """Test one input direction against the contraction 2W_suspect − W_base.

    For a partial removal P = αuuᵀ, I − 2P has operator norm at most one.
    For any unit z, (‖(B + 2Δ)z‖ − ‖Bz‖)/2 thus lower bounds the norm of
    the error on this input, even if the estimated removal axis was wrong.
    """
    residual, total = stats.fit_residual, stats.frobenius
    if residual is None or not total or not math.isfinite(total):
        return None
    length = np.linalg.norm(residual)
    if not length:
        return 0.
    z = residual / length
    abs_z = np.abs(z)
    scale = max(total, float(base.max()), -float(base.min()))
    base_norm = reflected_norm = envelope_norm = 0.
    # Bound temporary memory for wide matrices, especially residual × vocabulary.
    block_rows = max(1, _ROWS_PER_BLOCK * min(base.shape) // base.shape[1])
    for start in range(0, len(base), block_rows):
        sl = slice(start, start + block_rows)
        b = base[sl].astype(np.float64) / scale
        d = delta[sl].astype(np.float64) / scale
        x = b @ z
        reflected = x + 2 * (d @ z)
        envelope = (np.abs(b) + np.abs(d)) @ abs_z
        base_norm = math.hypot(base_norm, *x)
        reflected_norm = math.hypot(reflected_norm, *reflected)
        envelope_norm = math.hypot(envelope_norm, *envelope)

    # Triangle inequality charges rounding of *both* checkpoints. One ULP
    # at the coarser precision covers their two half-ULPs. Charge the
    # subtraction at its working precision, then the scaled float64 products
    # and summations. |B| + |Δ| bounds the entrywise magnitude of B + Δ.
    relative, absolute = map(max, zip(_SPACING[base_dtype], _SPACING[suspect_dtype]))
    arithmetic = float(np.finfo(delta.dtype).eps) + 4 * math.ulp(1.) * (sum(base.shape) + 2)
    budget = (relative + arithmetic) * envelope_norm
    budget += 2 * (absolute / scale) * math.sqrt(len(base)) * abs_z.sum()
    growth = (reflected_norm - base_norm) / 2
    if growth <= budget:
        return 0.
    denominator = total / scale + math.ulp(total) / scale
    return min(1., (growth - budget) / denominator) if denominator else None


def _skew_columns(base: np.ndarray, delta: np.ndarray, scale: float) -> tuple[int, int]:
    """Select two coordinates of the skew matrix K = BᵀΔ − ΔᵀB without forming K."""
    # A dense witness samples all modes, including ties. Select a coordinate
    # from Kz, then its strongest partner from that row of K. Only the final
    # two-coordinate discrepancy enters the test and its rounding allowance.
    z = np.random.default_rng(0).standard_normal(base.shape[1])
    z /= np.linalg.norm(z)
    block_rows = max(1, _ROWS_PER_BLOCK * min(base.shape) // base.shape[1])

    def action(column: int | None) -> np.ndarray:
        result = np.zeros(base.shape[1])
        for start in range(0, len(base), block_rows):
            sl = slice(start, start + block_rows)
            b = base[sl].astype(np.float64) / scale
            d = delta[sl].astype(np.float64) / scale
            bx, dx = (b @ z, d @ z) if column is None else (b[:, column], d[:, column])
            result += b.T @ dx - d.T @ bx
        return result

    i = int(np.argmax(np.abs(action(None))))
    return i, int(np.argmax(np.abs(action(i))))


def _fit_storage_box(weights, dtype, exponent, arithmetic):
    """Enclose the nearest-rounding cell of each power-of-two scaled weight."""
    relative, absolute = _SPACING[dtype]
    mantissa, power = np.frexp(weights)
    floor = math.ldexp(absolute, -exponent - 1)
    away = np.maximum(floor, np.ldexp(relative / 4, power))
    away[weights == 0] = floor
    # At a binade boundary the neighbor toward zero is twice as close,
    # except where both neighbors are on the uniform subnormal grid.
    toward = np.where(np.abs(mantissa) == 0.5, np.maximum(floor, away / 2), away)
    offset = (away - toward) / 2
    center = weights + np.copysign(offset, weights)
    radius = away - offset + arithmetic * np.abs(weights) + np.finfo(float).tiny
    return center, radius


def removal_excess(base: np.ndarray, suspect: np.ndarray, base_dtype: str, suspect_dtype: str) -> float | None:
    """Test the symmetric negative semidefinite form SᵀΔ of a partial removal.

    For S = (I − P)B, 0 ≼ P ≼ I, (Sx)ᵀΔx = −(Bx)ᵀ(I − P)P(Bx) ≤ 0.
    Return the fraction of a witnessed violation left after bounding rounding
    of both stored checkpoints, on a single input or jointly across inputs.
    """
    extrema = [base.min(), base.max(), suspect.min(), suspect.max()]
    if not np.isfinite(extrema).all():
        return None
    b_step = _SPACING[base_dtype][1]
    s_step = _SPACING[suspect_dtype][1]
    # A common power of two preserves the sign, including for subnormal
    # checkpoints. Include the storage floors before rescaling mixed dtypes.
    _, exponent = math.frexp(max(b_step, s_step, *(abs(float(v)) for v in extrema)))
    rows, cols = base.shape
    block_rows = max(1, _ROWS_PER_BLOCK * min(rows, cols) // cols)

    def blocks():
        for start in range(0, rows, block_rows):
            sl = slice(start, start + block_rows)
            yield (np.ldexp(base[sl], -exponent, dtype=np.float64),
                   np.ldexp(suspect[sl], -exponent, dtype=np.float64))

    def range_basis(probes):
        action = np.zeros_like(probes)
        for b, s in blocks():
            d = s - b
            action += s.T @ (d @ probes) + d.T @ (s @ probes)
        return np.linalg.qr(action)[0]

    def leading(basis):
        form = np.zeros((basis.shape[1], basis.shape[1]))
        for b, s in blocks():
            form += (s @ basis).T @ ((s - b) @ basis)
        values, vectors = np.linalg.eigh((form + form.T) / 2)
        return values[-1], basis @ vectors[:, -1]

    sketch_width = 8
    basis = (np.eye(cols) if cols <= sketch_width else
             range_basis(np.random.default_rng(0).standard_normal((cols, sketch_width))))
    value, x = leading(basis)
    abs_x = np.abs(x)

    arithmetic = 8 * math.ulp(1.) * (rows + cols + 2)

    def storage_radius(weights: np.ndarray, dtype: str) -> np.ndarray:
        # For a stored normal w = mantissa * 2**e, |mantissa| in [1/2,1),
        # half the larger adjacent spacing is 2**(e-2) times spacing at 1.
        relative, absolute = _SPACING[dtype]
        radius = np.ldexp(relative / 4, np.frexp(weights)[1])
        radius[weights == 0] = 0.
        np.maximum(radius, math.ldexp(absolute, -exponent), out=radius)
        return radius + arithmetic * np.abs(weights)

    gap = 0.
    # An absolute float64 floor also covers underflow in scaled arithmetic.
    budget = 16 * base.size * float(np.finfo(float).tiny)
    for b, s in blocks():
        bx, sx = b @ x, s @ x
        dx = sx - bx
        gap += float(sx @ dx)
        eb = storage_radius(b, base_dtype) @ abs_x
        es = storage_radius(s, suspect_dtype) @ abs_x
        # Expanding (s+δs)(s+δs−b−δb) row by row gives (2s−b)δs
        # − sδb − δsδb + δs². The final square is nonnegative.
        budget += float(np.abs(sx + dx) @ es + np.abs(sx) @ eb + es @ eb
                        + arithmetic * (np.abs(sx) @ np.abs(dx)))
    if gap > budget:
        return 1. - budget / gap

    # Refine the probe, then restrict the expensive joint interval test to
    # influential columns. These limits cap work and never relax a bound.
    if cols > sketch_width:
        value, x = leading(range_basis(basis))
    width = min(cols, 1024, (1 << 20) // rows)
    if not width:
        return 0.
    selected = np.argsort(-np.abs(x), kind="stable")[:width]
    b = np.ldexp(base[:, selected], -exponent, dtype=np.float64)
    s = np.ldexp(suspect[:, selected], -exponent, dtype=np.float64)
    rb, rs = storage_radius(b, base_dtype), storage_radius(s, suspect_dtype)
    z = x[selected]
    # Symmetry is required even if every quadratic response is nonpositive.
    skew, try_fit = skew_probe(b, s, rb, rs)
    excess = 0.
    if value > math.sqrt(np.finfo(float).tiny):
        excess = joint_excess(b, s, rb, rs, z, value)
        if not excess and not try_fit:
            try_fit = promising_symmetric_response(b, s, rb, rs, z)
    if not excess and try_fit:
        b, rb = _fit_storage_box(b, base_dtype, exponent, arithmetic)
        s, rs = _fit_storage_box(s, suspect_dtype, exponent, arithmetic)
        excess = fit_excess(b, s, rb, rs)
    # The peer ranker compares margins. A weak skew certificate must not
    # replace a stronger certificate from the quadratic response.
    return max(skew, excess)


def reciprocity_excess(base: np.ndarray, delta: np.ndarray, stats: DeltaStats,
                       base_dtype: str, suspect_dtype: str) -> float | None:
    """Fraction of a two-input reciprocity discrepancy left after bounding rounding.

    If S = (I − P)B for any symmetric P, then BᵀS is symmetric. For any probes,
    (Bx)ᵀ(Sy) = (By)ᵀ(Sx), even when neither probe recovers the true axis.
    """
    x, residual = stats.base_probe, stats.fit_residual
    scale = stats.frobenius
    if not scale or not math.isfinite(scale):
        return None
    scale = max(scale, float(base.max()), -float(base.min()))
    if stats.direction is None:
        if stats.tail_norm_fraction is None:
            return None
        i, j = _skew_columns(base, delta, scale)
        if i == j:
            return 0.
        base, delta = base[:, [i, j]], delta[:, [i, j]]
        probes = np.eye(2)
    else:
        if x is None or residual is None:
            return None
        y = residual - (x @ residual) * x
        length = np.linalg.norm(y)
        if length <= math.sqrt(math.ulp(1.)):
            return 0.
        probes = np.column_stack((x, y / length))
    magnitudes = np.abs(probes)
    # Half a storage ULP on each side, plus conservative error in the
    # subtraction, scaled matvecs, inner products and their accumulation.
    arithmetic = 2 * float(np.finfo(delta.dtype).eps) + 8 * math.ulp(1.) * (sum(base.shape) + 2)
    b_rel, b_abs = _SPACING[base_dtype]
    s_rel, s_abs = _SPACING[suspect_dtype]
    one_norms = magnitudes.sum(axis=0)
    gap = 0.
    # Scaled products can still underflow when most of the base's energy is
    # outside the probe subspace. Charge an absolute arithmetic floor too.
    budget = 8 * base.size * float(np.finfo(np.float64).tiny)
    block_rows = max(1, _ROWS_PER_BLOCK * min(base.shape) // base.shape[1])
    for start in range(0, len(base), block_rows):
        sl = slice(start, start + block_rows)
        b = base[sl].astype(np.float64) / scale
        d = delta[sl].astype(np.float64) / scale
        bp = b @ probes
        dp = d @ probes
        sp = bp + dp
        # BᵀB is symmetric, so the measured skew needs only B and Δ.
        gap += float(bp[:, 0] @ dp[:, 1] - bp[:, 1] @ dp[:, 0])
        b_mag = np.abs(b) @ magnitudes
        s_mag = b_mag + np.abs(d) @ magnitudes
        be = (0.5 * b_rel + arithmetic) * b_mag + (b_abs / scale) * one_norms
        se = (0.5 * s_rel + arithmetic) * s_mag + (s_abs / scale) * one_norms
        # Error of each product in (Bx)ᵀ(Sy) − (By)ᵀ(Sx), including
        # the product of the two input error bounds.
        budget += float(np.sum(np.abs(bp) * se[:, ::-1] + be * (np.abs(sp[:, ::-1]) + se[:, ::-1])))
    return 1. - budget / abs(gap) if abs(gap) > budget else 0.


def _projection_fit(right: np.ndarray, projected: np.ndarray, gain: float | None) -> tuple[float, float, np.ndarray]:
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
    return error, float(deviations.max() / bound), residual


def _subspace_dimension(energy: np.ndarray, total: float, shape: tuple[int, int], noise: float | None) -> int:
    """Choose a resolved prefix of the scaled Gram spectrum for peer comparison."""
    if noise is None:  # Legacy callers without checkpoint storage metadata.
        gaps = np.flatnonzero(energy[:-1] > 4 * energy[1:])
        dim = int(gaps[0] + 1) if gaps.size else len(energy)
        return dim if energy[dim - 1] >= min(energy[0] / 4, total / len(energy)) else 0

    # Weyl's inequality bounds each singular-value error by the operator
    # norm of the rounding perturbation, itself bounded by its Frobenius norm.
    # Also exclude eigenvalues at the scale of error in the Gram product.
    gram_floor = math.sqrt(math.ulp(1.) * sum(shape) * total)
    dim = int(np.count_nonzero(np.sqrt(energy) > max(noise, gram_floor)))
    if dim < 2:
        return 0
    if dim < shape[0] and (dim == len(energy) or energy[dim - 1] > 4 * energy[dim]):
        return dim
    # The floor cuts a cluster or the resolved modes span the whole output.
    # Find a distinct prefix above the bulk without splitting a tied space.
    gaps = np.flatnonzero((energy[:dim - 1] > 4 * energy[1:dim])
                          & (energy[:dim - 1] >= min(energy[0] / 4, total / len(energy))))
    return int(gaps[-1] + 1) if gaps.size else 0


def analyze_delta(w_base: np.ndarray, w_suspect: np.ndarray, k: int, subspace_budget: float | None = None) -> DeltaStats:
    """Spectrum of the change from `w_base` to `w_suspect` for one weight matrix.

    Inputs
        w_base, w_suspect: 2-D arrays of identical shape (out_features, in_features) and one
            dtype, as stored in the checkpoint: float32, or float64 when either side is F64.
            Both are fresh copies; modifying them is fine.
        k: how many leading singular values to return, k >= 1.
        subspace_budget: Frobenius bound on checkpoint rounding in physical units. Without
            storage metadata, retain the leading-cluster heuristic. Infinity skips extraction.

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
        fit_residual: unexplained leading coefficients in input space, used as a contraction probe.
        base_probe: normalized base coefficients along the leading output direction.
        output_subspace: left basis for a separated cluster of leading modes.

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
    # Eckart–Young, evaluated before restoring physical units: separate F64
    # subnormal norms can round differently and distort a ratio of the two.
    tail_norm_fraction = float(np.sqrt(tail / total))
    second_share = float(energy[1] / tail) if tail else 0.0

    # A repeated or nearly repeated leading singular value has no stable axis.
    direction = gain = projection_error = column_spike = fit_residual = base_probe = None
    output_subspace = None
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
            base_probe = projected
        projection_error, column_spike, fit_residual = _projection_fit(right, projected, gain)

    noise = None if subspace_budget is None else subspace_budget / scale
    dim = _subspace_dimension(energy, total, w_suspect.shape, noise)
    if 2 <= dim < len(w_suspect):
        small = vectors[:, -dim:]
        if tall:
            output_subspace = np.empty((len(dw), dim))
            singular = np.sqrt(energy[:dim][::-1])
            for start in range(0, len(dw), _ROWS_PER_BLOCK):
                sl = slice(start, start + _ROWS_PER_BLOCK)
                output_subspace[sl] = ((dw[sl].astype(np.float64) / scale) @ small) / singular
        else:
            output_subspace = small.copy()

    return DeltaStats(np.sqrt(energy[:k]) * scale, effective_rank, float(np.sqrt(total) * scale),
                      fraction, second_share, direction, gain, projection_error, column_spike, tail_norm_fraction,
                      fit_residual, base_probe, output_subspace)
