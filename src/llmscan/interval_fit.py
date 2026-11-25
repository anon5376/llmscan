"""Certify separation of two storage boxes from the partial-removal constraint."""

import math

import numpy as np


def _box_epsilon(shape):
    return 128 * (sum(shape) + 1) * math.ulp(1.)


def _box_excess(b, s, rb, rs, factor, skew):
    """Lower-bound <M, Sᵀ(S−B)> over both entrywise intervals, M = FFᵀ + K."""
    m = factor @ factor.T + (skew - skew.T) / 2
    d = s - b
    gap = np.sum((s.T @ d) * m, axis=1).sum()
    # All dot products and reductions in the certificate run along one
    # matrix dimension at a time. Include cancellation in FFᵀ + K, the
    # Gram product, the gradients, and the final row sums in the allowance.
    eps = _box_epsilon(b.shape)
    size = np.linalg.norm(factor)**2 + np.linalg.norm(skew)
    weights = sum(np.linalg.norm(a) for a in (b, s, rb, rs))
    guard = eps * size * weights**2 + (b.size + m.size) * np.finfo(float).tiny
    if gap <= guard:
        return 0.
    sm = s @ m
    gradient = np.column_stack((sm + d @ m.T, -sm))
    radius = np.column_stack((rs, rb))
    linear_cost = np.sum(np.abs(gradient) * radius, axis=1).sum()
    if gap <= linear_cost + guard:
        return 0.  # Spend the quadratic search only on a promising witness.

    # For one row with errors e = [e_s, e_b], the exact change in the
    # witness is gradient·e + eᵀHe. Account for the same errors in both
    # terms. Write H >= -t I - uuᵀ, keeping the worst curvature direction.
    h = np.block([[(m + m.T) / 2, -m / 2], [-m.T / 2, np.zeros_like(m)]])
    values, axes = np.linalg.eigh(h)
    t = max(0., -values[1])
    u = axes[:, 0] * math.sqrt(max(0., -values[0] - t))
    positive = axes * np.sqrt(np.maximum(0., values + t))
    # Verify the spectral lower bound by reconstruction, independently of
    # eigenvector accuracy. A Frobenius residual bounds the operator norm.
    remainder = h + np.outer(u, u) - positive @ positive.T
    remainder.flat[::len(h) + 1] += t
    t += np.linalg.norm(remainder) + eps * (size + t + u @ u + np.linalg.norm(positive)**2)

    # Write e = r*w, |w_i| <= 1. Weighted Cauchy-Schwarz gives
    # (uᵀe)^2 <= (|u|·r) sum_i |u_i| r_i w_i². Thus adding
    # sum_i penalty_i (w_i² - 1) gives a convex lower bound on the box.
    ru = radius * np.abs(u)
    penalty = t * radius**2 + ru.sum(axis=1)[:, None] * ru
    denominator = 2 * (np.linalg.norm(h) * radius**2 + penalty)
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    w = -np.sign(gradient)
    for _ in range(16):
        e = radius * w
        eh = e @ h
        slope = radius * (gradient + 2 * eh) + 2 * penalty * w
        # Minimize the tangent plane on the box. Every iterate gives a
        # valid lower bound, irrespective of convergence of the search.
        cost = np.sum(e * eh + penalty * (1 + w*w) + np.abs(slope), axis=1).sum()
        if gap > cost + guard:
            return min(1., float((gap - cost - guard) / gap))
        w = np.clip(w - slope / denominator, -1., 1.)
    return 0.


def _negative_definite(a):
    """Certify a zero PSD projection by reconstructing a shifted Gram matrix."""
    n = len(a)
    # If A = -LLᵀ - shift I + E and ||E||₂ < shift, then A ≺ 0.
    # Leave room for backward error even quite close to the cone boundary.
    scale = np.abs(a.diagonal()).sum()
    eps = 16 * (n + 1) * math.ulp(1.)
    floor = 16 * (n + 1)**2 * np.finfo(float).tiny
    shift = 4 * (eps * scale + floor)
    shifted = -a
    shifted.flat[::n + 1] -= shift
    try:
        lower = np.linalg.cholesky(shifted)
    except np.linalg.LinAlgError:
        return False
    remainder = a + lower @ lower.T
    remainder.flat[::n + 1] += shift
    # The absolute row-sum bounds the symmetric reconstruction residual.
    # Product roundoff has spectral norm <= gamma_n ||L||_F². Allow also
    # for both additions, the norm reductions and gradual underflow.
    guard = eps * (np.einsum("ij,ij->", lower, lower) + scale + shift) + floor
    return np.max(np.abs(remainder).sum(axis=1)) + guard < shift


def _correction_adjoint(sw, dw, w, residual, skew_only):
    """Gradient in the two storage boxes for a residual in whitened coordinates."""
    if skew_only:
        # RᵀWᵀ = -RWᵀ. Both gradients can use the same right product.
        right = residual @ w.T
        base = sw @ right
        return -base, base - dw @ right
    gb = -sw @ residual @ w.T
    return gb, dw @ residual.T @ w.T - gb


def _primal_remainder(s, d, eb, es, shift, weight_norm):
    """Bound the linear-cost gate in _box_excess for *every* witness.

    For any corrections inside the boxes, gap - linear_cost <= <M, A>,
    where A is the linearized corrected form below. If A = -LLᵀ + tI + E,
    this is at most (t + ||E||_F) (||F||_F² + ||K||_F) for M = FFᵀ + K.
    Cholesky only proposes L. Reconstruct in the original coordinates,
    including the skew part, independently of how the fit was obtained.
    """
    form = (s + es).T @ d + s.T @ (es - eb)
    shifted = -(form + form.T) / 2
    shifted.flat[::len(form) + 1] += shift
    try:
        lower = np.linalg.cholesky(shifted)
    except np.linalg.LinAlgError:
        return math.inf
    if np.linalg.norm(lower) > 2 * weight_norm:
        return math.inf
    remainder = form + lower @ lower.T
    remainder.flat[::len(form) + 1] -= shift
    # t and the measured reconstruction each spend 1/16 of the gate's
    # guard per unit witness size. For gamma = (rows+cols+1)*ulp(1),
    # each product's Frobenius error is <= gamma times its operand norms.
    # The two form products have summed norm products <= 2*weight_norm².
    # Bound LLᵀ's separately. The remaining 112*gamma covers these
    # products, additions, norms and the gap/linear-cost evaluation.
    return np.linalg.norm(remainder)


def _polish_primal(b, s, d, rb, rs, eb, es):
    """Douglas–Rachford projections between the two boxes and affine symmetry."""
    weights = sum(np.linalg.norm(x) for x in (b, s, rb, rs))
    # Stay in the normal range for products, squared norms and their
    # error estimates. Outside this range use the ordinary witness search.
    if not np.finfo(float).tiny**0.125 < weights < np.finfo(float).max**0.125:
        return False
    tol = (_box_epsilon(b.shape) / 16) * weights**2
    eb, es = np.clip(eb, -rb, rb), np.clip(es, -rs, rs)
    remainder = _primal_remainder(s, d, eb, es, tol, weights)
    if remainder <= tol:
        return True
    if not math.isfinite(remainder):
        return False

    # The skew constraint is skew(SᵀD + BᵀEs - SᵀEb) = 0. Its affine
    # projection adds SZ to Eb and subtracts BZ from Es, where Z is skew
    # and GZ + ZG equals twice the current skew, G = BᵀB + SᵀS.
    try:
        values, axes = np.linalg.eigh(b.T @ b + s.T @ s)
    except np.linalg.LinAlgError:
        return False
    denom = np.maximum(values[:, None] + values, max(math.ulp(values[-1]), np.finfo(float).tiny))
    bq, sq = b @ axes, s @ axes
    form = s.T @ d
    yb, ys = eb, es
    for step in range(1, 17):
        reflected = form + b.T @ (2 * es - ys) - s.T @ (2 * eb - yb)
        z = (axes.T @ (reflected - reflected.T) @ axes) / denom
        yb = eb + (sq @ z) @ axes.T
        ys = es - (bq @ z) @ axes.T
        eb, es = np.clip(yb, -rb, rb), np.clip(ys, -rs, rs)
        if step in (4, 16) and _primal_remainder(s, d, eb, es, tol, weights) <= tol:
            return True
    return False


def fit_excess(b, s, rb, rs):
    """Seek a separating witness by fitting interval corrections to the Gram form."""
    d = s - b
    values, axes = np.linalg.eigh(s.T @ s + d.T @ d)
    floor = values.mean() / 4
    if floor <= math.sqrt(np.finfo(float).tiny):
        return 0.
    # A congruence preserves symmetry and semidefiniteness. Limit inverse
    # scaling in weak directions when conditioning the search objective.
    w = axes / np.sqrt(np.maximum(values, floor))
    sw, dw = s @ w, d @ w
    form = sw.T @ dw
    eb, es = np.zeros_like(b), np.zeros_like(s)
    yb, ys = eb, es
    # The correction map L(b,s) = (sW)ᵀDW + SWᵀ((s-b)W) is linear.
    # Carry L of both iterates so a step and its extrapolation need only
    # one fresh evaluation. Recompute each candidate from its coordinates.
    correction = np.zeros_like(form)
    extrapolated = correction
    step = floor / 2
    negative_spectrum = False
    for iteration in range(1, 513):
        # Fit the *linearized* corrected form to the cone of symmetric NSD
        # matrices. Its residual is a PSD factor plus a skew matrix. Only
        # _box_excess can turn this search direction into a finding.
        trial = form + extrapolated
        symmetric = (trial + trial.T) / 2
        skew = (trial - trial.T) / 2
        if negative_spectrum and _negative_definite(symmetric):
            factor = w[:, :0]
            residual = skew
        else:
            vals, vecs = np.linalg.eigh(symmetric)
            negative_spectrum = vals[-1] < 0.
            factor = vecs * np.sqrt(np.maximum(0., vals))
            residual = factor @ factor.T + skew
        if iteration & (iteration - 1) == 0:
            excess = _box_excess(b, s, rb, rs, w @ factor, w @ skew @ w.T)
            if excess:
                return excess
            # Polish a box-feasible iterate, never the extrapolated point.
            # An inconclusive probe leaves the witness trajectory intact.
            if iteration >= 8 and _polish_primal(b, s, d, rb, rs, eb, es):
                return 0.
        gb, gs = _correction_adjoint(sw, dw, w, residual, negative_spectrum)
        # Accelerated projected gradient on the two boxes. The projection
        # onto the cone is nonexpansive, so the linear map bounds curvature.
        for _ in range(16):
            tb = np.clip(yb - step * gb, -rb, rb)
            ts = np.clip(ys - step * gs, -rs, rs)
            db, ds = tb - yb, ts - ys
            candidate = (ts @ w).T @ dw + sw.T @ ((ts - tb) @ w)
            change = candidate - extrapolated
            if step * np.linalg.norm(change)**2 <= np.linalg.norm(db)**2 + np.linalg.norm(ds)**2:
                break
            step /= 2
        else:
            return 0.
        if not np.any(tb - eb) and not np.any(ts - es):
            return 0.
        momentum = (iteration - 1) / (iteration + 2)
        yb, ys = tb + momentum * (tb - eb), ts + momentum * (ts - es)
        extrapolated = candidate + momentum * (candidate - correction)
        correction = candidate
        eb, es = tb, ts
        step *= 1.03
    return 0.
