"""Joint storage-interval certificate on a bounded set of input columns."""

import math

import numpy as np


def skew_excess(b, s, rb, rs):
    """Test reciprocity on a plane of the full skew form on the selected columns."""
    d = s - b

    def action(q):
        return b.T @ (d @ q) - d.T @ (b @ q)

    # Two range iterations find coherent skew modes without building a
    # columns-by-columns Gram matrix. The search only selects the witness.
    cols = b.shape[1]
    if cols < 2:
        return 0.
    if cols <= 8:
        basis = np.eye(cols)
    else:
        basis = np.random.default_rng(0).standard_normal((cols, 8))
        for _ in range(2):
            basis = np.linalg.qr(action(basis))[0]
    form = basis.T @ action(basis)
    left, singular, right = np.linalg.svd((form - form.T) / 2)
    if not singular[0]:
        return 0.
    x, y = basis @ left[:, 0], basis @ right[0]

    bx, by = b @ x, b @ y
    gap = abs(bx @ (d @ y) - by @ (d @ x))
    # For M = xyᵀ − yxᵀ the two linear terms charge each weight
    # only once. The mixed term is bounded by ||Rb||F ||Rs||F ||M||2,
    # where ||M||2 is the area of the parallelogram spanned by x and y.
    size = np.linalg.norm(x) * np.linalg.norm(y)
    cost = size * np.linalg.norm(rb) * np.linalg.norm(rs)
    weights = sum(np.linalg.norm(a) for a in (b, s, rb, rs))
    guard = 128 * (sum(b.shape) + 1) * math.ulp(1.) * weights**2 * size
    guard += 16 * b.size * np.finfo(float).tiny
    if gap <= cost + guard:
        return 0.
    cost += np.sum(rs * np.abs(np.outer(bx, y) - np.outer(by, x)), axis=1).sum()
    cost += np.sum(rb * np.abs(np.outer(s @ x, y) - np.outer(s @ y, x)), axis=1).sum()
    return float(1 - (cost + guard) / gap) if gap > cost + guard else 0.


def promising_symmetric_response(b, s, rb, rs, x):
    """Screen the wider fit by comparing its seed with a storage-noise scale."""
    z = x[:256]
    b, s, rb, rs = (a[:, :len(z)] for a in (b, s, rb, rs))
    form = s.T @ (s - b)
    gap = z @ form @ z
    if gap <= 0:
        return False
    # Linearization of sym(SᵀΔ) gives sym(Esᵀ(2S−B) − EbᵀS).
    # For independent uniform errors the expected squared Frobenius norm
    # before symmetrizing is the row sum below divided by three. This
    # scale *only* controls search effort; certification uses the full boxes.
    def row_energy(a):
        return np.einsum("ij,ij->i", a, a)

    noise = row_energy(rs) @ row_energy(2 * s - b) + row_energy(rb) @ row_energy(s)
    energy = z @ z
    if gap > energy * math.sqrt(noise / 3):
        return True
    # Retain the previous search for quiet, nearly reciprocal seeds. The
    # storage model may overestimate the actual variation, for example
    # when a few high-magnitude columns dominate the rounding radii.
    return gap > energy * np.linalg.norm(form - form.T)


def _sparse_input(b, s, rb, rs, x, eigenvalue):
    """Improve the measured gap / interval cost before choosing the columns."""
    def responses(z):
        sx = s @ z
        dx = sx - b @ z
        eb, es = rb @ np.abs(z), rs @ np.abs(z)
        gap = sx @ dx
        cost = np.abs(sx + dx) @ es + np.abs(sx) @ eb + eb @ es
        ratio = gap / cost if cost > 0 else 0.
        return ratio, sx, dx, eb, es

    step = 1 / eigenvalue
    ratio, sx, dx, eb, es = responses(x)
    for _ in range(8):
        if not 0 < ratio < 1:
            break
        # Differentiate gap - ratio * cost. The cost of |x| gives a
        # weighted soft threshold, so low-information columns can drop out.
        v = sx + dx
        signed = np.sign(v) * es
        grad = s.T @ (v - ratio * (2 * signed + np.sign(sx) * eb)) + b.T @ (ratio * signed - sx)
        penalty = ratio * (rs.T @ (np.abs(v) + eb) + rb.T @ (np.abs(sx) + es))
        for _ in range(16):
            z = x + step * grad
            z = np.sign(z) * np.maximum(0., np.abs(z) - step * penalty)
            norm = np.linalg.norm(z)
            if norm:
                z /= norm
                candidate = responses(z)
                if candidate[0] > ratio:
                    break
            step /= 2
        else:
            break
        ratio, sx, dx, eb, es = candidate
        x = z
        step *= 2
    return x


def joint_excess(b, s, rb, rs, x, eigenvalue):
    """Bound <M, Sᵀ(S−B)> for a witness with positive semidefinite symmetric part.

    For exact partial removal Sᵀ(S−B) is symmetric negative semidefinite.
    The skew part of M therefore contributes zero. Each physical weight has
    just one perturbation across *all* the input responses in this test.
    """
    # The input columns arrive in decreasing order of |x|. Only spend the
    # joint search when charging either checkpoint alone leaves a positive
    # gap on the initial support. This screen cannot produce a finding.
    width = min(256, len(x))
    z = x[:width]
    sx = s[:, :width] @ z
    dx = sx - b[:, :width] @ z
    if sx @ dx <= max(np.abs(sx + dx) @ (rs[:, :width] @ np.abs(z)),
                      np.abs(sx) @ (rb[:, :width] @ np.abs(z))):
        return 0.

    x = _sparse_input(b, s, rb, rs, x, eigenvalue)
    keep = np.argsort(-np.abs(x), kind="stable")[:width]
    x = x[keep]
    # A congruence preserves semidefiniteness. Weight the columns according
    # to the seed, with a finite scale for coordinates the search discarded.
    scale = np.sqrt(np.maximum(np.abs(x), 1 / width))
    b, s, rb, rs = (a[:, keep] * scale for a in (b, s, rb, rs))
    d = s - b
    form = s.T @ d
    mixed = np.linalg.norm(rs) * np.linalg.norm(rb)
    magnitude = sum(np.linalg.norm(a) for a in (s, b, rs, rb))
    # Dot products, positive sums and reconstruction of the witness from its
    # PSD factor and skew part. Bound them in absolute Frobenius norm even
    # when the measured inner product has severe cancellation.
    operations = 32 * (b.size + width**2 + 1)
    roundoff = operations * math.ulp(1.) * magnitude**2
    floor = operations * np.finfo(float).tiny

    def evaluate(m):
        gs, gb = d @ m.T + s @ m, s @ m
        gap = np.sum(form * m)
        # In the expansion in errors E_s and E_b, the gradients are gs and
        # -gb. The pure E_s square is nonnegative. The absolute mixed term
        # is at most ||R_s||_F ||R_b||_F ||M||_F by Cauchy-Schwarz.
        cost = np.sum(rs * np.abs(gs)) + np.sum(rb * np.abs(gb)) + mixed * np.linalg.norm(m)
        return gap, cost, gs, gb

    seed = x / scale
    m = np.outer(seed, seed)
    factor_size = seed @ seed
    step = width / max(np.linalg.norm(b)**2, eigenvalue)
    for _ in range(128):
        gap, cost, gs, gb = evaluate(m)
        budget = cost + roundoff * factor_size + floor
        if gap > budget:
            return float(1 - budget / gap)
        norm = np.linalg.norm(m)
        if not norm:
            break
        signed = rs * np.sign(gs)
        grad = form - signed.T @ d - s.T @ (signed + rb * np.sign(gb)) - (mixed / norm) * m
        for _ in range(16):
            trial = m + step * grad
            vals, vecs = np.linalg.eigh((trial + trial.T) / 2)
            factor = vecs * np.sqrt(np.maximum(0., vals))
            skew = (trial - trial.T) / 2
            candidate = factor @ factor.T + skew
            new_gap, new_cost, _, _ = evaluate(candidate)
            if new_gap - new_cost > gap - cost:
                break
            step /= 2
        else:
            break
        m = candidate
        factor_size = np.sum(factor**2) + np.linalg.norm(skew)
        step *= 1.5
    return 0.
