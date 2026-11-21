"""Peer comparisons for changed tensors of an expected kind."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from llmscan.inventory import InventoryRow
from llmscan.names import RESIDUAL_WRITERS, TOKEN_MATRICES, canonical_kind, stack_of


class ProjectionResidual(NamedTuple):
    """Orthogonal parts of the signed projection's misfit, in units of ‖ΔW‖F."""
    leading: float
    tail: float

    def leading_share(self) -> float | None:
        norm = math.hypot(self.leading, self.tail)
        return self.leading / norm if norm else None


class Measurements(NamedTuple):
    effective_rank: float | None = None
    sv_001: float | None = None
    frobenius: float | None = None
    projection_gain: float | None = None
    second_tail_share: float | None = None
    token_row_fraction: float | None = None
    projection_error: float | None = None
    token_row_spike: float | None = None
    rank_one_excess: float | None = None
    tail_norm_fraction: float | None = None
    contraction_excess: float | None = None
    reciprocity_excess: float | None = None
    removal_excess: float | None = None

    def projection_residual(self) -> ProjectionResidual | None:
        if self.projection_error is None or not self.frobenius or not self.sv_001:
            return None
        tail = self.tail_norm_fraction
        if tail is None:  # Older CSVs only store the two norms in physical units.
            leading = min(1., self.sv_001 / self.frobenius)
            tail = math.sqrt((1 - leading) * (1 + leading))
        else:
            leading = math.sqrt((1 - tail) * (1 + tail))
        return ProjectionResidual(self.projection_error * leading, tail)


@dataclass(frozen=True)
class Peers:
    count: int
    median: float
    mad: float
    outer_fence: float

    @classmethod
    def summarize(cls, values: list[float]) -> Peers:
        median = statistics.median(values)
        q1, _, q3 = statistics.quantiles(values, method="inclusive")
        return cls(len(values), median, statistics.median(abs(x - median) for x in values), q3 + 3 * (q3 - q1))

    def high(self, value: float, floor: float) -> bool:
        # Six robust standard deviations and a practical effect size. Small
        # groups don't estimate dispersion reliably. MAD may be exactly zero.
        return self.count >= 8 and value - self.median > max(6 * 1.4826 * self.mad, floor)


def peer_groups(rows: list[InventoryRow], stats: dict[str, Measurements]) -> dict[tuple, Peers]:
    groups = defaultdict(list)
    for r in rows:
        m = stats[r.name]
        kind = canonical_kind(r.kind)
        if m.effective_rank is not None:
            groups[kind, "rank"].append(m.effective_rank)
        elif m.sv_001 is not None:
            groups[kind, "sv_only"].append(m.sv_001)
        fit_group = (kind, r.base_shape, r.base_dtype, r.suspect_dtype)
        if m.projection_error is not None:
            groups[*fit_group, "projection_error"].append(m.projection_error)
        elif m.reciprocity_excess is not None:
            groups[*fit_group, "multimode_reciprocity"].append(m.reciprocity_excess)
        if m.removal_excess is not None:
            groups[*fit_group, "removal_excess"].append(m.removal_excess)
        residual = m.projection_residual()
        share = residual.leading_share() if residual is not None else None
        if share is not None:
            groups[*fit_group, "leading_share"].append(share)
        if m.rank_one_excess is not None:
            groups[kind, r.base_shape, "rank_one_excess"].append(m.rank_one_excess)
        # A gain measures the leading change relative to the base's weight in
        # that direction. With older CSVs only raw magnitudes are available.
        for metric in ("projection_gain", "frobenius", "sv_001"):
            magnitude = getattr(m, metric)
            if magnitude is not None:
                groups[kind, r.base_shape, metric].append(math.log(magnitude))
    return {group: Peers.summarize(v) for group, v in groups.items() if len(v) >= 3}


def _projection_reason(r: InventoryRow, m: Measurements, peers: dict[tuple, Peers], factor: float) -> str | None:
    """Choose one finding from correlated measurements of the leading change."""
    kind = canonical_kind(r.kind)
    # A local magnitude bound also works for a singleton or a shifted peer
    # median: removal of a unit direction can account for at most one full
    # projection. Allow the same factor of slack as the coarse rank check.
    if m.projection_gain is not None and kind in RESIDUAL_WRITERS | TOKEN_MATRICES and m.projection_gain > max(1., factor):
        return (f"projection_gain {m.projection_gain:.4g} exceeds a full projection of the base "
                f"(threshold {max(1., factor):g}x)")
    residual = m.projection_residual()
    if m.projection_error is None and m.reciprocity_excess is not None and m.reciprocity_excess > 0:
        group = peers.get((kind, r.base_shape, r.base_dtype, r.suspect_dtype, "multimode_reciprocity"))
        if group is None or group.count < 8 or group.high(m.reciprocity_excess, math.sqrt(math.ulp(1.))):
            return (f"multi-mode change violates projection reciprocity "
                    f"({m.reciprocity_excess:.1%} of the discrepancy exceeds rounding)")
    if residual is not None and m.projection_error is not None:
        fit_group = (kind, r.base_shape, r.base_dtype, r.suspect_dtype)
        fits = peers.get((*fit_group, "projection_error"))
        large_error = (m.projection_error > 0.05
                       and (fits is None or fits.count < 8 or fits.high(m.projection_error, 0.05)))
        shares = peers.get((*fit_group, "leading_share"))
        share = residual.leading_share()
        if fits and shares and share is not None and shares.high(share, 0.05):
            # Demand a multiple of the upper peer noise level (median plus
            # dispersion), so a naturally variable cohort gets more latitude.
            # The float64 Gram spectrum resolves squared norms.
            above_noise = fits.high(m.projection_error / max(1., factor), math.sqrt(math.ulp(1.)))
            if large_error or above_noise:
                return (f"projection law leaves {100 * m.projection_error:.3g}% of the leading change unexplained "
                        f"(peer median {100 * fits.median:.3g}%), and its axis carries {share:.1%} "
                        f"of the residual norm (peer median {shares.median:.1%})")
        # Without corroboration from residual shares, retain the conservative
        # local test, including for singleton kinds.
        if large_error and residual.leading > factor * residual.tail:
            return (f"projection law leaves {m.projection_error:.1%} of the leading change unexplained "
                    "after fitting partial removal against the base")
        if fits is None or fits.count < 8 or fits.high(m.projection_error, math.sqrt(math.ulp(1.))):
            # The directional rounding budgets already exclude exact removals.
            # With enough peers, allow for systematic non-projection edits.
            if m.contraction_excess is not None and m.contraction_excess > 0:
                return (f"partial removal fails the contraction test on the unexplained input direction "
                        f"by {100 * m.contraction_excess:.3g}% of the change norm after allowing for rounding")
            if m.reciprocity_excess is not None and m.reciprocity_excess > 0:
                return (f"base and edited responses violate projection reciprocity "
                        f"({m.reciprocity_excess:.1%} of the discrepancy exceeds rounding)")
    if m.removal_excess is not None and m.removal_excess > 0:
        excess = peers.get((kind, r.base_shape, r.base_dtype, r.suspect_dtype, "removal_excess"))
        # Values below the rounding bound are clipped to zero. Their MAD can
        # vanish even when resolvable violations recur throughout the cohort.
        if excess is None or excess.count < 8 or m.removal_excess > excess.outer_fence + math.sqrt(math.ulp(1.)):
            return (f"edited responses violate the inner product constraints of partial removal "
                    f"({100 * m.removal_excess:.3g}% of the measured violation exceeds rounding)")
    for metric in ("projection_gain", "frobenius", "sv_001"):
        magnitude = getattr(m, metric)
        group_stats = peers.get((kind, r.base_shape, metric))
        if group_stats and group_stats.count >= 8 and magnitude is not None:
            # Gain is dimensionless and physically bounded. Raw norms, which
            # vary widely between genuine layers, need a larger effect size.
            floor = math.log1p(0.05) if metric == "projection_gain" else math.log(2)
            if group_stats.high(math.log(magnitude), floor) and (metric != "projection_gain" or magnitude > 1.05):
                return f"{metric} {magnitude:.4g} above peer dispersion (median {math.exp(group_stats.median):.4g})"
            break
    return None


def outlier_reasons(r: InventoryRow, m: Measurements, peers: dict[tuple, Peers],
                    direction_reason: str | None, factor: float) -> list[str]:
    reasons = []
    kind = canonical_kind(r.kind)
    er = m.effective_rank
    label, value, group = ("effective_rank", er, "rank") if er is not None else ("sv_001", m.sv_001, "sv_only")
    group_stats = peers.get((kind, group))
    if group_stats and value is not None:
        median = group_stats.median
        if value > factor * median:
            reasons.append(f"{label} {value:.4g} is {value / median:.1f}x the median {median:.4g} of changed {kind}"
                           f" tensors (threshold {factor:g}x)")
        elif (m.second_tail_share is not None and m.second_tail_share >= 0.25 and er is not None
              and group_stats.high(er, 0.01 * median)):
            reasons.append(f"effective_rank {er:.4g} above peer dispersion (median {median:.4g}); "
                           f"second mode carries {m.second_tail_share:.0%} of non-leading energy")

    # Give spectral shape one vote even when more than one test explains it.
    if not reasons and m.rank_one_excess is not None and m.rank_one_excess > 0.05:
        tails = peers.get((kind, r.base_shape, "rank_one_excess"))
        if tails is None or tails.count < 8 or tails.high(m.rank_one_excess, 0.05):
            reasons.append(f"at least {m.rank_one_excess:.1%} of the change norm lies outside a rank-one "
                           "removal after allowing for checkpoint rounding")

    projection = _projection_reason(r, m, peers, factor)
    if projection:
        reasons.append(projection)
    if kind in TOKEN_MATRICES and r.layer is None:
        if m.token_row_fraction is not None and m.token_row_fraction < 0.05:
            reasons.append(f"token-row energy concentrated in {m.token_row_fraction:.2%} effective rows")
        if m.token_row_spike is not None and m.token_row_spike > 1:
            reasons.append(f"token-row residual spike {m.token_row_spike:.2f}x the vocabulary-adjusted bound")
    if direction_reason:
        reasons.append(direction_reason)
    return reasons


def _orthogonal_fraction(cosine: float) -> float:
    c = min(1., abs(cosine))
    return math.sqrt((1 - c) * (1 + c))


def _recurring_rank_shift(names: list[str], members: np.ndarray, stats: dict[str, Measurements]) -> str | None:
    """Corroborate a recurring subspace by a consistent one-sided departure in effective rank."""
    if not np.any(np.diff(np.flatnonzero(members)) > 1):
        return None  # A single run among the observed layers can be a local editing regime.
    ranks = np.array([stats[n].effective_rank for n in names], dtype=float)
    background = ranks[~members]
    if len(background) < 8 or not np.isfinite(ranks).all():
        return None
    ref = Peers.summarize(background.tolist())
    width = max(6 * 1.4826 * ref.mad, 0.01 * ref.median)
    low, high = ranks[members].min(), ranks[members].max()
    if high < min(background.min(), ref.median - width) or low > max(background.max(), ref.median + width):
        return f"effective ranks {low:.3g}–{high:.3g} vs other layers' median {ref.median:.3g}"
    return None


def _secondary_switch(ids: list[int], names: list[str], majority: np.ndarray, angles: np.ndarray,
                      secondary_overlap: np.ndarray, agreement: np.ndarray, directions: dict[str, np.ndarray] | None,
                      width: int, factor: float) -> str | None:
    """Look for an isolated switch of secondary modes under a stable leading edit."""
    if majority.sum() < 8 or (~majority).sum() < 3:
        return None
    same = majority[:, None] == majority
    if not np.array_equal(agreement, same):
        return None  # Need two distinct, internally coherent groups and no transitional layers.
    adjacent = np.diff(ids) == 1
    switches = majority[:-1] != majority[1:]
    terminal = switches.sum() == 1
    if terminal:
        if not adjacent.all():
            return None  # Gaps can hide a transition or a return to the majority.
    elif not (np.any(adjacent & majority[:-1] & ~majority[1:])
              and np.any(adjacent & ~majority[:-1] & majority[1:])):
        return None
    if not directions or any(name not in directions or directions[name].shape != (width,) for name in names):
        return None  # A tied leading space has no independently resolved primary direction.
    axes = np.stack([directions[name] for name in names])
    primary = np.arccos(np.clip(np.abs(axes @ axes.T), 0., 1.))
    floor = math.sqrt(2 * axes.shape[1] * math.ulp(1.))
    factor = max(1., factor)
    if not (primary.max() <= math.acos(0.9)
            and primary[~same].max() <= factor * max(floor, primary[same].max())):
        return None
    across, within = angles[~same].min(), angles[same].max()
    separation_limit = factor * max(floor, within)
    if across <= separation_limit:
        return None
    boundary = ""
    if terminal:
        i = int(np.flatnonzero(switches)[0])
        primary_steps = primary.diagonal(offset=1)
        if primary_steps[i] > factor * max(floor, primary_steps[~switches].max()):
            return None  # The leading edit itself changes at this boundary.
        # Use the smallest angle after the best aligned pair. Even the
        # most similar remaining directions must separate by more than
        # ordinary variation across either group, including adjacent steps.
        secondary = math.acos(min(1., secondary_overlap[~same].max()))
        if secondary <= separation_limit:
            return None
        boundary = (f" at boundary {ids[i]}–{ids[i+1]} "
                    f"(minimum secondary angle {math.degrees(secondary):.2f} degrees)")
    return (f"abrupt secondary subspace switch in {(~majority).sum()} layers{boundary} "
            f"(minimum between-group angle {math.degrees(across):.2f} degrees, "
            f"maximum within-group angle {math.degrees(within):.2f} degrees, "
            f"leading axes within {math.degrees(primary.max()):.2f} degrees)")


def subspace_outliers(rows: list[InventoryRow], subspaces: dict[str, np.ndarray], factor: float,
                      stats: dict[str, Measurements] | None = None,
                      directions: dict[str, np.ndarray] | None = None) -> dict[str, str]:
    """Compare equal-dimensional output spaces, invariant to signs and rotations inside a basis."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        kind = canonical_kind(r.kind)
        if r.name in subspaces and r.layer is not None and kind in RESIDUAL_WRITERS:
            key = (stack_of(r.name), kind, r.base_shape, r.base_dtype, r.suspect_dtype, subspaces[r.name].shape)
            groups[key][r.layer].append(r.name)

    findings = {}
    for layers in groups.values():
        # One independent vote per layer. Ambiguous layers (e.g. several
        # same-kind experts in one block) do not define a reference trajectory.
        ids = sorted(i for i in layers if len(layers[i]) == 1)
        if len(ids) < 9:
            continue
        names = [layers[i][0] for i in ids]
        width, dim = subspaces[names[0]].shape
        angles = np.zeros((len(ids), len(ids)))
        secondary_overlap = np.ones_like(angles)
        for i, name in enumerate(names):
            for j in range(i):
                cosines = np.linalg.svd(subspaces[name].T @ subspaces[names[j]], compute_uv=False)
                angles[i, j] = angles[j, i] = math.acos(min(1., float(cosines[-1])))
                if dim >= 2:
                    secondary_overlap[i, j] = secondary_overlap[j, i] = cosines[1]
        agreement = angles <= math.acos(0.9)
        supporters = agreement[np.argmax(agreement.sum(axis=1))]
        if supporters.sum() <= len(layers) / 2:
            # A repeated space can itself be exceptional among otherwise
            # unrelated spaces. Require an isolated clique, recurrence across
            # separate layer ranges, and an independent spectral departure.
            member_ids = np.flatnonzero(supporters)
            if (stats is not None and 3 <= len(member_ids) < len(ids) / 2
                    and np.all(agreement[supporters] == supporters)
                    and np.all(agreement[~supporters].sum(axis=1) < 3)):
                shift = _recurring_rank_shift(names, supporters, stats)
                if shift:
                    for i in member_ids:
                        findings[names[i]] = (f"recurring {dim}-dimensional output subspace in {len(member_ids)} "
                                              f"layers without a comparable peer group ({shift})")
            continue
        switch = _secondary_switch(ids, names, supporters, angles, secondary_overlap, agreement, directions,
                                   width, factor)
        # Compare each candidate with the coherent population excluding itself.
        np.fill_diagonal(angles, np.nan)
        separation = np.nanmedian(angles[:, supporters], axis=1)
        for i, name in enumerate(names):
            if switch and not supporters[i]:
                findings[name] = switch
                continue
            peers = Peers.summarize(np.delete(separation, i).tolist())
            limit = max(max(1., factor) * peers.median, peers.median + 6 * 1.4826 * peers.mad,
                        math.sqrt(2 * width * math.ulp(1.)))
            if separation[i] <= limit:
                continue
            # Several mutually aligned layers can represent another editing
            # concept. Demand spectral corroboration for an isolated cohort.
            members = agreement[i]
            if (stats is not None and members.sum() >= 3 and not np.any(members & supporters)
                    and np.all(agreement[members] == members) and not _recurring_rank_shift(names, members, stats)):
                continue
            # Allow smooth drift. The largest principal angle is a metric.
            # A constant-speed path through the adjacent layers covers each
            # interval in proportion to its length, also at either endpoint.
            other = [j for j in range(len(ids)) if j != i]
            start = max(0, min(i - 1, len(other) - 2))
            a, b = other[start:start + 2]
            t = (ids[i] - ids[a]) / (ids[b] - ids[a])
            if (-1 <= t <= 2 and agreement[a, b] and angles[i, a] <= abs(t) * angles[a, b] + limit
                    and angles[i, b] <= abs(1 - t) * angles[a, b] + limit):
                continue
            findings[name] = (f"{dim}-dimensional output subspace differs from coherent layers "
                              f"(largest principal angle, median {math.degrees(separation[i]):.2f} degrees "
                              f"vs peer upper bound {math.degrees(limit):.2f})")
    return findings


def _follows_drift(layer: int, axis: np.ndarray, neighbors: list[int], axes: dict[int, np.ndarray],
                   tolerance: float) -> bool:
    """Compare with the local trajectory of two independent same-kind layers."""
    # Bracket the target where possible. At either end, extrapolate by at most
    # one observed interval. Missing/sparsely sampled blocks need no special ids.
    pos = int(np.searchsorted(neighbors, layer))
    start = max(0, min(pos - 1, len(neighbors) - 2))
    i, j = neighbors[start:start + 2]
    t = (layer - i) / (j - i)
    a, b = axes[i], axes[j]
    dot = float(a @ b)
    if t < -1 or t > 2 or abs(dot) < 0.9:
        return False
    # Sign-align the neighboring axes and interpolate along their shorter arc.
    angle = math.acos(min(1., abs(dot)))
    if angle == 0:
        prediction = a
    else:
        prediction = (math.sin((1 - t) * angle) * a
                      + math.copysign(1., dot) * math.sin(t * angle) * b) / math.sin(angle)
    return _orthogonal_fraction(float(axis @ prediction) / np.linalg.norm(prediction)) <= tolerance


def direction_outliers(rows: list[InventoryRow], directions: dict[str, np.ndarray],
                       cosines: dict[str, float], factor: float) -> dict[str, str]:
    """Calibrate departure from a coherent stack against comparable writer layers."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        kind = canonical_kind(r.kind)
        if r.name in cosines and r.layer is not None and kind in RESIDUAL_WRITERS:
            key = (stack_of(r.name), len(directions[r.name]), kind, r.base_shape, r.base_dtype, r.suspect_dtype)
            groups[key][r.layer].append(r.name)
    findings = {name: f"residual direction disagrees with coherent peer layers (absolute cosine {cosine:.3f})"
                for name, cosine in cosines.items() if cosine < 0.5}
    for layers in groups.values():
        if len(layers) < 9:  # eight other layer votes per candidate
            continue
        departures = {i: statistics.median(_orthogonal_fraction(cosines[n]) for n in names)
                      for i, names in layers.items()}
        # Only unambiguous same-kind axes can define a local trajectory.
        axes = {i: directions[names[0]] for i, names in layers.items() if len(names) == 1}
        for layer, names in layers.items():
            peers = Peers.summarize([v for i, v in departures.items() if i != layer])
            width = len(directions[names[0]])
            # Dot-product roundoff is O(width * eps) in cosine, hence its
            # square root in sine near perfectly aligned directions.
            limit = max(max(1., factor) * peers.median, peers.median + 6 * 1.4826 * peers.mad,
                        math.sqrt(2 * width * math.ulp(1.)))
            for name in names:
                findings.pop(name, None)  # a calibrated decision replaces the sparse-cohort fallback
                separation = _orthogonal_fraction(cosines[name])
                if separation <= limit:
                    continue
                neighbors = sorted(i for i in axes if i != layer)
                if len(neighbors) >= 2 and _follows_drift(layer, directions[name], neighbors, axes, limit):
                    continue
                findings[name] = (f"residual direction has {separation:.2%} orthogonal component to coherent layers "
                                  f"(peer median {peers.median:.2%}, upper bound {limit:.2%})")
    return findings


def direction_cosines(rows: list[InventoryRow], directions: dict[str, np.ndarray]) -> dict[str, float]:
    """Compare to other layers only. Sign is arbitrary. Give each layer one vote.

    With no tightly aligned majority (e.g. different legitimate axes across
    layers), abstain. Embeddings can be judged but never set the reference.
    """
    spaces = defaultdict(list)
    stacks = defaultdict(set)
    for r in rows:
        if r.name in directions and r.layer is not None and canonical_kind(r.kind) in RESIDUAL_WRITERS:
            stacks[len(directions[r.name])].add(stack_of(r.name))
    for r in rows:
        if r.name not in directions:
            continue
        width = len(directions[r.name])
        stack = stack_of(r.name)
        if r.layer is None and canonical_kind(r.kind) in TOKEN_MATRICES and stack not in stacks[width]:
            # An outer head may sit above the block stack (lm_head vs
            # model.layers). Never choose between multiple matching subtrees.
            candidates = [s for s in stacks[width] if not stack or s.startswith(stack + ".")]
            if len(candidates) == 1:
                stack = candidates[0]
        spaces[stack, width].append(r)
    result = {}
    for members in spaces.values():
        layers = defaultdict(list)
        for r in members:
            if r.layer is not None and canonical_kind(r.kind) in RESIDUAL_WRITERS:
                layers[r.layer].append(directions[r.name])
        refs = {}
        for layer, vectors in layers.items():
            dots = np.asarray(vectors) @ vectors[0]
            if np.all(np.abs(dots) >= 0.9):
                axis = np.sign(dots) @ np.asarray(vectors)
                refs[layer] = axis / np.linalg.norm(axis)
        if len(refs) < 3:
            continue
        layer_ids = np.array(sorted(refs))
        axes = np.stack([refs[i] for i in layer_ids])
        similarity = np.abs(axes @ axes.T)
        # Find the layer with the most agreeing layers. Require a strict
        # majority and use just its supporters as the reference population.
        agreement = similarity >= 0.9
        supporters = agreement[np.argmax(agreement.sum(axis=1))]
        if supporters.sum() <= len(layers) / 2:
            continue
        axes, layer_ids = axes[supporters], layer_ids[supporters]
        for r in members:
            other = axes if r.layer is None else axes[layer_ids != r.layer]
            if len(other) >= 3:
                result[r.name] = min(1.0, float(np.median(np.abs(other @ directions[r.name]))))
    return result
