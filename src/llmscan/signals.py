"""Peer comparisons for changed tensors of an expected kind."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterator
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


def _on_plateau(layer: int, neighbors: set[int]) -> bool:
    """Does this layer belong to three consecutive, independently matching observations?"""
    return any(layer + a in neighbors and layer + b in neighbors for a, b in [(-2, -1), (-1, 1), (1, 2)])


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


def _coherent_groups(angles: np.ndarray, layer_count: int, factor: float, floor: float) -> Iterator[np.ndarray]:
    """Yield partitions with a gap between all within- and between-group angles."""
    labels = np.zeros(len(angles), dtype=int)
    nearest = angles[0].copy()
    # If every inter-group distance exceeds every group diameter, farthest-first
    # visits each group once before revisiting any. Test the partition at each size.
    for label in range(1, len(angles) // 3):
        center = int(nearest.argmax())
        spacing = nearest[center]
        if spacing <= floor:
            break
        closer = angles[center] < nearest
        labels[closer] = label
        nearest[closer] = angles[center, closer]
        sizes = np.bincount(labels)
        if sizes.min() < 3:
            break  # An existing center can only lose members to later centers.
        if sizes.max() <= layer_count / 2 or sizes.max() < 8 or spacing <= factor * max(floor, nearest.max()):
            continue
        same = labels[:, None] == labels
        within = angles[same].max()
        if within <= math.acos(0.9) and angles[~same].min() > factor * max(floor, within):
            yield labels.copy()


def _trajectory_groups(ids: list[int], angles: np.ndarray, layer_count: int,
                       factor: float, floor: float) -> Iterator[tuple[np.ndarray, float]]:
    """Find angularly isolated groups whose successive observations follow a slow path."""
    # Single linkage lets a path accumulate a large total rotation. At each
    # merge the shortest edge between components is the separation of the
    # current partition. Visit each pair once, checking only gaps in that scale.
    left, right = np.triu_indices(len(ids), 1)
    distances = angles[left, right]
    labels = np.arange(len(ids))
    layers = np.asarray(ids)
    previous = floor
    for edge in np.argsort(distances, kind="stable"):
        a, b = labels[left[edge]], labels[right[edge]]
        if a == b:
            continue
        separation = distances[edge]
        if separation > factor * previous:
            _, groups, sizes = np.unique(labels, return_inverse=True, return_counts=True)
            if sizes.min() >= 3 and sizes.max() >= 8 and sizes.max() > layer_count / 2:
                # A chain of close points in arbitrary order need not be
                # smooth in layer order. Include the return to a group after
                # an intervening run, using its angular displacement per layer.
                step = 0.
                for group in range(len(sizes)):
                    members = np.flatnonzero(groups == group)
                    rates = angles[members[:-1], members[1:]] / np.diff(layers[members])
                    step = max(step, float(rates.max()))
                if step <= math.acos(0.9) and separation > factor * max(floor, step):
                    yield groups, step
        labels[labels == b] = a
        previous = max(floor, separation)


def _subspace_groups(ids: list[int], angles: np.ndarray, layer_count: int, factor: float,
                     floor: float, extent: tuple[int, int]) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Visit supported partitions from fine to coarse, with their angular resolution."""
    coherent = list(_coherent_groups(angles, layer_count, factor, floor))
    for labels in reversed(coherent):
        within = angles[labels[:, None] == labels].max()
        yield labels, np.full(labels.max() + 1, within), "maximum within-group angle"
    if coherent:
        return
    found = False
    for labels, step in _trajectory_groups(ids, angles, layer_count, factor, floor):
        found = True
        yield labels, np.full(labels.max() + 1, step), "maximum within-group step per layer"
    if not found:
        cliques = _isolated_cliques(ids, angles, layer_count, factor, floor, extent)
        if cliques is not None:
            yield cliques


def _isolated_cliques(ids: list[int], angles: np.ndarray, layer_count: int, factor: float,
                      floor: float, extent: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Identify compact groups with exceptional boundary steps against wandering peers."""
    radius = math.acos(0.9)
    adjacent = np.diff(ids) == 1
    steps = angles.diagonal(offset=1)
    agreement = angles <= radius
    labels = np.zeros(len(ids), dtype=int)
    diameters = [0.]  # Label zero collects the background observations.
    for i, members in enumerate(agreement):
        # In an isolated clique, every member has exactly the same neighbors.
        # Check it once, even if its members occupy separate layer ranges.
        if members[:i].any() or not 3 <= members.sum() < len(ids) / 2:
            continue
        if not np.all(agreement[members] == members):
            continue
        diameter = angles[np.ix_(members, members)].max()
        separation = angles[np.ix_(members, ~members)].min()
        if separation <= factor * max(floor, diameter):
            continue
        observed = _observed_runs(ids, members, adjacent, extent)
        if not observed.any():
            continue
        boundaries = observed[:-1] != observed[1:]
        background = steps[adjacent & ~members[:-1] & ~members[1:]]
        if len(background) < 8:
            continue
        # A pause on a smooth path can have zero diameter. Require its entry
        # and exit to exceed the peer step distribution outside the run too.
        peers = Peers.summarize(background.tolist())
        resolution = max(floor, diameter)
        if peers.median <= factor * resolution or not peers.high(steps[boundaries].min(), resolution):
            continue
        labels[members] = len(diameters)
        diameters.append(diameter)
    background_count = np.count_nonzero(labels == 0)
    if len(diameters) == 1 or background_count < 8 or background_count <= layer_count / 2:
        return None
    return labels, np.asarray(diameters), "maximum within-clique angle"


def _stable_angles(angles: np.ndarray, members: np.ndarray, background: np.ndarray,
                   adjacent: np.ndarray, factor: float, floor: float,
                   sampled: np.ndarray | None = None, max_boundary_step: float = math.inf) -> float | None:
    """Check this candidate against the majority and its observed boundary neighbors."""
    within = (members[:, None] & members) | (background[:, None] & background)
    cross = members[:, None] & background
    # Consecutive sampled axes can be several layers apart. Only measure a
    # within-regime rate if every intervening layer belongs to the same run.
    runs = np.r_[0, np.cumsum(~(adjacent & within.diagonal(offset=1)))]
    if sampled is None:
        ids = np.arange(len(angles))
    else:
        pairs = sampled[:, None] & sampled
        within &= pairs
        cross &= pairs
        ids = np.flatnonzero(sampled)
    spread = angles[within].max(initial=0.)
    across = angles[cross].max(initial=0.)
    if not (spread <= math.acos(0.9) and across <= math.acos(0.9)):
        return None
    if np.any(np.triu(within, k=1)) and across > factor * max(spread, floor):
        return None
    a, b = ids[:-1], ids[1:]
    same_run = runs[a] == runs[b]
    boundary = (b == a + 1) & (members[a] != members[b])
    boundary_limit = min(math.acos(0.9), max_boundary_step)
    if sampled is None or same_run.any():
        rates = angles[a[same_run], b[same_run]] / (b - a)[same_run]
        boundary_limit = min(boundary_limit, factor * rates.max(initial=floor))
    if not np.all(angles[a[boundary], b[boundary]] <= boundary_limit):
        return None
    return max(spread, across)


def _principal_cosines(bases: list[np.ndarray]) -> np.ndarray:
    """Descending principal cosines. Each pair has min(rank_i, rank_j) measurements."""
    dim = max(q.shape[1] for q in bases)
    if dim == 1:
        axes = np.stack([q[:, 0] for q in bases])
        return np.abs(axes @ axes.T)[None]
    overlap = np.full((dim, len(bases), len(bases)), np.nan)
    for i, basis in enumerate(bases):
        overlap[:basis.shape[1], i, i] = 1.
        for j in range(i):
            values = np.linalg.svd(basis.T @ bases[j], compute_uv=False)
            overlap[:len(values), i, j] = overlap[:len(values), j, i] = values
    return overlap


class _OutputGeometry(NamedTuple):
    bases: list[np.ndarray]
    cosines: np.ndarray
    leading_axes: np.ndarray
    leading_angles: np.ndarray
    extent: tuple[int, int]


def _shared_space_switch(geometry: _OutputGeometry, members: np.ndarray,
                         background: np.ndarray, adjacent: np.ndarray,
                         resolution: float, factor: float, floor: float) -> str | None:
    """Confirm a common intersection and a separated complement at the observed boundaries."""
    neighbors = np.r_[False, members[:-1]] | np.r_[members[1:], False]
    reference = (background | neighbors) & ~members
    relevant = members | reference
    cross = geometry.cosines[:, members][:, :, reference]
    cross = cross[np.isfinite(cross).all(axis=(1, 2))]
    widest = np.arccos(np.clip(cross.min(axis=(1, 2)), 0., 1.))
    k = int(np.count_nonzero(widest <= resolution))
    if k == 0 or k == len(widest):
        return None
    separation = math.acos(min(1., cross[k].max()))
    if separation <= max(resolution, factor * widest[k - 1]):
        return None

    # Pairwise intersections can point in different directions. Within one
    # member's basis, find the k axes best retained by all relevant projectors.
    # Work with r-by-r matrices, independent of the output width.
    bases = geometry.bases
    anchor = bases[np.flatnonzero(members)[0]]
    active = np.flatnonzero(relevant)
    maps = [bases[i].T @ anchor for i in active]
    _, vectors = np.linalg.eigh(sum(m.T @ m for m in maps))
    common = {}
    for i, m in zip(active, maps):
        coordinates, retained, _ = np.linalg.svd(m @ vectors[:, -k:], full_matrices=False)
        if math.acos(min(1., retained[-1])) > resolution:
            return None
        common[i] = bases[i] @ coordinates
    shared_angles = np.zeros_like(geometry.leading_angles)
    shared_cosines = _principal_cosines(list(common.values()))[-1]
    shared_angles[np.ix_(active, active)] = np.arccos(np.clip(shared_cosines, 0., 1.))
    width = _stable_angles(shared_angles, members, background, adjacent, factor, floor)
    if width is None or separation <= factor * width:
        return None
    # Independently resolved axes must lie in the common space and must not
    # contradict its stability. Missing measurements supply no veto or vote.
    known = relevant & np.isfinite(geometry.leading_angles.diagonal())
    if known.any():
        alignment = min(np.linalg.norm(common[i].T @ geometry.leading_axes[i]) for i in np.flatnonzero(known))
        if (math.acos(min(1., alignment)) > min(resolution, math.acos(0.9))
                or _stable_angles(geometry.leading_angles, members, background, adjacent,
                                  factor, floor, sampled=known, max_boundary_step=separation / factor) is None):
            return None
    return (f"shared output dimension {k}, maximum shared-space angle {math.degrees(width):.2f} degrees, "
            f"minimum switched angle {math.degrees(separation):.2f} degrees")


def _subspace_switches(ids: list[int], names: list[str], layer_count: int, angles: np.ndarray,
                       overlap: np.ndarray, directions: dict[str, np.ndarray] | None,
                       bases: list[np.ndarray], factor: float, extent: tuple[int, int]) -> dict[str, str]:
    """Locate isolated changes in all modes or in the complement of a stable edit space."""
    width = bases[0].shape[0]
    floor = math.sqrt(2 * width * math.ulp(1.))
    factor = max(1., factor)
    adjacent = np.diff(ids) == 1
    findings = {}
    axes = np.full((len(names), width), np.nan)
    for i, name in enumerate(names):
        if directions and name in directions and directions[name].shape == (width,):
            axes[i] = directions[name]
    geometry = _OutputGeometry(bases, overlap, axes, np.arccos(np.clip(np.abs(axes @ axes.T), 0., 1.)), extent)
    partitions = _subspace_groups(ids, angles, layer_count, factor, floor, extent)
    for i, (labels, widths, spread) in enumerate(partitions):
        # A resolved leading axis uses the finest partition. A coarser one
        # can establish a collective switch by separating every changing mode.
        found = _partition_switches(ids, names, angles, geometry, factor, floor, adjacent,
                                    labels, widths, spread, i == 0)
        # A supported coarser partition can explain neighboring subgroups as
        # one switch. Report the collective evidence for those layers.
        findings.update(found)
    return findings


def _independent_mode_switch(cosines: np.ndarray, angles: np.ndarray, members: np.ndarray,
                             same: np.ndarray, adjacent: np.ndarray, switches: np.ndarray,
                             limit: float, floor: float) -> tuple[str, str] | None:
    """Test complete separation or at least two independently moving paired modes."""
    outside = members[:, None] & ~members
    # Compare each moving mode at every boundary with ordinary outside steps.
    steps = angles.diagonal(1)[adjacent & same.diagonal(1) & ~members[:-1]]
    reference = "outside step limit"
    if len(steps) >= 8:
        peers = Peers.summarize(steps.tolist())
        step_limit = peers.median + max(floor, 6 * 1.4826 * peers.mad)
    elif len(steps):
        step_limit = steps.max() + floor
    else:
        # In an alternating partition every adjacent pair may switch groups.
        # Use the full observed diameter of each outside regime as an upper
        # bound on movement within it, even across nonconsecutive layers.
        step_limit = angles[same & ~members[:, None]].max() + floor
        reference = "outside within-regime diameter plus tolerance"
    boundary_cosines = cosines.diagonal(1, axis1=1, axis2=2)[:, switches]
    cross = cosines[:, outside]
    closest = math.acos(min(1., cross[0].max()))
    boundary = math.acos(min(1., boundary_cosines[0].max()))
    lines = np.isnan(cosines[1].diagonal())
    if closest > limit and boundary > step_limit and not (outside & lines[:, None] & lines).any():
        switch = "whole output subspace"
    else:
        # Count from the widest angle in each pair. Missing angles sort last
        # and cannot contribute evidence of an additional moving mode.
        widest = np.arccos(np.clip(np.sort(cross, axis=0).max(axis=1), 0., 1.))
        steps = np.arccos(np.clip(np.sort(boundary_cosines, axis=0).max(axis=1), 0., 1.))
        moved = np.flatnonzero((widest > limit) & (steps > step_limit))
        if len(moved) < 2:
            return None
        closest, boundary = widest[moved[-1]], steps[moved[-1]]
        switch = f"{len(moved)} output modes"
    return switch, (f"smallest cross-group principal angle among moving modes {math.degrees(closest):.2f} degrees, "
                    f"minimum boundary angle {math.degrees(boundary):.2f} degrees "
                    f"vs {reference} {math.degrees(step_limit):.2f} degrees")


def _observed_runs(ids: list[int], members: np.ndarray, adjacent: np.ndarray,
                   extent: tuple[int, int]) -> np.ndarray:
    """Members of physical runs with a measurement on each outer boundary."""
    starts = np.flatnonzero(members & np.r_[True, ~(adjacent & members[:-1])])
    ends = np.flatnonzero(members & np.r_[~(adjacent & members[1:]), True])
    observed = np.zeros_like(members)
    for start, end in zip(starts, ends):
        left = adjacent[start - 1] if start else ids[0] == extent[0]
        right = adjacent[end] if end < len(adjacent) else ids[-1] == extent[1]
        if left and right:
            observed[start:end + 1] = True
    # A single terminal transition requires a continuous reference trajectory.
    if np.count_nonzero(observed[:-1] != observed[1:]) == 1 and not adjacent.all():
        observed[:] = False
    return observed


def _partition_switches(ids: list[int], names: list[str], angles: np.ndarray, geometry: _OutputGeometry,
                        factor: float, floor: float, adjacent: np.ndarray,
                        labels: np.ndarray, widths: np.ndarray, spread: str, finest: bool) -> dict[str, str]:
    sizes = np.bincount(labels)
    background = labels == sizes.argmax()
    same = labels[:, None] == labels
    findings = {}
    for label in np.flatnonzero(sizes < sizes.max()):
        within = widths[label]
        members = labels == label
        observed = _observed_runs(ids, members, adjacent, geometry.extent)
        if not observed.any():
            continue
        switches = observed[:-1] != observed[1:]
        outside = members[:, None] & ~members
        limit = factor * max(floor, within)
        # A stable primary or intersection gives a more specific attribution.
        # If it cannot be established, two exceptional principal angles still
        # certify movement that changing a single leading axis cannot explain.
        switch = "secondary subspace"
        primary = _stable_angles(geometry.leading_angles, members, background, adjacent, factor, floor)
        terminal = switches.sum() == 1
        second = geometry.cosines[1]
        secondary = None
        if np.isfinite(second.diagonal(1)[switches]).all():
            # Every boundary measured a second mode. Other pairs can bound
            # its separation only where both writers resolved that mode.
            secondary = math.acos(min(1., np.nanmax(second[outside])))
        if finest and primary is not None and secondary is not None and (not terminal or secondary > limit):
            across = angles[outside].min()
            evidence = (f"minimum between-group angle {math.degrees(across):.2f} degrees, "
                        f"leading axes within {math.degrees(primary):.2f} degrees")
            if terminal:
                i = int(np.flatnonzero(switches)[0])
                evidence += (f", minimum secondary angle {math.degrees(secondary):.2f} degrees "
                             f"at boundary {ids[i]}–{ids[i+1]}")
        else:
            evidence = _shared_space_switch(geometry, members, background, adjacent, limit, factor, floor)
            if evidence is None:
                independent = _independent_mode_switch(geometry.cosines, angles, members, same, adjacent,
                                                       switches, limit, floor)
                if independent is None:
                    continue
                switch, evidence = independent
        reason = (f"abrupt {switch} switch in {observed.sum()} layers "
                  f"({evidence}, {spread} {math.degrees(within):.2f} degrees)")
        for i in np.flatnonzero(observed):
            findings[names[i]] = reason
    return findings


def _subspace_cohorts(rows: list[InventoryRow], subspaces: dict[str, np.ndarray],
                      directions: dict[str, np.ndarray] | None):
    """Compare each measured rank with all layers that resolve at least that many modes."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        kind = canonical_kind(r.kind)
        if r.name in subspaces and r.layer is not None and kind in RESIDUAL_WRITERS:
            key = (stack_of(r.name), kind, r.base_shape, r.base_dtype, r.suspect_dtype, subspaces[r.name].shape[0])
            groups[key][r.layer].append(r.name)

    for layers in groups.values():
        # One independent vote per layer. Ambiguous layers (e.g. several
        # same-kind experts in one block) do not define a reference trajectory.
        ids = sorted(i for i in layers if len(layers[i]) == 1)
        if len(ids) < 9:
            continue
        names = [layers[i][0] for i in ids]
        bases = [subspaces[name] for name in names]
        ranks = np.array([q.shape[1] for q in bases])
        if ranks.max() == 1 and directions and all(n in directions for n in names):
            continue  # The direction signal already calibrates this pure leading-axis cohort.
        overlap = _principal_cosines(bases)
        extent = min(layers), max(layers)
        yield ids, names, bases, overlap, len(layers), extent, True
        # Containment agrees only on the paired modes. A line inside two
        # distinct planes supplies no vote about their secondary directions.
        # Retain physical layer IDs so missing modes cannot invent a boundary.
        for rank in np.unique(ranks)[1:]:
            keep = np.flatnonzero(ranks >= rank)
            if len(keep) < 9:
                break
            count = sum(any(subspaces[n].shape[1] >= rank for n in group) for group in layers.values())
            yield ([ids[i] for i in keep], [names[i] for i in keep], [bases[i] for i in keep],
                   overlap[:, keep[:, None], keep], count, extent, False)


def _secondary_departures(angles: np.ndarray, names: list[str], bases: list[np.ndarray],
                          directions: dict[str, np.ndarray]) -> np.ndarray:
    """Which pairwise departures persist when accounting for resolved primary axes?"""
    secondary = []
    for name, q in zip(names, bases):
        axis = directions.get(name)
        if axis is None or axis.shape != (q.shape[0],):
            secondary.append(q)
            continue
        coordinates = q.T @ axis
        if np.linalg.norm(coordinates) < 0.9:
            secondary.append(q)
            continue
        rotation, _ = np.linalg.qr(coordinates[:, None], mode="complete")
        secondary.append(q @ rotation[:, 1:])
    if all(a is b for a, b in zip(secondary, bases)):
        return angles
    cosines = np.nanmin(_principal_cosines(secondary), axis=0)
    return np.minimum(angles, np.arccos(np.clip(cosines, 0., 1.)))


def subspace_outliers(rows: list[InventoryRow], subspaces: dict[str, np.ndarray], factor: float,
                      stats: dict[str, Measurements] | None = None,
                      directions: dict[str, np.ndarray] | None = None,
                      direction_flags: dict[str, str] | None = None) -> dict[str, str]:
    """Refine leading-axis findings using the resolved spaces in the same output coordinates."""
    leading = dict(direction_flags or {})
    findings = {}
    for ids, names, bases, overlap, layer_count, extent, full in _subspace_cohorts(rows, subspaces, directions):
        width = bases[0].shape[0]
        ranks = np.array([q.shape[1] for q in bases])
        angles = np.arccos(np.clip(np.nanmin(overlap, axis=0), 0., 1.))
        np.fill_diagonal(angles, 0.)
        if len(overlap) > 1:
            findings.update(_subspace_switches(ids, names, layer_count, angles, overlap,
                                               directions, bases, factor, extent))
        if not full and directions:
            # Dropping lower ranks changes the leading-axis reference population.
            # Require the higher-rank outlier to differ in its additional modes.
            angles = _secondary_departures(angles, names, bases, directions)
        agreement = angles <= math.acos(0.9)
        supporters = agreement[np.argmax(agreement.sum(axis=1))]
        if supporters.sum() <= layer_count / 2:
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
                        findings[names[i]] = (f"recurring {ranks[i]}-dimensional output subspace "
                                              f"in {len(member_ids)} layers without a comparable peer group ({shift})")
            continue
        # Compare each candidate with the coherent population excluding itself.
        np.fill_diagonal(angles, np.nan)
        separation = np.nanmedian(angles[:, supporters], axis=1)
        for i, name in enumerate(names):
            peers = Peers.summarize(np.delete(separation, i).tolist())
            limit = max(max(1., factor) * peers.median, peers.median + 6 * 1.4826 * peers.mad,
                        math.sqrt(2 * width * math.ulp(1.)))
            if full and name in leading:
                # A change of resolved rank can change which axis leads. Count
                # containment votes from measured spaces even if their leading
                # axes were tied or unavailable.
                contained = supporters & (ranks != ranks[i]) & agreement[i] & (angles[i] <= limit)
                if contained.sum() > (layer_count - 1) / 2:
                    del leading[name]
            if name in findings or separation[i] <= limit:
                continue
            # Several mutually aligned layers can represent another editing
            # concept. Demand spectral corroboration for an isolated cohort.
            members = agreement[i]
            if (stats is not None and members.sum() >= 3 and not np.any(members & supporters)
                    and np.all(agreement[members] == members)):
                if not _recurring_rank_shift(names, members, stats):
                    continue
            else:
                # At a plateau edge a constant-speed prediction can fail even
                # though two consecutive peers confirm the measured space.
                matching = np.flatnonzero(members & (angles[i] <= limit))
                if _on_plateau(ids[i], {ids[j] for j in matching}):
                    continue
            # Allow smooth drift between the measured spaces.
            # A constant-speed path through the adjacent layers covers each
            # interval in proportion to its length, also at either endpoint.
            other = [j for j in range(len(ids)) if j != i]
            start = max(0, min(i - 1, len(other) - 2))
            a, b = other[start:start + 2]
            t = (ids[i] - ids[a]) / (ids[b] - ids[a])
            if (-1 <= t <= 2 and agreement[a, b] and angles[i, a] <= abs(t) * angles[a, b] + limit
                    and angles[i, b] <= abs(1 - t) * angles[a, b] + limit):
                continue
            findings[name] = (f"{ranks[i]}-dimensional output subspace differs from coherent layers "
                              f"(largest principal angle, median {math.degrees(separation[i]):.2f} degrees "
                              f"vs peer upper bound {math.degrees(limit):.2f})")
    return leading | findings


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
                       cosines: dict[str, float], factor: float,
                       subspaces: dict[str, np.ndarray] | None = None) -> dict[str, str]:
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
                if subspaces and name in subspaces and layer in axes:
                    # A sustained multi-concept edit may change both its leading
                    # axis and full edit space. Require support for both locally.
                    space = subspaces[name]
                    matching = set()
                    for j in (layer - 2, layer - 1, layer + 1, layer + 2):
                        if j not in axes or _orthogonal_fraction(float(axes[j] @ axes[layer])) > limit:
                            continue
                        peer = subspaces.get(layers[j][0])
                        if (peer is not None and peer.shape[0] == space.shape[0]
                                and max(peer.shape[1], space.shape[1]) > 1):
                            if np.linalg.svd(space.T @ peer, compute_uv=False)[-1] >= 0.9:
                                matching.add(j)
                    if _on_plateau(layer, matching):
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
