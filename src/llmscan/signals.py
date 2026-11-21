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

    @classmethod
    def summarize(cls, values: list[float]) -> Peers:
        median = statistics.median(values)
        return cls(len(values), median, statistics.median(abs(x - median) for x in values))

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
    if residual is not None and m.projection_error is not None and m.projection_error > 0.05:
        fit_group = (kind, r.base_shape, r.base_dtype, r.suspect_dtype)
        fits = peers.get((*fit_group, "projection_error"))
        unusual_fit = fits is None or fits.count < 8 or fits.high(m.projection_error, 0.05)
        # Comparable layers measure how much of a rounding/healing residual
        # normally lies along the leading axis. Give correlated tests one vote.
        shares = peers.get((*fit_group, "leading_share"))
        share = residual.leading_share()
        if unusual_fit and shares and share is not None and shares.high(share, 0.05):
            return (f"projection law leaves {m.projection_error:.1%} of the leading change unexplained, and "
                    f"its axis carries {share:.1%} of the residual norm (peer median {shares.median:.1%})")
        if unusual_fit and residual.leading > factor * residual.tail:
            return (f"projection law leaves {m.projection_error:.1%} of the leading change unexplained "
                    "after fitting partial removal against the base")
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
                    cosine: float | None, factor: float) -> list[str]:
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
    if cosine is not None and cosine < 0.5:
        reasons.append(f"residual direction disagrees with coherent peer layers (absolute cosine {cosine:.3f})")
    return reasons


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
