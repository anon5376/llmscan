"""Peer comparisons for changed tensors of an expected kind."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from llmscan.inventory import InventoryRow
from llmscan.names import EMBEDDINGS, RESIDUAL_WRITERS, canonical_kind, stack_of


class Measurements(NamedTuple):
    effective_rank: float | None = None
    sv_001: float | None = None
    frobenius: float | None = None
    projection_gain: float | None = None
    second_tail_share: float | None = None
    token_row_fraction: float | None = None


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
        # A gain measures the leading change relative to the base's weight in
        # that direction. With older CSVs only raw magnitudes are available.
        for metric in ("projection_gain", "frobenius", "sv_001"):
            magnitude = getattr(m, metric)
            if magnitude is not None:
                groups[kind, r.base_shape, metric].append(math.log(magnitude))
    return {group: Peers.summarize(v) for group, v in groups.items() if len(v) >= 3}


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

    # A local magnitude bound also works for a singleton or a shifted peer
    # median: removal of a unit direction can account for at most one full
    # projection. Allow the same factor of slack as the coarse rank check.
    over_projection = (m.projection_gain is not None and kind in RESIDUAL_WRITERS | EMBEDDINGS
                       and m.projection_gain > max(1., factor))
    if over_projection:
        reasons.append(f"projection_gain {m.projection_gain:.4g} exceeds a full projection of the base "
                       f"(threshold {max(1., factor):g}x)")
    for metric in ("projection_gain", "frobenius", "sv_001"):
        magnitude = getattr(m, metric)
        group_stats = peers.get((kind, r.base_shape, metric))
        if group_stats and group_stats.count >= 8 and magnitude is not None:
            if (not over_projection and group_stats.high(math.log(magnitude), math.log(2))
                    and (metric != "projection_gain" or magnitude > 1.05)):
                reasons.append(f"{metric} {magnitude:.4g} above peer dispersion "
                               f"(median {math.exp(group_stats.median):.4g})")
            break
    if kind in EMBEDDINGS:
        if m.token_row_fraction is not None and m.token_row_fraction < 0.05:
            reasons.append(f"token-row energy concentrated in {m.token_row_fraction:.2%} effective rows")
    if cosine is not None and cosine < 0.5:
        reasons.append(f"residual direction disagrees with coherent peer layers (absolute cosine {cosine:.3f})")
    return reasons


def direction_cosines(rows: list[InventoryRow], directions: dict[str, np.ndarray]) -> dict[str, float]:
    """Compare to other layers only. Sign is arbitrary. Give each layer one vote.

    With no tightly aligned majority (e.g. different legitimate axes across
    layers), abstain. Embeddings can be judged but never set the reference.
    """
    spaces = defaultdict(list)
    for r in rows:
        if r.name in directions:
            spaces[stack_of(r.name), len(directions[r.name])].append(r)
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
