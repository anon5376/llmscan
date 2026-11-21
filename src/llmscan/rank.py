"""Rank suspect tensors, most suspicious first, from the inventory and (when present) delta.csv.

Identical tensors are not ranked; a float re-save with identical values is already `identical`
in the inventory. Every ranked tensor lands in one tier and carries a human-readable `reason`.

    T1 structure     present in only one model, shape mismatch, or a non-float dtype mismatch.
                     A backdoor can add or resize tensors; an honest abliteration cannot.
    T2 location      changed, but its kind is outside the expected set. Defaults: o_proj, down_proj,
                     embed_tokens (HF) and attn_output, ffn_down, token_embd (GGUF); MoE expert and
                     shared-expert down projections count as down_proj. q/k/v, gate/up, norms,
                     biases and an untied lm_head are T2.
    T3 outlier       changed, expected kind, and its effective_rank is more than `factor` times the
                     median of the changed tensors of the same kind in this model (default 3x).
                     Needs delta.csv and at least three changed tensors of that kind with a
                     positive effective_rank; a tensor without one is judged by sv_001 against its
                     kind's median the same way. The median, not a fixed rank, is the yardstick:
                     Heretic's rank-3 edits and norm-preserving variants are honest. It assumes
                     fewer than half of a kind's changed tensors are tampered with.
    T4 consistent    every other changed tensor.

`delta` is None when no ΔW scan ran (--inventory-only, stub core): every other changed tensor is T4
"(no ΔW stats)". Any mapping, even an empty one, means the scan ran. Then a changed expected-kind tensor
whose stats are unusable is T3 with reason "stats unusable: <why>", never T4: no delta row (a tensor the
2-D float scanner skips says "not scanned (<why>)"), both cells blank, or any given effective_rank / sv_001
that is NaN, infinite, zero or negative. An unusable row is excluded as a whole from every median.

Inside a tier: effective_rank, then sv_001, both descending; tensors without usable stats follow in
natural name order, then raw name. Unusable-stats T3 rows come after the genuine outliers.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from llmscan.csv_safety import csv_fieldnames, decode_row, encode_row
from llmscan.delta import sv_field
from llmscan.inventory import CHANGED, IDENTICAL, ONLY_BASE, ONLY_SUSPECT, SHAPE_MISMATCH, InventoryRow, natural_key
from llmscan.safetensors_io import FLOAT_DTYPES

T1, T2, T3, T4 = "T1", "T2", "T3", "T4"
TIERS = (T1, T2, T3, T4)
DEFAULT_EXPECTED_KINDS = frozenset({"o_proj", "down_proj", "embed_tokens", "attn_output", "ffn_down", "token_embd"})
DEFAULT_OUTLIER_FACTOR = 3.0
MIN_GROUP = 3  # changed tensors of one kind needed before a median means anything

SV1 = sv_field(1)
FIELDS = ["rank", "tier", "name", "layer", "kind", "status", "effective_rank", SV1, "reason"]


@dataclass
class RankedRow:
    rank: int
    tier: str
    name: str
    layer: int | None
    kind: str
    status: str
    effective_rank: float | None
    sv_001: float | None
    reason: str


def parse_kinds(text: str) -> frozenset[str]:
    kinds = frozenset(k.strip() for k in text.split(",") if k.strip())
    if not kinds:
        raise ValueError("expected kinds: give at least one, e.g. o_proj,down_proj")
    return kinds


def parse_factor(text: str) -> float:
    return check_factor(float(text))


def check_factor(factor: float) -> float:
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"outlier factor must be a finite number above 0, got {factor}")
    return factor


def canonical_kind(kind: str) -> str:
    """`experts.down_proj` (fused MoE) -> `down_proj`; GGUF `ffn_down_exps` / `ffn_down_shexp` -> `ffn_down`."""
    return re.sub(r"_(exps|shexp)$", "", kind.removeprefix("experts."))


def read_delta_csv(path: Path) -> dict[str, tuple[float | None, float | None]]:
    """name -> (effective_rank, sv_001); an empty cell (identical tensor) is None."""
    def num(cell: str | None) -> float | None:
        return float(cell) if cell not in (None, "") else None

    with open(path, newline="", encoding="utf-8") as f:
        return {r["name"]: (num(r.get("effective_rank")), num(r.get(SV1)))
                for r in map(decode_row, csv.DictReader(f))}


def _structure_reason(r: InventoryRow) -> str:
    if r.status == ONLY_BASE:
        return "in base but missing from suspect"
    if r.status == ONLY_SUSPECT:
        return "in suspect but not in base: a tensor the base does not have"
    if r.status == SHAPE_MISMATCH:
        return f"shape differs: base {r.base_shape} vs suspect {r.suspect_shape}"
    return f"dtype differs and is not a float re-save: base {r.base_dtype} vs suspect {r.suspect_dtype}"


def _unusable(stats: tuple[float | None, float | None]) -> str | None:
    """Why a delta.csv row cannot be judged, or None. Blank cells are fine as long as one stat is given."""
    er, sv = stats
    if er is None and sv is None:
        return "no ΔW stats for this tensor"
    for label, v in (("effective_rank", er), (SV1, sv)):
        if v is not None and not math.isfinite(v):
            return f"{label} {v} is not finite"
        if v is not None and v <= 0:
            return f"{label} {v:g} is not positive"
    return None


def _not_scanned(r: InventoryRow) -> str:
    """Why a changed tensor has no delta.csv row; mirrors what delta.scan visits (2-D, float on both sides)."""
    if r.base_shape.count("x") != 1:
        return f"not scanned (shape {r.base_shape or 'scalar'} is not 2-D)"
    for dtype in (r.base_dtype, r.suspect_dtype):
        if dtype not in FLOAT_DTYPES:
            return f"not scanned (dtype {dtype} is not float)"
    return "no ΔW stats for this tensor"


def rank_tensors(
    rows: list[InventoryRow],
    delta: dict[str, tuple[float | None, float | None]] | None = None,
    expected: frozenset[str] = DEFAULT_EXPECTED_KINDS,
    factor: float = DEFAULT_OUTLIER_FACTOR,
) -> list[RankedRow]:
    check_factor(factor)
    scanned = delta is not None  # an empty mapping is a scan that produced no rows, not "no scan"
    delta = delta or {}
    expected = frozenset(canonical_kind(k) for k in expected)
    stats = {r.name: delta.get(r.name, (None, None)) for r in rows}
    bad = {r.name: None if not scanned else _unusable(stats[r.name]) if r.name in delta else _not_scanned(r) for r in rows}

    # medians per kind over changed, expected-kind tensors; effective_rank and sv_001 kept apart
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        if r.status == CHANGED and canonical_kind(r.kind) in expected:
            er, sv = stats[r.name]
            if bad[r.name]:
                continue
            if er is not None:
                groups[canonical_kind(r.kind), 0].append(er)
            elif sv is not None:
                groups[canonical_kind(r.kind), 1].append(sv)
    medians = {g: statistics.median(v) for g, v in groups.items() if len(v) >= MIN_GROUP}

    ranked: list[tuple[str, InventoryRow, str]] = []
    unusable: set[str] = set()
    for r in rows:
        if r.status == IDENTICAL:
            continue
        kind = canonical_kind(r.kind)
        if r.status != CHANGED:
            ranked.append((T1, r, _structure_reason(r)))
        elif kind not in expected:
            ranked.append((T2, r, f"changed {r.kind}: outside expected kinds ({', '.join(sorted(expected))})"))
        else:
            er, sv = stats[r.name]
            if bad[r.name]:
                ranked.append((T3, r, f"stats unusable: {bad[r.name]}; cannot be judged consistent with abliteration"))
                unusable.add(r.name)
                continue
            which, value = (0, er) if er is not None else (1, sv)
            median = medians.get((kind, which))
            if value is not None and value > 0 and median and value > factor * median:
                label = "effective_rank" if which == 0 else SV1
                ranked.append((T3, r, f"{label} {value:.4g} is {value / median:.1f}x the median {median:.4g} of changed {kind}"
                                      f" tensors (threshold {factor:g}x)"))
            else:
                note = "" if scanned else " (no ΔW stats)"
                ranked.append((T4, r, f"changed {r.kind}, consistent with abliteration{note}"))

    def key(item):
        tier, r, _ = item
        er, sv = (v if v is not None and math.isfinite(v) and r.name not in unusable else None for v in stats[r.name])
        return (tier, r.name in unusable, er is None, -(er or 0.0), sv is None, -(sv or 0.0), natural_key(r.name), r.name)

    ranked.sort(key=key)
    return [
        RankedRow(i, tier, r.name, r.layer, r.kind, r.status, *stats[r.name], reason)
        for i, (tier, r, reason) in enumerate(ranked, 1)
    ]


def write_csv(ranked: list[RankedRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fieldnames(FIELDS))
        w.writeheader()
        for r in ranked:
            d = asdict(r)
            d["layer"] = "" if r.layer is None else r.layer
            for col in ("effective_rank", SV1):
                d[col] = "" if d[col] is None else f"{d[col]:.6g}"
            w.writerow(encode_row(d))


def summarize(ranked: list[RankedRow], top: int = 10) -> str:
    counts = {t: sum(r.tier == t for r in ranked) for t in TIERS}
    lines = ["ranking: " + "  ".join(f"{t}={n}" for t, n in counts.items()) + "  (identical tensors not ranked)"]
    lines += [f"{r.rank:>4}  {r.tier}  {r.name}  {r.reason}" for r in ranked[:top]]
    return "\n".join(lines)
