"""Figures from delta.csv: singular-value spectra per matrix kind, effective rank vs layer.

Static PNGs for the report and README; delta.csv is the table view behind them.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from llmscan.delta import sv_field

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER = "#898781"
# fixed categorical slots, in validated order; a kind keeps its slot across every run
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
KIND_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"]
# sequential blue, ordinal range (step 250 -> 700): early layers light, late layers dark
LAYER_CMAP = LinearSegmentedColormap.from_list(
    "layers", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def kind_colors(kinds: list[str]) -> dict[str, str]:
    """Known kinds take their fixed slot; unknown kinds fill the remaining slots, then gray."""
    colors = {k: SERIES[i] for i, k in enumerate(KIND_ORDER) if k in kinds}
    free = [c for c in SERIES if c not in colors.values()]
    for k in sorted(set(kinds) - colors.keys()):
        colors[k] = free.pop(0) if free else OTHER
    return colors


def _sv(row: dict) -> list[float]:
    out = []
    i = 1
    while (v := row.get(sv_field(i))) not in (None, ""):
        out.append(float(v))
        i += 1
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _changed(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["effective_rank"] not in (None, "")]


def _unchanged_kinds(rows: list[dict]) -> list[str]:
    changed = {r["kind"] for r in _changed(rows)}
    return sorted({r["kind"] for r in rows} - changed)


def plot_spectra(rows: list[dict], path: Path, title: str) -> None:
    """One panel per changed matrix kind; one line per layer, light (early) to dark (late)."""
    changed = _changed(rows)
    by_kind = defaultdict(list)
    for r in changed:
        by_kind[r["kind"]].append(r)
    kinds = sorted(by_kind, key=lambda k: (KIND_ORDER.index(k) if k in KIND_ORDER else 99, k))
    if not kinds:
        return
    layers = [int(r["layer"]) for r in changed if r["layer"] not in (None, "")]
    lo, hi = (min(layers), max(layers)) if layers else (0, 1)

    cols = min(len(kinds), 3)
    nrows = math.ceil(len(kinds) / cols)
    fig = Figure(figsize=(4.2 * cols, 3.2 * nrows + 0.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, cols, squeeze=False, sharey=True)
    for ax, kind in zip(axes.flat, kinds):
        _style(ax)
        for r in sorted(by_kind[kind], key=lambda r: int(r["layer"] or -1)):
            sv = _sv(r)
            if r["layer"] in (None, ""):
                color = INK_2
            else:
                color = LAYER_CMAP((int(r["layer"]) - lo) / max(hi - lo, 1))
            ax.plot(range(1, len(sv) + 1), [max(s, 1e-12) for s in sv], color=color, linewidth=1.2)
        ax.set_yscale("log")
        n = len(by_kind[kind])
        ax.set_title(f"{kind}  ({n} tensor{'s' * (n != 1)})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("singular value index i")
    for ax in axes[:, 0]:
        ax.set_ylabel("σᵢ(ΔW)")
    for ax in list(axes.flat)[len(kinds):]:
        ax.set_visible(False)
    if layers:
        sm = fig.colorbar(ScalarMappable(cmap=LAYER_CMAP, norm=Normalize(lo, hi)), ax=axes, shrink=0.8, pad=0.01)
        sm.set_label("layer", color=INK_2)
        sm.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
        sm.outline.set_edgecolor(AXIS)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_effective_rank(rows: list[dict], path: Path, title: str) -> None:
    """Effective rank of ΔW against layer, one line per matrix kind."""
    changed = [r for r in _changed(rows) if r["layer"] not in (None, "")]
    by_kind = defaultdict(list)
    for r in changed:
        by_kind[r["kind"]].append((int(r["layer"]), float(r["effective_rank"])))
    fig = Figure(figsize=(7.5, 4.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    _style(ax)
    colors = kind_colors(list(by_kind))
    for kind in sorted(by_kind, key=lambda k: (KIND_ORDER.index(k) if k in KIND_ORDER else 99, k)):
        pts = sorted(by_kind[kind])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[kind], linewidth=1.5,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=kind)
    values = [v for pts in by_kind.values() for _, v in pts]
    if values and min(values) > 0 and max(values) / min(values) > 30:
        ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("effective rank of ΔW")
    if by_kind:
        leg = ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        for t in leg.get_texts():
            t.set_color(INK_2)
    unchanged = _unchanged_kinds(rows)
    if unchanged:
        ax.text(0.0, -0.2, "bit-identical to base: " + ", ".join(unchanged), transform=ax.transAxes,
                color=MUTED, fontsize=8, va="top", wrap=True)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
