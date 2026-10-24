"""Tensor inventory: which tensors exist in base and suspect, and which are bit-identical."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

from llmscan.names import kind_of, layer_of
from llmscan.safetensors_io import Checkpoint

# status values, in the order they are reported
IDENTICAL = "identical"
CHANGED = "changed"
DTYPE_MISMATCH = "dtype_mismatch"
SHAPE_MISMATCH = "shape_mismatch"
ONLY_BASE = "only_base"
ONLY_SUSPECT = "only_suspect"
STATUSES = (IDENTICAL, CHANGED, DTYPE_MISMATCH, SHAPE_MISMATCH, ONLY_BASE, ONLY_SUSPECT)


@dataclass
class InventoryRow:
    name: str
    layer: int | None
    kind: str
    status: str
    base_dtype: str
    suspect_dtype: str
    base_shape: str
    suspect_shape: str
    numel: int


def natural_key(name: str) -> list:
    """Sort `layers.2` before `layers.10`."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def bytes_equal(base: Checkpoint, suspect: Checkpoint, name: str) -> bool:
    for a, b in zip_longest(base.iter_bytes(name), suspect.iter_bytes(name)):
        if a != b:
            return False
    return True


def compare(base: Checkpoint, suspect: Checkpoint) -> list[InventoryRow]:
    """One row per tensor name found in either checkpoint, in natural name order.

    Same name, dtype and shape -> byte comparison decides identical vs changed. A dtype
    mismatch (e.g. a bf16 base re-saved as fp16) makes byte comparison meaningless, so it
    is reported as its own status.
    """
    rows = []
    for name in sorted(base.tensors.keys() | suspect.tensors.keys(), key=natural_key):
        b, s = base.tensors.get(name), suspect.tensors.get(name)
        if b is None:
            status = ONLY_SUSPECT
        elif s is None:
            status = ONLY_BASE
        elif b.shape != s.shape:
            status = SHAPE_MISMATCH
        elif b.dtype != s.dtype:
            status = DTYPE_MISMATCH
        else:
            status = IDENTICAL if bytes_equal(base, suspect, name) else CHANGED
        rows.append(InventoryRow(
            name=name,
            layer=layer_of(name),
            kind=kind_of(name),
            status=status,
            base_dtype=b.dtype if b else "",
            suspect_dtype=s.dtype if s else "",
            base_shape="x".join(map(str, b.shape)) if b else "",
            suspect_shape="x".join(map(str, s.shape)) if s else "",
            numel=(b or s).numel,
        ))
    return rows


def write_csv(rows: list[InventoryRow], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(InventoryRow.__dataclass_fields__))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def summarize(rows: list[InventoryRow]) -> str:
    """Plain-text summary: totals, a per-kind status table, then every structural mismatch."""
    counts = {s: 0 for s in STATUSES}
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in STATUSES})
    identical_params = total_params = 0
    for r in rows:
        counts[r.status] += 1
        by_kind[r.kind][r.status] += 1
        total_params += r.numel
        identical_params += r.numel if r.status == IDENTICAL else 0

    lines = [
        f"tensors: {len(rows)}  " + "  ".join(f"{s}={n}" for s, n in counts.items() if n),
        f"parameters bit-identical to base: {identical_params / max(total_params, 1):.2%}",
        "",
    ]
    shown = [s for s in STATUSES if counts[s]]
    width = max(len("kind"), *(len(k) for k in by_kind))
    lines.append(f"{'kind':<{width}}  " + "  ".join(f"{s:>{max(len(s), 5)}}" for s in shown))
    for kind in sorted(by_kind, key=lambda k: (-by_kind[k][CHANGED], k)):
        lines.append(f"{kind:<{width}}  " + "  ".join(f"{by_kind[kind][s]:>{max(len(s), 5)}}" for s in shown))

    mismatches = [r for r in rows if r.status not in (IDENTICAL, CHANGED)]
    if mismatches:
        lines += ["", "structural mismatches:"]
        lines += [
            f"  {r.status:<14} {r.name}  base={r.base_dtype}[{r.base_shape}]  suspect={r.suspect_dtype}[{r.suspect_shape}]"
            for r in mismatches
        ]
    return "\n".join(lines)
