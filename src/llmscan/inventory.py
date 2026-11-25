"""Tensor inventory: which tensors exist in base and suspect, and which are bit-identical."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np

from llmscan.csv_safety import csv_fieldnames, encode_row
from llmscan.names import EMBEDDINGS, RESIDUAL_WRITERS, canonical_kind, kind_of, layer_of
from llmscan.safetensors_io import FLOAT_DTYPES, Checkpoint

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
    resave_dtype: str = ""


def natural_key(name: str) -> list:
    """Sort `layers.2` before `layers.10`."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def bytes_equal(base: Checkpoint, suspect: Checkpoint, name: str) -> bool:
    chunk = 64 << 10  # mmap slices are copies; keep the two byte streams' working set small
    for a, b in zip_longest(base.iter_bytes(name, chunk), suspect.iter_bytes(name, chunk)):
        if a != b:
            return False
    return True


def values_equal(base: Checkpoint, suspect: Checkpoint, name: str) -> bool:
    """Exact value comparison across float dtypes, chunk by chunk. Chunks are float64, which holds
    every supported float dtype exactly, so rounding can neither hide a difference nor invent one."""
    for a, b in zip(base.iter_values(name), suspect.iter_values(name)):
        if not np.array_equal(a, b):
            return False
    return True


def _bf16_resave(base: Checkpoint, suspect: Checkpoint, name: str) -> bool:
    """Compare every value after IEEE round-to-nearest, ties-to-even to BF16."""
    for a, b in zip(base.iter_values(name), suspect.iter_values(name), strict=True):
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return False
        bits = a.astype(np.float32).view(np.uint32)
        rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)
        if not np.array_equal(rounded, b) or not np.array_equal(np.signbit(rounded), np.signbit(b)):
            return False
    return True


def _mark_bf16_resave(rows: list[InventoryRow], base: Checkpoint, suspect: Checkpoint) -> None:
    # A location exception needs evidence of a common re-save across layers.
    # Check unchanged locations too: selective rounding is not a global cast.
    outside = [r for r in rows if canonical_kind(r.kind) not in RESIDUAL_WRITERS | EMBEDDINGS]
    changed = [r for r in outside if r.status == CHANGED]
    if len({r.layer for r in changed if r.layer is not None}) < 2:
        return
    for r in outside:
        if r.status not in (IDENTICAL, CHANGED) or r.base_dtype != r.suspect_dtype:
            return
        if r.base_dtype not in ("F16", "F32"):
            if r.status != IDENTICAL or r.base_dtype == "F64":
                return
            continue
        if not _bf16_resave(base, suspect, r.name):
            return
    for r in changed:
        r.resave_dtype = "BF16"


def compare(base: Checkpoint, suspect: Checkpoint) -> list[InventoryRow]:
    """One row per tensor name found in either checkpoint, in natural name order.

    Same name, dtype and shape -> byte comparison decides identical vs changed. Different
    float dtypes (e.g. a bf16 base re-saved as fp32) -> exact value comparison instead. Only
    non-float dtype mismatches get their own status.
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
            if b.dtype in FLOAT_DTYPES and s.dtype in FLOAT_DTYPES:
                status = IDENTICAL if values_equal(base, suspect, name) else CHANGED
            else:
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
    _mark_bf16_resave(rows, base, suspect)
    return rows


def write_csv(rows: list[InventoryRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fieldnames(list(InventoryRow.__dataclass_fields__)))
        w.writeheader()
        for r in rows:
            w.writerow(encode_row(asdict(r)))


def summarize(rows: list[InventoryRow]) -> str:
    """Plain-text summary: totals, a per-kind status table, then every structural mismatch."""
    if not rows:
        return "tensors: 0  (neither checkpoint contains any tensors)"
    counts = {s: 0 for s in STATUSES}
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in STATUSES})
    identical_params = total_params = 0
    for r in rows:
        counts[r.status] += 1
        by_kind[r.kind][r.status] += 1
        total_params += r.numel
        identical_params += r.numel if r.status == IDENTICAL else 0

    recast = [r for r in rows if r.status in (IDENTICAL, CHANGED) and r.base_dtype != r.suspect_dtype]
    lines = [
        f"tensors: {len(rows)}  " + "  ".join(f"{s}={n}" for s, n in counts.items() if n),
        f"parameters in unchanged tensors: {identical_params / max(total_params, 1):.2%}",
    ]
    if recast:
        pairs = sorted({f"{r.base_dtype}->{r.suspect_dtype}" for r in recast})
        lines.append(f"dtype differs ({', '.join(pairs)}) in {len(recast)} tensors: compared by value, not bytes")
    lines.append("")
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
