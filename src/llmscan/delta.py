"""Run the ΔW core over every 2-D tensor, one pair at a time, and stream results to CSV."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from llmscan import core
from llmscan.inventory import CHANGED, DTYPE_MISMATCH, IDENTICAL, InventoryRow
from llmscan.names import RESIDUAL_WRITERS, TOKEN_MATRICES, canonical_kind
from llmscan.safetensors_io import FLOAT_DTYPES, Checkpoint

FIELDS = ["name", "layer", "kind", "status", "shape", "effective_rank", "seconds"]
EXTRA_FIELDS = ["frobenius", "projection_gain", "second_tail_share", "token_row_fraction",
                "projection_error", "token_row_spike", "rank_one_excess", "tail_norm_fraction"]


def sv_field(i: int) -> str:
    return f"sv_{i:03d}"


def scan(
    base: Checkpoint,
    suspect: Checkpoint,
    inventory: list[InventoryRow],
    out_csv: Path,
    k: int = 32,
    log: Callable[[str], None] = print,
    directions: dict[str, np.ndarray] | None = None,
) -> int:
    """Analyse each 2-D float tensor present in both checkpoints with the same shape.

    Identical tensors get a row with no stats (ΔW = 0, the core is skipped). Rows are
    flushed as they are produced, so an interrupted run keeps what it finished.
    Returns the number of tensors sent to the core.
    """
    todo = [
        r for r in inventory
        if r.status in (IDENTICAL, CHANGED, DTYPE_MISMATCH)
        and r.base_shape.count("x") == 1
        and r.base_dtype in FLOAT_DTYPES and r.suspect_dtype in FLOAT_DTYPES
    ]
    n_core = sum(r.status != IDENTICAL for r in todo)
    done = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + [sv_field(i + 1) for i in range(k)] + EXTRA_FIELDS)
        w.writeheader()
        for r in todo:
            row = {"name": r.name, "layer": r.layer, "kind": r.kind, "status": r.status, "shape": r.base_shape}
            if r.status != IDENTICAL:
                t0 = time.perf_counter()
                load = "to_float64" if "F64" in (r.base_dtype, r.suspect_dtype) else "to_float32"
                a, b = getattr(base, load)(r.name), getattr(suspect, load)(r.name)
                kind = canonical_kind(r.kind)
                token_matrix = kind in TOKEN_MATRICES and r.layer is None
                projection = token_matrix or kind in RESIDUAL_WRITERS
                if token_matrix:
                    a, b = a.T, b.T
                stats = core.analyze_delta(a, b, k)
                excess = core.rank_one_excess(a, stats, r.base_dtype, r.suspect_dtype)
                del a, b
                if directions is not None and projection and stats.direction is not None:
                    directions[r.name] = stats.direction
                values = (stats.frobenius, stats.projection_gain if projection else None, stats.second_tail_share,
                          stats.column_fraction if token_matrix else None, stats.projection_error if projection else None,
                          stats.column_spike if token_matrix else None, excess, stats.tail_norm_fraction)
                for field, value in zip(EXTRA_FIELDS, values):
                    if value is not None:
                        row[field] = f"{value:.8g}"
                row["seconds"] = f"{time.perf_counter() - t0:.2f}"
                row["effective_rank"] = f"{float(stats.effective_rank):.6g}"
                for i, s in enumerate(stats.singular_values[:k]):
                    row[sv_field(i + 1)] = f"{float(s):.6g}"
                done += 1
                log(f"[{done}/{n_core}] {r.name}  eff_rank={row['effective_rank']}  ({row['seconds']}s)")
            w.writerow(row)
            f.flush()
    return done
