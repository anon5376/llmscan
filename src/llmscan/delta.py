"""Run the ΔW core over every 2-D tensor, one pair at a time, and stream results to CSV."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

from llmscan import core
from llmscan.inventory import CHANGED, DTYPE_MISMATCH, IDENTICAL, InventoryRow
from llmscan.safetensors_io import FLOAT_DTYPES, Checkpoint

FIELDS = ["name", "layer", "kind", "status", "shape", "effective_rank", "seconds"]


def sv_field(i: int) -> str:
    return f"sv_{i:03d}"


def scan(
    base: Checkpoint,
    suspect: Checkpoint,
    inventory: list[InventoryRow],
    out_csv: Path,
    k: int = 32,
    log: Callable[[str], None] = print,
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
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + [sv_field(i + 1) for i in range(k)])
        w.writeheader()
        for r in todo:
            row = {"name": r.name, "layer": r.layer, "kind": r.kind, "status": r.status, "shape": r.base_shape}
            if r.status != IDENTICAL:
                t0 = time.perf_counter()
                stats = core.analyze_delta(base.to_float32(r.name), suspect.to_float32(r.name), k)
                row["seconds"] = f"{time.perf_counter() - t0:.2f}"
                row["effective_rank"] = f"{float(stats.effective_rank):.6g}"
                for i, s in enumerate(stats.singular_values[:k]):
                    row[sv_field(i + 1)] = f"{float(s):.6g}"
                done += 1
                log(f"[{done}/{n_core}] {r.name}  eff_rank={row['effective_rank']}  ({row['seconds']}s)")
            w.writerow(row)
            f.flush()
    return done
