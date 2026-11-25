"""Run the ΔW core over every 2-D tensor, one pair at a time, and stream results to CSV."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from llmscan import core
from llmscan.gguf_io import GGUFCheckpoint
from llmscan.inventory import CHANGED, DTYPE_MISMATCH, IDENTICAL, InventoryRow
from llmscan.names import RESIDUAL_WRITERS, TOKEN_MATRICES, canonical_kind
from llmscan.safetensors_io import FLOAT_DTYPES, Checkpoint
from llmscan.token_blocks import token_blocks

FIELDS = ["name", "layer", "kind", "status", "shape", "effective_rank", "seconds"]
EXTRA_FIELDS = ["frobenius", "projection_gain", "second_tail_share", "token_row_fraction",
                "projection_error", "token_row_spike", "rank_one_excess", "tail_norm_fraction", "contraction_excess",
                "reciprocity_excess", "removal_excess"]


def sv_field(i: int) -> str:
    return f"sv_{i:03d}"


# Stream token matrices when decoding both checkpoints would exceed this size.
_TOKEN_WORKING_BYTES = 64 << 20


def _tensor_inputs(base, suspect, row, token_matrix):
    """Return both weights and, for streamed inputs, a blockwise difference."""
    wide_dtype = "F64" in (row.base_dtype, row.suspect_dtype)
    if token_matrix and row.numel * (16 if wide_dtype else 8) > _TOKEN_WORKING_BYTES:
        tokens, features = map(int, row.base_shape.split("x"))
        # Virtual readers can override whole-tensor decoding. Only read
        # mapped bytes directly for the two built-in checkpoint formats.
        if tokens > features and all(type(c) in (Checkpoint, GGUFCheckpoint) for c in (base, suspect)):
            return token_blocks(base, suspect, row.name, core._ROWS_PER_BLOCK)

    load = "to_float64" if wide_dtype else "to_float32"
    a, b = getattr(base, load)(row.name), getattr(suspect, load)(row.name)
    return (a.T, b.T, None) if token_matrix else (a, b, None)


def scan(
    base: Checkpoint,
    suspect: Checkpoint,
    inventory: list[InventoryRow],
    out_csv: Path,
    k: int = 32,
    log: Callable[[str], None] = print,
    directions: dict[str, np.ndarray] | None = None,
    subspaces: dict[str, np.ndarray] | None = None,
    spectra: dict[str, core.OutputSpectrum] | None = None,
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
                kind = canonical_kind(r.kind)
                token_matrix = kind in TOKEN_MATRICES and r.layer is None
                projection = token_matrix or kind in RESIDUAL_WRITERS
                compare_spaces = subspaces is not None and kind in RESIDUAL_WRITERS
                a, b, difference = _tensor_inputs(base, suspect, r, token_matrix)
                dtypes = r.base_dtype, r.suspect_dtype
                removal, refined = None, float("inf")
                if projection:
                    removal, refined = core.removal_bounds(a, b, *dtypes, difference=compare_spaces)
                budget = core.rounding_budget(a, *dtypes)
                space_budget = float("inf")
                if compare_spaces:
                    space_budget = (budget, refined) if refined < budget else budget
                if difference is None:
                    stats = core.analyze_delta(a, b, k, space_budget)
                    difference = b  # analyze_delta subtracted the base in place
                else:
                    stats = core.analyze_difference(a, difference, k, space_budget)
                excess = stats.tail_excess(budget)
                contraction = core.contraction_excess(a, difference, stats, *dtypes) if projection else None
                reciprocity = core.reciprocity_excess(a, difference, stats, *dtypes) if projection else None
                del a, b, difference
                if directions is not None and projection and stats.direction is not None:
                    directions[r.name] = stats.direction
                if subspaces is not None and kind in RESIDUAL_WRITERS and stats.output_subspace is not None:
                    subspaces[r.name] = stats.output_subspace
                    if spectra is not None and stats.output_spectrum is not None:
                        spectra[r.name] = stats.output_spectrum
                values = (stats.frobenius, stats.projection_gain if projection else None, stats.second_tail_share,
                          stats.column_fraction if token_matrix else None, stats.projection_error if projection else None,
                          stats.column_spike if token_matrix else None, excess, stats.tail_norm_fraction, contraction,
                          reciprocity, removal)
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
