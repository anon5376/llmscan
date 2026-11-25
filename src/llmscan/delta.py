"""Run the ΔW core over every 2-D tensor and stream results to CSV in inventory order."""

from __future__ import annotations

import concurrent.futures
import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from llmscan import core
from llmscan._blas import BlasThreadCap
from llmscan.gguf_io import GGUFCheckpoint
from llmscan.inventory import CHANGED, DTYPE_MISMATCH, IDENTICAL, InventoryRow
from llmscan.names import RESIDUAL_WRITERS, TOKEN_MATRICES, canonical_kind
from llmscan.safetensors_io import FLOAT_DTYPES, Checkpoint
from llmscan.token_blocks import token_blocks

FIELDS = ["name", "layer", "kind", "status", "shape", "effective_rank", "seconds"]
EXTRA_FIELDS = ["frobenius", "projection_gain", "second_tail_share", "token_row_fraction",
                "projection_error", "token_row_spike", "rank_one_excess", "tail_norm_fraction", "contraction_excess",
                "reciprocity_excess", "removal_excess"]
DEFAULT_MAX_MEMORY_MIB = 512

# Stream token matrices when decoding both checkpoints would exceed this size.
_TOKEN_WORKING_BYTES = 64 << 20


def sv_field(i: int) -> str:
    return f"sv_{i:03d}"


def estimate_tensor_bytes(row: InventoryRow) -> int:
    """Estimated peak working memory (bytes) for one 2-D tensor in `_analyze_tensor`.

    Accounts for both decoded weight arrays (float32 or float64) plus the largest
    non-overlapping temporary phase: any intermediate float32 widening array for mixed
    F64, the `core.removal_bounds` scaled-block cache and radius buffers for projection
    tensors, the full-tensor `np.isfinite` boolean mask, or the float64 Gram/block
    workspace.
    """
    d0, d1 = (int(x) for x in row.base_shape.split("x"))
    m, n = (d0, d1) if d0 >= d1 else (d1, d0)
    use_f64 = "F64" in (row.base_dtype, row.suspect_dtype)
    itemsize = 8 if use_f64 else 4
    decoded = 2 * m * n * itemsize
    # Checkpoint.to_float64 widens non-F64 tensors via an intermediate float32 array.
    decode_temp = m * n * 4 if use_f64 and (row.base_dtype != "F64" or row.suspect_dtype != "F64") else 0
    finite_mask = m * n
    gram = 2 * n * n * 8
    block = 2 * min(m, core._ROWS_PER_BLOCK) * n * 8
    kind = canonical_kind(row.kind)
    token_matrix = kind in TOKEN_MATRICES and row.layer is None
    removal_ws = 0
    if token_matrix or kind in RESIDUAL_WRITERS:
        rows, cols = (d1, d0) if token_matrix else (d0, d1)
        r_block_rows = min(rows, max(1, core._ROWS_PER_BLOCK * min(rows, cols) // cols))
        r_block = r_block_rows * cols * 8
        r_cache = min(2 * rows * cols * 8, core._REMOVAL_CACHE_BYTES)
        r_stream = 2 * r_block if 2 * rows * cols * 8 > core._REMOVAL_CACHE_BYTES else 0
        r_radius = (3 if kind in RESIDUAL_WRITERS else 1) * r_block
        removal_ws = r_cache + r_stream + r_radius
    # The widening temporary, removal-check workspace, finite-value mask, and Gram/block
    # workspace live in separate phases while the two decoded arrays stay resident.
    return decoded + max(decode_temp, removal_ws, finite_mask, gram + block)


def _max_parallelism(costs: list[int], jobs: int, max_memory_bytes: int) -> int:
    """Largest number of consecutive changed tensors that fit within `jobs` and `max_memory_bytes`."""
    if jobs <= 1 or len(costs) <= 1:
        return 1
    best = 1
    total = 0
    left = 0
    for right, cost in enumerate(costs):
        total += cost
        while left <= right and (total > max_memory_bytes or right - left + 1 > jobs):
            total -= costs[left]
            left += 1
        if right - left + 1 > best:
            best = right - left + 1
    return best


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


def _analyze_tensor(
    a,
    b,
    difference,
    r: InventoryRow,
    k: int,
    projection: bool,
    compare_spaces: bool,
) -> tuple[core.DeltaStats, float | None, float | None, float | None, float | None, float]:
    t0 = time.perf_counter()
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
    contraction = reciprocity = None
    if projection:
        contraction, reciprocity = core.projection_excesses(a, difference, stats, *dtypes)
    del a, b, difference
    return stats, excess, contraction, reciprocity, removal, time.perf_counter() - t0


def _write_delta_row(
    w: csv.DictWriter,
    f,
    r: InventoryRow,
    kind: str,
    token_matrix: bool,
    projection: bool,
    stats: core.DeltaStats,
    excess: float | None,
    contraction: float | None,
    reciprocity: float | None,
    removal: float | None,
    elapsed: float,
    k: int,
    done: int,
    n_core: int,
    log: Callable[[str], None],
    directions: dict[str, np.ndarray] | None,
    subspaces: dict[str, np.ndarray] | None,
    spectra: dict[str, core.OutputSpectrum] | None,
) -> None:
    if directions is not None and projection and stats.direction is not None:
        directions[r.name] = stats.direction
    if subspaces is not None and kind in RESIDUAL_WRITERS and stats.output_subspace is not None:
        subspaces[r.name] = stats.output_subspace
        if spectra is not None and stats.output_spectrum is not None:
            spectra[r.name] = stats.output_spectrum
    row = {
        "name": r.name,
        "layer": r.layer,
        "kind": r.kind,
        "status": r.status,
        "shape": r.base_shape,
        "seconds": f"{elapsed:.2f}",
        "effective_rank": f"{float(stats.effective_rank):.6g}",
    }
    values = (
        stats.frobenius,
        stats.projection_gain if projection else None,
        stats.second_tail_share,
        stats.column_fraction if token_matrix else None,
        stats.projection_error if projection else None,
        stats.column_spike if token_matrix else None,
        excess,
        stats.tail_norm_fraction,
        contraction,
        reciprocity,
        removal,
    )
    for field, value in zip(EXTRA_FIELDS, values):
        if value is not None:
            row[field] = f"{value:.8g}"
    for i, s in enumerate(stats.singular_values[:k]):
        row[sv_field(i + 1)] = f"{float(s):.6g}"
    log(f"[{done}/{n_core}] {r.name}  eff_rank={row['effective_rank']}  ({row['seconds']}s)")
    w.writerow(row)
    f.flush()


def _scan_serial(
    base: Checkpoint,
    suspect: Checkpoint,
    todo: list[InventoryRow],
    w: csv.DictWriter,
    f,
    k: int,
    n_core: int,
    log: Callable[[str], None],
    directions: dict[str, np.ndarray] | None,
    subspaces: dict[str, np.ndarray] | None,
    spectra: dict[str, core.OutputSpectrum] | None,
) -> int:
    done = 0
    for r in todo:
        if r.status == IDENTICAL:
            w.writerow({"name": r.name, "layer": r.layer, "kind": r.kind, "status": r.status, "shape": r.base_shape})
            f.flush()
            continue
        t0 = time.perf_counter()
        kind = canonical_kind(r.kind)
        token_matrix = kind in TOKEN_MATRICES and r.layer is None
        projection = token_matrix or kind in RESIDUAL_WRITERS
        compare_spaces = subspaces is not None and kind in RESIDUAL_WRITERS
        a, b, difference = _tensor_inputs(base, suspect, r, token_matrix)
        stats, excess, contraction, reciprocity, removal, _ = _analyze_tensor(
            a, b, difference, r, k, projection, compare_spaces,
        )
        del a, b, difference
        done += 1
        _write_delta_row(
            w, f, r, kind, token_matrix, projection, stats, excess, contraction, reciprocity, removal,
            time.perf_counter() - t0, k, done, n_core, log, directions, subspaces, spectra,
        )
    return done


def _scan_concurrent(
    base: Checkpoint,
    suspect: Checkpoint,
    todo: list[InventoryRow],
    changed_indices: list[int],
    costs: list[int],
    w: csv.DictWriter,
    f,
    k: int,
    eff_workers: int,
    max_memory_bytes: int,
    log: Callable[[str], None],
    directions: dict[str, np.ndarray] | None,
    subspaces: dict[str, np.ndarray] | None,
    spectra: dict[str, core.OutputSpectrum] | None,
) -> int:
    n_core = len(changed_indices)
    max_buffered = max(eff_workers * 4, 16)
    done = 0
    write_pos = 0
    next_change = 0
    running: dict[concurrent.futures.Future, tuple[int, int, float]] = {}
    completed: dict[int, tuple[concurrent.futures.Future, float]] = {}
    in_flight_bytes = 0
    stop_submitting = False

    blas_cap = BlasThreadCap(eff_workers)
    blas_cap.apply()
    restore_blas = True
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="llmscan")
    try:
        while write_pos < len(todo):
            while write_pos < len(todo):
                r = todo[write_pos]
                if r.status == IDENTICAL:
                    w.writerow({"name": r.name, "layer": r.layer, "kind": r.kind, "status": r.status, "shape": r.base_shape})
                    f.flush()
                    write_pos += 1
                elif write_pos in completed:
                    fut, load_sec = completed.pop(write_pos)
                    stats, excess, contraction, reciprocity, removal, calc_sec = fut.result()
                    kind = canonical_kind(r.kind)
                    token_matrix = kind in TOKEN_MATRICES and r.layer is None
                    projection = token_matrix or kind in RESIDUAL_WRITERS
                    done += 1
                    _write_delta_row(
                        w, f, r, kind, token_matrix, projection, stats, excess, contraction, reciprocity, removal,
                        load_sec + calc_sec, k, done, n_core, log, directions, subspaces, spectra,
                    )
                    write_pos += 1
                else:
                    break
            if write_pos >= len(todo):
                break

            while not stop_submitting and next_change < n_core:
                if len(running) >= eff_workers or len(running) + len(completed) >= max_buffered:
                    break
                cost = costs[next_change]
                if running and in_flight_bytes + cost > max_memory_bytes:
                    break
                idx = changed_indices[next_change]
                r = todo[idx]
                next_change += 1
                kind = canonical_kind(r.kind)
                token_matrix = kind in TOKEN_MATRICES and r.layer is None
                projection = token_matrix or kind in RESIDUAL_WRITERS
                compare_spaces = subspaces is not None and kind in RESIDUAL_WRITERS
                try:
                    t_load = time.perf_counter()
                    a, b, difference = _tensor_inputs(base, suspect, r, token_matrix)
                    load_sec = time.perf_counter() - t_load
                except Exception as load_exc:
                    failed: concurrent.futures.Future = concurrent.futures.Future()
                    failed.set_exception(load_exc)
                    completed[idx] = (failed, 0.0)
                    stop_submitting = True
                    break
                if difference is not None:
                    # Streamed token blocks read from the checkpoint mmaps on demand, so
                    # keep them on the main thread rather than sharing Checkpoint._maps.
                    fut: concurrent.futures.Future = concurrent.futures.Future()
                    try:
                        fut.set_result(_analyze_tensor(a, b, difference, r, k, projection, compare_spaces))
                    except Exception as run_exc:
                        fut.set_exception(run_exc)
                        stop_submitting = True
                    del a, b, difference
                    completed[idx] = (fut, load_sec)
                    if stop_submitting:
                        break
                    continue
                fut = ex.submit(_analyze_tensor, a, b, None, r, k, projection, compare_spaces)
                del a, b
                running[fut] = (idx, cost, load_sec)
                in_flight_bytes += cost

            if not running:
                continue

            done_futs, _ = concurrent.futures.wait(
                running, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done_futs:
                idx, cost, load_sec = running.pop(fut)
                in_flight_bytes -= cost
                completed[idx] = (fut, load_sec)
                if fut.exception() is not None:
                    stop_submitting = True
    except BaseException as exc:
        if isinstance(exc, Exception):
            ex.shutdown(wait=True, cancel_futures=True)
        else:
            ex.shutdown(wait=False, cancel_futures=True)
            if any(not fut.done() for fut in running):
                restore_blas = False
        raise
    else:
        ex.shutdown(wait=True)
    finally:
        if restore_blas:
            blas_cap.restore()
    return done


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
    *,
    jobs: int = 1,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_MIB * 1024 * 1024,
) -> int:
    """Analyse each 2-D float tensor present in both checkpoints with the same shape.

    Identical tensors get a row with no stats (ΔW = 0, the core is skipped). Rows are
    flushed in inventory order as they finish, so an interrupted run keeps its completed
    prefix. When `jobs > 1`, independent changed tensors may be analysed concurrently while
    keeping estimated in-flight working memory within `max_memory_bytes` (a single larger
    tensor still runs alone). Returns the number of tensors sent to the core.
    """
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if max_memory_bytes < 1:
        raise ValueError(f"max_memory_bytes must be >= 1, got {max_memory_bytes}")

    todo = [
        r for r in inventory
        if r.status in (IDENTICAL, CHANGED, DTYPE_MISMATCH)
        and r.base_shape.count("x") == 1
        and r.base_dtype in FLOAT_DTYPES and r.suspect_dtype in FLOAT_DTYPES
    ]
    changed_indices = [i for i, r in enumerate(todo) if r.status != IDENTICAL]
    n_core = len(changed_indices)
    costs = [estimate_tensor_bytes(todo[i]) for i in changed_indices] if jobs > 1 and n_core > 1 else []
    eff_workers = _max_parallelism(costs, jobs, max_memory_bytes) if costs else 1

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + [sv_field(i + 1) for i in range(k)] + EXTRA_FIELDS)
        w.writeheader()
        if eff_workers <= 1:
            return _scan_serial(
                base, suspect, todo, w, f, k, n_core, log, directions, subspaces, spectra,
            )
        return _scan_concurrent(
            base, suspect, todo, changed_indices, costs, w, f, k, eff_workers, max_memory_bytes, log,
            directions, subspaces, spectra,
        )
