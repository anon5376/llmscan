#!/usr/bin/env python3
"""Measure residual edit directions in a local checkpoint pair."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmscan import atomic, inventory
from llmscan.checkpoint import open_checkpoint
from llmscan.cli import _create_private_dirs, _refuse_shared_location
from llmscan.direction import Direction, alignment_scores, analyze_direction, residual_axis
from llmscan.safetensors_io import FLOAT_DTYPES, Checkpoint, CheckpointError


def replacement(base, scratch, kind, norm, rng):
    """Replace one edit in memory. The rank-one change has the honest norm before the BF16 save."""
    left, right = (rng.standard_normal(n) for n in base.shape)
    left *= norm / np.linalg.norm(left)
    right /= np.linalg.norm(right)
    for start in range(0, len(base), 128):
        rows = slice(start, start + 128)
        saved = (base[rows] + left[rows, None] * right).astype(np.float32)
        bits = saved.view(np.uint32)
        bits += 0x7FFF + ((bits >> 16) & 1)  # round the edited weights to nearest BF16, ties to even
        bits &= 0xFFFF0000
        scratch[rows] = saved
    return analyze_direction(base, scratch, kind)


def measure(base: Checkpoint, suspect: Checkpoint, row: inventory.InventoryRow, probe: bool, rng):
    reason = ""
    if row.base_shape.count("x") != 1:
        reason = "not 2-D"
    elif row.base_dtype not in FLOAT_DTYPES or row.suspect_dtype not in FLOAT_DTYPES:
        reason = "unsupported dtype"
    elif residual_axis(row.kind) is None:
        reason = "unknown residual side"
    if reason:
        return Direction(None, math.nan, math.nan, reason), None

    a = base.to_float64(row.name) if row.base_dtype == "F64" else base.to_float32(row.name)
    b = suspect.to_float64(row.name) if row.suspect_dtype == "F64" else suspect.to_float32(row.name)
    honest = analyze_direction(a, b, row.kind)
    trial = None
    if probe and honest.vector is not None and math.isfinite(honest.delta_norm):
        trial = replacement(a, b, row.kind, honest.delta_norm, rng)
    return honest, trial  # only vectors and scalars survive this pair


def write_csv(path, rows, stats, scores):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("name", "layer", "kind", "base_dtype", "suspect_dtype", "energy_share",
                         "alignment", "delta_norm", "reason"))
        for row, stat, score in zip(rows, stats, scores):
            reason = stat.reason or ("no comparable peer" if math.isnan(score) else "")
            numbers = ["" if math.isnan(n) else n for n in (stat.energy_share, score, stat.delta_norm)]
            writer.writerow((row.name, row.layer, row.kind, row.base_dtype, row.suspect_dtype, *numbers, reason))


def summary(values):
    usable = [v for v in values if math.isfinite(v)]
    if not usable:
        return "n=0 min=NA median=NA"
    return f"n={len(usable)} min={min(usable):.8f} median={np.median(usable):.8f}"


def print_layer_pairs(rows, vectors):
    by_kind = {"o_proj": {}, "down_proj": {}}
    aliases = {"attn_output": "o_proj", "ffn_down": "down_proj",
               "experts.down_proj": "down_proj", "ffn_down_exps": "down_proj", "ffn_down_shexp": "down_proj"}
    for row, vector in zip(rows, vectors):
        kind = aliases.get(row.kind, row.kind)
        if kind in by_kind and row.layer is not None and vector is not None:
            by_kind[kind].setdefault(row.layer, []).append(vector)

    def compare(left, right, offset):
        return summary(min(abs(float(a @ b)), 1.0)
                       for layer, group in left.items()
                       for a in group for b in right.get(layer + offset, ())
                       if a.shape == b.shape)

    within = compare(by_kind["o_proj"], by_kind["down_proj"], 0)
    print(f"same layer |cos(o_proj, down_proj)| {within}")
    for kind, layers in by_kind.items():
        print(f"adjacent layers |cos| {kind} {compare(layers, layers, 1)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--suspect", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new or empty directory for directions.csv")
    ap.add_argument("--probe-layers", type=int, nargs="+", default=(), metavar="N",
                    help="try independent norm-matched BF16 replacements in o_proj and down_proj at these layers")
    args = ap.parse_args(argv)
    start = time.perf_counter()
    out = args.out.expanduser().resolve()
    existing_dirs = [d for d in (out, *out.parents) if d.is_dir()]
    for spec in (args.base, args.suspect):
        for path in (spec, spec.resolve()):
            model_dir = (path if path.is_dir() else path.parent).resolve()
            # resolve() preserves case even when the filesystem treats two spellings as one directory.
            if model_dir.is_dir() and any(d.samefile(model_dir) for d in existing_dirs):
                raise SystemExit(f"measure_directions: --out is inside the model directory {model_dir}")
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit("measure_directions: --out is not empty")
    out = _refuse_shared_location(out)
    _create_private_dirs(out)

    with open_checkpoint(args.base) as base, open_checkpoint(args.suspect) as suspect:
        rows = [r for r in inventory.compare(base, suspect) if r.status == inventory.CHANGED]
    # Drop the inventory's mappings of unchanged weights before allocating the Gram.
    targets = {(layer, kind) for layer in args.probe_layers for kind in ("o_proj", "down_proj")}
    rng = np.random.default_rng(0)
    stats, trials = [], {}
    with open_checkpoint(args.base) as base, open_checkpoint(args.suspect) as suspect:
        for i, row in enumerate(rows):
            stat, trial = measure(base, suspect, row, (row.layer, row.kind) in targets, rng)
            stats.append(stat)
            if trial is not None:
                trials[i] = trial

    vectors = [s.vector for s in stats]
    scores = alignment_scores(vectors)
    atomic.write(out / "directions.csv", lambda p: write_csv(p, rows, stats, scores))
    for kind in sorted({r.kind for r in rows}):
        group = [i for i, r in enumerate(rows) if r.kind == kind]
        print(f"{kind} changed={len(group)}  alignment {summary(scores[group])}  "
              f"energy_share {summary([stats[i].energy_share for i in group])}")
    print_layer_pairs(rows, vectors)

    for i, trial in trials.items():
        peers = vectors.copy()
        peers[i] = trial.vector
        result = alignment_scores(peers)
        other = np.delete(result, i)
        finite = other[np.isfinite(other)]
        score = result[i]
        rank = 1 + np.count_nonzero(finite < score) if math.isfinite(score) else "NA"
        print(f"probe {rows[i].name}  alignment={score:.8f}  rank={rank}/{np.isfinite(result).sum()}  "
              f"other_min={min(finite, default=math.nan):.8f}  energy_share={trial.energy_share:.8f}  "
              f"saved_norm/original={trial.delta_norm / stats[i].delta_norm:.8f}")
    for layer, kind in sorted(targets - {(rows[i].layer, rows[i].kind) for i in trials}):
        print(f"probe unavailable: layer={layer} kind={kind} has no usable changed matrix")
    print(f"wrote {out / 'directions.csv'}  changed={len(rows)}  elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except (CheckpointError, OSError) as exc:
        sys.exit(f"measure_directions: {exc}")
