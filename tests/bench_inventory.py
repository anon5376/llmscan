"""Compare two source trees on the same checkpoints. Usage: python tests/bench_inventory.py BASELINE_SRC"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from helpers import tiny_llama, write_sharded


def checkpoints(root):
    weights = tiny_llama(np.random.default_rng(42), layers=4, d=1024, ff=3072, vocab=38912)
    shards = [{k: v for k, v in weights.items() if ("layers." in k) == layers} for layers in (False, True)]
    write_sharded(root / "base", shards)
    shutil.copytree(root / "base", root / "identical")
    weights["model.layers.0.self_attn.o_proj.weight"][1][0] *= -1
    weights["model.layers.3.mlp.down_proj.weight"][1][-1] *= -1
    write_sharded(root / "changed", shards)
    return [(root / "base", root / case) for case in ("identical", "changed")]


def outputs(out):
    result = {p.name: p.read_bytes() for p in out.iterdir() if p.name != "delta.csv"}
    with (out / "delta.csv").open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        seconds = header.index("seconds")
        rows = [row[:seconds] + row[seconds + 1:] for row in reader]
    return result, header, rows


INVENTORY = """
import sys, time
from llmscan.checkpoint import open_checkpoint
from llmscan.inventory import compare
with open_checkpoint(sys.argv[1]) as a, open_checkpoint(sys.argv[2]) as b:
    start = time.perf_counter()
    compare(a, b)
    print(time.perf_counter() - start)
"""


def benchmark(root, baseline, pairs):
    sources = [baseline, Path(__file__).resolve().parents[1] / "src"]
    for src in sources:
        if not (src / "llmscan" / "inventory.py").is_file():
            raise ValueError(f"{src} is not a llmscan source tree")
    print("case / stage: baseline -> candidate (median ms)", flush=True)
    for n, pair in enumerate(pairs or checkpoints(root)):
        outs = [root / f"out-{n}-{j}" for j in range(2)]
        for stage in ("inventory", "full CLI"):
            samples = [[], []]
            for i in range(7):  # one warm-up; alternate order over six measured pairs
                for j in ((0, 1) if i % 2 == 0 else (1, 0)):
                    env = dict(os.environ, PYTHONPATH=str(sources[j]), OPENBLAS_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
                    if stage == "inventory":
                        elapsed = float(subprocess.check_output([sys.executable, "-c", INVENTORY, *pair], env=env))
                    else:
                        start = time.perf_counter()
                        subprocess.run([sys.executable, "-m", "llmscan", "--base", pair[0], "--suspect", pair[1],
                                        "--out", outs[j]], check=True, stdout=subprocess.DEVNULL, env=env)
                        elapsed = time.perf_counter() - start
                    if i:
                        samples[j].append(1000 * elapsed)
            print(f"{pair[1].name} / {stage}: " + " -> ".join(f"{statistics.median(s):.1f}" for s in samples), flush=True)
        if outputs(outs[0]) != outputs(outs[1]):
            raise AssertionError(f"{pair}: report outputs differ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_src", type=Path)
    ap.add_argument("--pair", action="append", nargs=2, type=Path, metavar=("BASE", "SUSPECT"),
                    help="benchmark local checkpoints instead of generating the fixture; repeat for more pairs")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        benchmark(Path(tmp), args.baseline_src.resolve(), args.pair)
