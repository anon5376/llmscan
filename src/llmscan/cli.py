"""llmscan command line: inventory -> ΔW scan -> figures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from llmscan import delta, inventory
from llmscan.safetensors_io import Checkpoint, CheckpointError


def resolve_model(spec: str) -> Path:
    """A local file/directory, or an `org/name` Hugging Face id already in the local HF cache."""
    p = Path(spec).expanduser()
    if p.exists():
        return p
    if spec.count("/") == 1 and not spec.startswith((".", "~")):
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
        repo = hub / ("models--" + spec.replace("/", "--"))
        ref = repo / "refs" / "main"
        if ref.is_file() and (snap := repo / "snapshots" / ref.read_text().strip()).is_dir():
            return snap
        raise SystemExit(f"llmscan: {spec} is not a local path and not in the Hugging Face cache at {hub}")
    raise SystemExit(f"llmscan: {spec}: no such file or directory")


def _slug(spec: str) -> str:
    return spec.rstrip("/\\").replace("\\", "/").split("/")[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="llmscan",
        description="Diff a suspect model against the base it claims to come from, one tensor at a time.",
    )
    ap.add_argument("--base", required=True, help="base model: local directory or .safetensors file, or a cached HF id")
    ap.add_argument("--suspect", required=True, help="suspect model, same forms as --base")
    ap.add_argument("--out", type=Path, help="output directory (default: out/<suspect>-vs-<base>)")
    ap.add_argument("-k", type=int, default=32, help="leading singular values kept per tensor (default: 32)")
    ap.add_argument("--inventory-only", action="store_true", help="stop after the tensor inventory")
    args = ap.parse_args(argv)

    out = args.out or Path("out") / f"{_slug(args.suspect)}-vs-{_slug(args.base)}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        base_ckpt, suspect_ckpt = Checkpoint(resolve_model(args.base)), Checkpoint(resolve_model(args.suspect))
    except CheckpointError as e:
        raise SystemExit(f"llmscan: {e}")

    with base_ckpt as base, suspect_ckpt as suspect:
        print(f"base:    {base.path}  ({len(base.tensors)} tensors)")
        print(f"suspect: {suspect.path}  ({len(suspect.tensors)} tensors)\n")
        rows = inventory.compare(base, suspect)
        inventory.write_csv(rows, out / "inventory.csv")
        summary = inventory.summarize(rows)
        (out / "inventory.txt").write_text(summary + "\n")
        print(summary + "\n")
        if args.inventory_only:
            print(f"wrote {out / 'inventory.csv'}")
            return 0
        try:
            delta.scan(base, suspect, rows, out / "delta.csv", k=args.k)
        except NotImplementedError as e:
            print(f"ΔW scan stopped: {e}", file=sys.stderr)
            print(f"wrote {out / 'inventory.csv'}")
            return 2

    from llmscan import plots  # matplotlib import is slow; only needed from here on

    drows = plots.read_rows(out / "delta.csv")
    pair = f"{_slug(args.suspect)}  vs  base {_slug(args.base)}"
    plots.plot_spectra(drows, out / "spectra.png", f"Singular values of ΔW = W_suspect − W_base\n{pair}")
    plots.plot_effective_rank(drows, out / "effective_rank.png", f"Effective rank of ΔW by layer\n{pair}")
    for name in ("inventory.csv", "delta.csv", "spectra.png", "effective_rank.png"):
        if (out / name).exists():
            print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
