"""llmscan command line: inventory -> ΔW scan -> figures."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

from llmscan import atomic, delta, inventory
from llmscan.checkpoint import open_checkpoint
from llmscan.safetensors_io import CheckpointError


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


OUTPUTS = ("pair.txt", "inventory.csv", "inventory.txt", "delta.csv", "spectra.png", "effective_rank.png",
           "ranking.csv")  # every file llmscan writes; none is ever written through a link


def _refuse_shared_location(out: Path) -> None:
    """On POSIX, refuse an --out whose directory, or any existing ancestor, other users could rename.

    Write permission for group or others without the sticky bit is enough to rename a directory
    (including llmscan's private .llmscan-* one) and plant a symlink at its name. Runs before
    anything is created. Windows has no such mode bits, so it is skipped there.
    """
    if os.name == "nt":
        return
    resolved = out.resolve()
    for d in (resolved, *resolved.parents):
        try:
            mode = d.stat().st_mode
        except OSError:  # does not exist yet (or is not a directory): llmscan will create it
            continue
        if stat.S_ISDIR(mode) and mode & (stat.S_IWGRP | stat.S_IWOTH) and not mode & stat.S_ISVTX:
            raise SystemExit(f"llmscan: {d} is writable by other users without the sticky bit, so they could "
                             f"replace llmscan's output; choose a directory only you can write to (pass --out)"
                             + (f"\n(unsafe directory is an ancestor of {resolved})" if d != resolved else ""))


def _slug(spec: str) -> str:
    return spec.rstrip("/\\").replace("\\", "/").split("/")[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="llmscan",
        description="Diff a suspect model against the base it claims to come from, one tensor at a time.",
    )
    ap.add_argument("--base", required=True, help="base model: local directory, .safetensors or .gguf file, or a cached HF id")
    ap.add_argument("--suspect", required=True, help="suspect model, same forms as --base")
    ap.add_argument("--out", type=Path, help="output directory (default: out/<suspect>-vs-<base>)")
    ap.add_argument("-k", type=int, default=32, help="leading singular values kept per tensor (default: 32)")
    ap.add_argument("--inventory-only", action="store_true", help="stop after the tensor inventory")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # a redirected Windows console may not encode every path
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    out = args.out or Path("out") / f"{_slug(args.suspect)}-vs-{_slug(args.base)}"
    _refuse_shared_location(out)
    base_path, suspect_path = resolve_model(args.base), resolve_model(args.suspect)
    pair = f"base: {base_path.resolve()}\nsuspect: {suspect_path.resolve()}\n"
    marker = out / "pair.txt"
    if any((out / name).is_symlink() for name in OUTPUTS):
        raise SystemExit(f"llmscan: {out} holds a symbolic link where llmscan writes; pass --out to choose another directory")
    if marker.is_file() and marker.stat().st_nlink > 1:
        raise SystemExit(f"llmscan: {marker} has other hard links, so llmscan did not create it; pass --out to choose another directory")
    if out.is_dir() and not marker.is_file() and any(out.iterdir()):
        raise SystemExit(f"llmscan: {out} is not empty and has no pair.txt from an earlier llmscan run; "
                         "pass --out to choose a new or empty directory")
    if marker.is_file() and (held := marker.read_text(encoding="utf-8", errors="replace")) != pair:
        raise SystemExit(f"llmscan: {out} holds a different comparison; pass --out to choose another directory\n"
                         + held.rstrip())
    out.mkdir(parents=True, exist_ok=True)
    atomic.write_text(marker, pair)
    try:
        base_ckpt, suspect_ckpt = open_checkpoint(base_path), open_checkpoint(suspect_path)
    except CheckpointError as e:
        raise SystemExit(f"llmscan: {e}")

    with base_ckpt as base, suspect_ckpt as suspect:
        for label, ck in (("base:   ", base), ("suspect:", suspect)):
            print(f"{label} {ck.path}  ({len(ck.tensors)} tensors in {', '.join(f.name for f in ck.files)})")
        print()
        rows = inventory.compare(base, suspect)
        atomic.write(out / "inventory.csv", lambda tmp: inventory.write_csv(rows, tmp))
        summary = inventory.summarize(rows)
        for label, ck in (("base", base), ("suspect", suspect)):
            summary += "".join(f"\n{label}: ignored {n} ({why})" for n, why in ck.ignored.items())
        atomic.write_text(out / "inventory.txt", summary + "\n")
        print(summary + "\n")
        if not rows:
            print("llmscan: neither checkpoint contains any tensors", file=sys.stderr)
            return 1
        if args.inventory_only:
            print(f"wrote {out / 'inventory.csv'}")
            return 0
        try:
            atomic.write(out / "delta.csv", lambda tmp: delta.scan(base, suspect, rows, tmp, k=args.k), keep_partial=True)
        except NotImplementedError as e:
            print(f"ΔW scan stopped: {e}", file=sys.stderr)
            print(f"wrote {out / 'inventory.csv'}")
            return 2

    from llmscan import plots  # matplotlib import is slow; only needed from here on

    drows = plots.read_rows(out / "delta.csv")
    pair = f"{_slug(args.suspect)}  vs  base {_slug(args.base)}"
    wrote_spectra = atomic.write(out / "spectra.png", lambda tmp: plots.plot_spectra(
        drows, tmp, f"Singular values of ΔW = W_suspect − W_base\n{pair}"))
    atomic.write(out / "effective_rank.png", lambda tmp: plots.plot_effective_rank(
        drows, tmp, f"Effective rank of ΔW by layer\n{pair}"))
    for name in ("inventory.csv", "delta.csv", "spectra.png", "effective_rank.png"):
        if name != "spectra.png" or wrote_spectra:
            print(f"wrote {out / name}")
    if not wrote_spectra and (out / "spectra.png").exists():
        print(f"llmscan: no tensor changed; {out / 'spectra.png'} is left over from an earlier run", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
