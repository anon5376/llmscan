"""Check the honest-abliteration regression set, running one full llmscan at a time.

    PYTHONPATH=src python scripts/check_honest.py --models data/models --out out/honest-gate

CLEAN means every ranked tensor was T4 and at least one tensor changed.
Exit 0 for all CLEAN, 1 for FLAGGED, or 2 for an incomplete check.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

DEFAULT_PAIRS = (
    ("Qwen--Qwen3-1.7B", "huihui-ai--Huihui-Qwen3-1.7B-abliterated-v2"),
    ("Qwen--Qwen3-1.7B", "mlabonne--Qwen3-1.7B-abliterated"),
    ("Qwen--Qwen3-1.7B", "Goekdeniz-Guelmez--Josiefied-Qwen3-1.7B-abliterated-v1"),
    ("Qwen--Qwen3-0.6B", "DavidAU--Qwen3-0.6B-heretic-abliterated-uncensored"),
    ("Qwen--Qwen3-0.6B", "huihui-ai--Huihui-Qwen3-0.6B-abliterated-v2"),
    ("Qwen--Qwen3-0.6B", "mlabonne--Qwen3-0.6B-abliterated"),
)
TIERS = ("T1", "T2", "T3", "T4")


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        names = line.partition("#")[0].split()
        if not names:
            continue
        if len(names) != 2 or any(n in {".", ".."} or Path(n).name != n or "\\" in n for n in names):
            raise ValueError(f"{path}:{i}: expected two directory names (base suspect)")
        pairs.append((names[0], names[1]))
    if not pairs:
        raise ValueError(f"{path}: no pairs")
    return pairs


def outside_models(path: Path, models: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(models):
        raise ValueError(f"output {path} is inside --models {models}")
    return path


def scan(base: Path, suspect: Path, out: Path) -> dict[str, int]:
    ranking = out / "ranking.csv"
    try:
        previous = ranking.stat()
    except FileNotFoundError:
        previous = None
    result = subprocess.run(
        [sys.executable, "-m", "llmscan", "--base", str(base), "--suspect", str(suspect), "--out", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace",
    )
    if result.returncode:
        raise ValueError(f"llmscan exited {result.returncode}: {result.stderr.strip()}")
    if previous is not None:
        current = ranking.stat()
        if current.st_ino == previous.st_ino and current.st_mtime_ns == previous.st_mtime_ns:
            raise ValueError(f"llmscan left an old {ranking}")
    counts = dict.fromkeys(TIERS, 0)
    with ranking.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f, strict=True)
        if "tier" not in (rows.fieldnames or ()):
            raise ValueError(f"{ranking}: missing tier column")
        for row in rows:
            tier = row["tier"]
            if tier not in counts:
                raise ValueError(f"{ranking}:{rows.line_num}: unknown tier {tier!r}")
            counts[tier] += 1
    return counts


def error(message: str) -> None:
    print("check_honest:", " ".join(message.splitlines()))


def main(argv: list[str] | None = None) -> int:
    ap = ArgumentParser(description="Run llmscan on the honest abliterations and count the ranked tiers.")
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="output directory (default: a new temporary directory)")
    ap.add_argument("--pairs-file", type=Path, help="replace the six defaults with lines of base suspect")
    try:
        args = ap.parse_args(argv)
        models = args.models.resolve()
        pairs = read_pairs(args.pairs_file) if args.pairs_file else DEFAULT_PAIRS
        if args.out is None:
            tmp = outside_models(Path(tempfile.gettempdir()), models)
            out = Path(tempfile.mkdtemp(prefix="check-honest-", dir=tmp))
        else:
            out = outside_models(args.out, models)
    except (OSError, ValueError, RuntimeError) as e:
        error(str(e))
        return 2

    totals = Counter()
    print("suspect", "changed", *TIERS, "seconds", "verdict", sep="\t", flush=True)
    for base, suspect in pairs:
        start = time.perf_counter()
        cells = ["-"] * 5
        try:
            missing = [str(models / n) for n in (base, suspect) if not (models / n).is_dir()]
            if missing:
                verdict = "MISSING"
                error(f"{suspect}: missing model directory: {', '.join(missing)}")
            else:
                pair_out = outside_models(out / f"{suspect}-vs-{base}", models)
                counts = scan(models / base, models / suspect, pair_out)
                changed = sum(counts.values())
                cells = [str(v) for v in (changed, *counts.values())]
                verdict = "CLEAN" if changed and counts["T4"] == changed else "FLAGGED"
        except (OSError, ValueError, RuntimeError, csv.Error) as e:
            verdict = "ERROR"
            error(f"{suspect}: {e}")
        totals[verdict] += 1
        print(suspect, *cells, f"{time.perf_counter() - start:.2f}", verdict, sep="\t", flush=True)
    print("summary:", *(f"{v}={totals[v]}" for v in ("CLEAN", "FLAGGED", "MISSING", "ERROR")), f"out={out}")
    if totals["MISSING"] or totals["ERROR"]:
        return 2
    return int(bool(totals["FLAGGED"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.exit(main())
