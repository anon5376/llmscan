"""Score a llmscan ranking against an answer key, for the blind test. Standard library only.

    python scripts/score_blind.py --ranking out/<run>/ranking.csv --key key.txt

The key holds one tensor name per line; blank lines, `#` comments and repeated names are ignored.
A ranking that lists a name twice is rejected. With n key
tensors, PASS means the top n ranked names are exactly the key set. Exit 0 on PASS, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_ranking(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["name"] for row in csv.DictReader(f)]


def read_key(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(line for line in lines if line))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a llmscan ranking.csv against an answer key.")
    ap.add_argument("--ranking", required=True, type=Path, help="ranking.csv written by llmscan")
    ap.add_argument("--key", required=True, type=Path, help="answer key: one tensor name per line")
    args = ap.parse_args(argv)

    try:  # the ranking is read first; nothing is printed until both files are in
        ranked = read_ranking(args.ranking)
        key = read_key(args.key)
    except (OSError, KeyError, UnicodeDecodeError) as e:
        raise SystemExit(f"score_blind: {e!r}")
    if not key:
        raise SystemExit(f"score_blind: {args.key}: no tensor names")

    if len(set(ranked)) != len(ranked):
        raise SystemExit(f"score_blind: {args.ranking}: duplicate tensor names in the ranking")
    position = {name: i for i, name in enumerate(ranked, 1)}
    n, wanted = len(key), set(key)
    precision = sum(name in wanted for name in ranked[:n]) / n
    recall10 = sum(name in wanted for name in ranked[:10]) / n
    passed = set(ranked[:n]) == wanted and len(ranked) >= n

    for name in sorted(key, key=lambda k: position.get(k, len(ranked) + 1)):
        print(f"{'rank ' + str(position[name]) if name in position else 'not ranked':>12}  {name}")
    print(f"precision@{n} = {precision:.3f}")
    print(f"recall@10 = {recall10:.3f}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
