"""Freeze the detector, commit one blind ranking, then open the committed key."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "scripts" / "score_blind.py"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unused(path: Path) -> None:
    if os.path.lexists(path):  # includes dangling links
        raise ValueError(f"{path.name} already exists")


def publish(path: Path, record: dict) -> None:
    from llmscan import atomic

    # atomic.write_text replaces its destination. Serialize the absence check and
    # publication so two invocations cannot both install the same record.
    lock = path.with_name("." + path.name + ".lock")
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise ValueError(f"publication of {path.name} already started") from None
    try:
        unused(path)
        atomic.write_text(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    finally:
        lock.rmdir()


def package_dir() -> Path:
    import llmscan

    origin = Path(llmscan.__file__).resolve()
    if origin.name != "__init__.py" or not origin.is_relative_to(ROOT):
        raise ValueError("imported llmscan is outside this checkout or is not a source package; set PYTHONPATH to its src")
    return origin.parent


def detector_snapshot(package: Path) -> tuple[dict, dict]:
    hashes, stamps = {}, {}
    for path in (SCORER, *package.rglob("*")):
        name = path.relative_to(ROOT).as_posix()
        if path.is_symlink() and (path.suffix == ".py" or path.is_dir()):
            raise ValueError(f"linked detector source: {name}")
        if path.suffix == ".py":
            hashes[name] = sha256(path)
            st = path.stat()
            stamps[name] = (st.st_ino, st.st_mtime_ns, st.st_ctime_ns)
    return hashes, stamps


def git(*args: str) -> bytes:
    proc = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True)
    if proc.returncode:
        raise ValueError(os.fsdecode(proc.stderr).strip() or "git failed")
    return proc.stdout


def freeze(args) -> int:
    import numpy

    path = args.run / "freeze.json"
    unused(path)
    if git("rev-parse", "--show-prefix").strip():
        raise ValueError("freeze requires scripts/blind.py from the Git repository root")
    package = package_dir()
    hashes, _ = detector_snapshot(package)
    scope = ("--", f":(glob){package.relative_to(ROOT).as_posix()}/**/*.py", SCORER.relative_to(ROOT).as_posix())
    indexed = git("ls-files", "-v", "-z", *scope).split(b"\0")[:-1]
    # H is an ordinary tracked file. Git can hide edits on assume-unchanged or
    # skip-worktree entries. The pathspec also catches staged deletions of .py files.
    if (set(indexed) != {os.fsencode("H " + name) for name in hashes}
            or git("status", "--porcelain", "--untracked-files=no", *scope)):
        raise ValueError("freeze requires tracked, unmodified detector files")
    commit = git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
    from llmscan import cli

    cli._create_private_dirs(cli._refuse_shared_location(args.run))
    publish(path, {"commit": commit, "files": hashes, "python": platform.python_version(),
                   "numpy": numpy.__version__, "utc": utc_now()})
    print(f"wrote {path}")
    return 0


def read_record(path: Path, missing: str) -> tuple[dict, str]:
    if not path.is_file():
        raise ValueError(missing)
    data = path.read_bytes()
    record = json.loads(data)
    if not isinstance(record, dict):
        raise ValueError(f"invalid {path.name}")
    return record, hashlib.sha256(data).hexdigest()


def verify_detector(record: dict, package: Path) -> dict:
    import numpy

    if not isinstance(record.get("files"), dict) or not isinstance(record.get("commit"), str):
        raise ValueError("invalid detector freeze")
    hashes, stamps = detector_snapshot(package)
    if hashes.keys() != record["files"].keys():
        raise ValueError("detector file set differs from freeze.json")
    for name, digest in hashes.items():
        if digest != record["files"][name]:
            raise ValueError(f"detector hash mismatch: {name}")
    if (platform.python_version(), numpy.__version__) != (record["python"], record["numpy"]):
        raise ValueError("Python or NumPy version differs from freeze.json")
    return stamps


def key_commitment(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("key commitment must be 64 hex characters")
    return value.lower()


def rank(args) -> int:
    path = args.run / "frozen-ranking.json"
    unused(path)
    commitment = key_commitment(args.key_sha256)
    scan = args.run / "scan"
    unused(scan)
    freeze_path = args.run / "freeze.json"
    record, freeze_sha = read_record(freeze_path, "no detector freeze")
    package = package_dir()
    before = verify_detector(record, package)
    from llmscan import cli

    cli._refuse_shared_location(args.run)
    scan.mkdir(mode=0o700)  # exclusively claim this attempt, even if the scan later fails
    # -P excludes a shadow package in cwd. -B and a fresh cache prefix force
    # source imports even after a same-size edit with a stale timestamp-based .pyc.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(package.parent), env.get("PYTHONPATH")]))
    proc = subprocess.run([sys.executable, "-P", "-B", "-X", f"pycache_prefix={scan}", "-X", "utf8",
                           "-m", "llmscan", "--base", args.base, "--suspect", args.suspect, "--out", str(scan)],
                          env=env, stderr=subprocess.PIPE, encoding="utf-8", errors="replace")
    if before != verify_detector(record, package):
        raise ValueError("detector changed during scan")
    if sha256(freeze_path) != freeze_sha:
        raise ValueError("freeze.json changed during scan")
    if proc.returncode:
        raise ValueError(proc.stderr.strip().removeprefix("llmscan: ") or f"llmscan exited {proc.returncode}")
    publish(path, {"ranking_sha256": sha256(scan / "ranking.csv"), "key_sha256": commitment,
                   "freeze_sha256": freeze_sha, "utc": utc_now()})
    if proc.stderr:
        print("blind:", " ".join(proc.stderr.strip().splitlines()), file=sys.stderr)
    print(f"wrote {path}")
    return 0


def score(args) -> int:
    result = args.run / "result.json"
    unused(result)
    ranked, _ = read_record(args.run / "frozen-ranking.json", "no frozen ranking")
    record, freeze_sha = read_record(args.run / "freeze.json", "no detector freeze")
    if freeze_sha != ranked["freeze_sha256"]:
        raise ValueError("freeze.json hash mismatch")
    ranking = args.run / "scan" / "ranking.csv"
    ranking_sha = sha256(ranking)
    if ranking_sha != ranked["ranking_sha256"]:
        raise ValueError("ranking.csv hash mismatch")
    commitment = key_commitment(ranked["key_sha256"])
    package = package_dir()
    before = verify_detector(record, package)
    from llmscan import cli

    cli._refuse_shared_location(args.run)
    if not args.key.is_file():  # first access to the key; a pipe cannot be reopened for scoring
        raise ValueError(f"key is not a regular file: {args.key}")
    key_sha = sha256(args.key)
    if key_sha != commitment:
        raise ValueError("key SHA-256 does not match the commitment")
    proc = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", str(SCORER),
                           "--ranking", str(ranking), "--key", str(args.key)],
                          capture_output=True, text=True, encoding="utf-8")
    # The scorer reopens these paths. Withhold its output if an edit crossed that read.
    if sha256(args.key) != key_sha or sha256(ranking) != ranking_sha or before != verify_detector(record, package):
        raise ValueError("scoring inputs changed while the scorer ran")
    publish(result, {"commit": record["commit"], "ranking_sha256": ranking_sha, "key_sha256": key_sha,
                     "scorer_exit_code": proc.returncode, "scorer_output": proc.stdout.splitlines() + proc.stderr.splitlines()})
    print(proc.stdout, end="")
    if proc.stderr:
        print("blind:", " ".join(proc.stderr.splitlines()), file=sys.stderr)
    return proc.returncode


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    sys.dont_write_bytecode = True  # even a refusal must leave the detector checkout alone
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = Parser(prog="blind", description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in (freeze, rank, score):
        sub = commands.add_parser(action.__name__, allow_abbrev=False)
        sub.set_defaults(action=action)
        sub.add_argument("--run", required=True, type=Path)
        if action is rank:
            sub.add_argument("--base", required=True)
            sub.add_argument("--suspect", required=True)
            sub.add_argument("--key-sha256", required=True)
        if action is score:
            sub.add_argument("--key", required=True, type=Path)
    try:
        args = parser.parse_args(argv)
        args.run = args.run.resolve()
        return args.action(args)
    except SystemExit as exc:  # llmscan reports user errors this way; keep argparse's help exit.
        if not isinstance(exc.code, str):
            raise
        message = exc.code.removeprefix("llmscan: ")
    except Exception as exc:
        message = str(exc)
    print("blind:", " ".join(message.splitlines()), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
