"""Exercise the blind CLI across fresh Python processes in a disposable Git repository."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from helpers import tiny_llama, write_safetensors

ROOT = Path(__file__).resolve().parents[1]
PLANT = "model.layers.0.self_attn.q_proj.weight"
DECOY = "model.layers.0.mlp.down_proj.weight"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestBlindProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.repo = self.tmp / "repo"
        self.package = self.repo / "src" / "llmscan"
        shutil.copytree(ROOT / "src" / "llmscan", self.package, ignore=shutil.ignore_patterns("__pycache__"))
        (self.repo / "scripts").mkdir()
        for name in ("blind.py", "score_blind.py"):
            shutil.copyfile(ROOT / "scripts" / name, self.repo / "scripts" / name)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        PYTHONPATH=str(self.package.parent), PYTHONDONTWRITEBYTECODE="1",
                        MPLCONFIGDIR=self.env.get("MPLCONFIGDIR", str(self.tmp / "mpl")))
        self.git("init", "--quiet", "--template=")
        self.git("add", "src", "scripts")
        self.git("commit", "--quiet", "-m", "Detector fixture")

        self.run_dir = self.tmp / "run"
        self.key = self.tmp / "key.txt"
        self.key.write_bytes((PLANT + "\n").encode())
        base = tiny_llama(np.random.default_rng(9), layers=1)
        suspect = dict(base)
        for name in (DECOY, PLANT):
            dtype, weight = base[name]
            suspect[name] = (dtype, weight + 1)
        self.base, self.suspect = (self.tmp / name for name in ("base.safetensors", "suspect.safetensors"))
        write_safetensors(self.base, base)
        write_safetensors(self.suspect, suspect)

    def git(self, *args):
        cmd = ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", *args]
        return subprocess.run(cmd, cwd=self.repo, env=self.env, check=True, capture_output=True).stdout

    def blind(self, command, *args):
        return subprocess.run([sys.executable, "-X", "utf8", "scripts/blind.py", command,
                               "--run", str(self.run_dir), *map(str, args)], cwd=self.repo, env=self.env,
                              capture_output=True, text=True, encoding="utf-8")

    def rank(self, key_hash=None):
        return self.blind("rank", "--base", self.base, "--suspect", self.suspect,
                          "--key-sha256", digest(self.key) if key_hash is None else key_hash)

    def ok(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def refused(self, proc, message):
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr.splitlines(), ["blind: " + message])
        self.assertEqual(proc.stdout, "")

    def files(self):
        return {p.relative_to(self.run_dir).as_posix(): p.read_bytes()
                for p in self.run_dir.rglob("*") if p.is_file()}

    def test_rank_refuses_detector_edited_after_freeze(self):
        self.ok(self.blind("freeze"))
        original = self.files()
        for bad in ("g" * 64, "a" * 63, "a" * 65):
            self.refused(self.rank(bad), "key commitment must be 64 hex characters")
        self.refused(self.blind("rank", "--base", self.base, "--suspect", self.suspect, "--key-sha256", "a" * 64,
                                "--key", self.key), f"unrecognized arguments: --key {self.key}")
        with (self.package / "rank.py").open("ab") as f:
            f.write(b"\n# changed detector\n")
        self.refused(self.rank(), "detector hash mismatch: src/llmscan/rank.py")
        self.assertEqual(self.files(), original)
        self.assertFalse((self.run_dir / "scan").exists())

    def test_score_checks_detector_and_both_committed_files_before_key(self):
        self.ok(self.blind("freeze"))
        self.ok(self.rank())
        for path, change, message in (
            (self.repo / "scripts/score_blind.py", b"\n# edit\n", "detector hash mismatch: scripts/score_blind.py"),
            (self.run_dir / "scan/ranking.csv", b"\n", "ranking.csv hash mismatch"),
            (self.run_dir / "freeze.json", b" ", "freeze.json hash mismatch"),
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + change)
                self.refused(self.blind("score", "--key", self.tmp / "missing-key"), message)
                self.assertFalse((self.run_dir / "result.json").exists())
                path.write_bytes(original)
        (self.package / "new_module.py").touch()
        self.refused(self.blind("score", "--key", self.tmp / "missing-key"), "detector file set differs from freeze.json")

    def test_score_before_rank_never_opens_key(self):
        self.ok(self.blind("freeze"))
        self.refused(self.blind("score", "--key", self.tmp / "missing-dir" / "key.txt"), "no frozen ranking")
        self.assertEqual(set(self.files()), {"freeze.json"})

    def test_wrong_key_never_reaches_scorer(self):
        self.ok(self.blind("freeze"))
        self.ok(self.rank())
        self.key.write_bytes(self.key.read_bytes() + b"# same tensor, different bytes\n")
        self.refused(self.blind("score", "--key", self.key), "key SHA-256 does not match the commitment")
        self.assertFalse((self.run_dir / "result.json").exists())

    def test_second_rank_and_score_preserve_first_files(self):
        self.key.write_bytes((DECOY + "\n").encode())
        self.ok(self.blind("freeze"))
        self.ok(self.rank())
        ranked = self.files()
        self.refused(self.blind("freeze"), "freeze.json already exists")
        self.refused(self.rank("0" * 64), "frozen-ranking.json already exists")
        self.assertEqual(self.files(), ranked)
        proc = self.blind("score", "--key", self.key)
        self.assertEqual((proc.returncode, proc.stdout.splitlines()[-1]), (1, "FAIL"))
        scored = self.files()
        result = json.loads(scored["result.json"])
        self.assertEqual((result["scorer_exit_code"], result["scorer_output"]), (1, proc.stdout.splitlines()))
        self.refused(self.rank(), "frozen-ranking.json already exists")
        self.refused(self.blind("score", "--key", self.tmp / "missing-key"), "result.json already exists")
        self.assertEqual(self.files(), scored)

    def test_freeze_requires_clean_tracked_imported_sources(self):
        scorer = self.repo / "scripts" / "score_blind.py"
        original = scorer.read_bytes()
        scorer.write_bytes(original + b"\n# dirty\n")
        dirty = "freeze requires tracked, unmodified detector files"
        self.refused(self.blind("freeze"), dirty)  # unstaged
        self.git("add", "scripts/score_blind.py")
        self.refused(self.blind("freeze"), dirty)  # staged
        scorer.write_bytes(original)
        self.refused(self.blind("freeze"), dirty)  # worktree matches HEAD, index still differs
        self.git("add", "scripts/score_blind.py")
        (self.package / "extra.py").touch()
        self.refused(self.blind("freeze"), dirty)  # untracked Python
        self.env["PYTHONPATH"] = str(self.tmp / "installed")
        shutil.copytree(self.package, self.tmp / "installed" / "llmscan")
        self.refused(self.blind("freeze"),
                     "imported llmscan is outside this checkout or is not a source package; set PYTHONPATH to its src")
        self.assertFalse(self.run_dir.exists())

    def test_planted_tensor_passes_with_independent_receipts(self):
        self.ok(self.blind("freeze"))
        frozen_path = self.run_dir / "freeze.json"
        freeze = json.loads(frozen_path.read_bytes())
        sources = [*self.package.rglob("*.py"), self.repo / "scripts/score_blind.py"]
        self.assertEqual(freeze["files"], {p.relative_to(self.repo).as_posix(): digest(p) for p in sources})
        commit = self.git("rev-parse", "HEAD").decode().strip()
        self.assertEqual((freeze["commit"], freeze["python"], freeze["numpy"]),
                         (commit, platform.python_version(), np.__version__))
        key_hash = digest(self.key)
        self.ok(self.rank(key_hash.upper()))
        ranked = json.loads((self.run_dir / "frozen-ranking.json").read_bytes())
        ranking_hash = digest(self.run_dir / "scan" / "ranking.csv")
        self.assertEqual((ranked["freeze_sha256"], ranked["ranking_sha256"], ranked["key_sha256"]),
                         (digest(frozen_path), ranking_hash, key_hash))
        for record in (freeze, ranked):
            self.assertEqual(datetime.fromisoformat(record["utc"]).utcoffset(), timedelta(0))
        proc = self.blind("score", "--key", self.key)
        self.ok(proc)
        result = json.loads((self.run_dir / "result.json").read_bytes())
        self.assertEqual((result["commit"], result["ranking_sha256"], result["key_sha256"]), (commit, ranking_hash, key_hash))
        self.assertEqual((result["scorer_exit_code"], result["scorer_output"]), (0, proc.stdout.splitlines()))
        self.assertEqual([line.strip() for line in result["scorer_output"]],
                         ["rank 1  " + PLANT, "precision@1 = 1.000", "recall@10 = 1.000", "PASS"])

    def test_detector_edit_during_cli_leaves_unscorable_scan(self):
        # This fixture's plotter edits a detector file while the real CLI is finishing the scan.
        with (self.package / "plots.py").open("a", encoding="utf-8") as f:
            f.write("\nfrom pathlib import Path\nwith Path(__file__).with_name('rank.py').open('ab') as f:\n"
                    "    f.write(b'\\n# mid-scan edit\\n')\n")
        self.git("add", "src/llmscan/plots.py")
        self.git("commit", "--quiet", "-m", "Simulate detector drift during scan")
        self.ok(self.blind("freeze"))
        proc = self.rank()
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr.splitlines(), ["blind: detector hash mismatch: src/llmscan/rank.py"])
        self.assertTrue((self.run_dir / "scan/ranking.csv").is_file())
        self.refused(self.blind("score", "--key", self.tmp / "missing-key"), "no frozen ranking")
        before = self.files()
        self.refused(self.rank(), "scan already exists")
        self.assertEqual(self.files(), before)

    def test_failed_cli_cannot_commit_its_preliminary_ranking(self):
        with (self.package / "core.py").open("a", encoding="utf-8") as f:
            f.write("\ndef analyze_delta(*args, **kwargs):\n    raise NotImplementedError('fixture interruption')\n")
        self.git("add", "src/llmscan/core.py")
        self.git("commit", "--quiet", "-m", "Interrupt the fixture scan")
        self.ok(self.blind("freeze"))
        proc = self.rank()
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stderr.splitlines(), ["blind: ΔW scan stopped: fixture interruption"])
        self.assertTrue((self.run_dir / "scan/ranking.csv").is_file())
        self.refused(self.blind("score", "--key", self.key), "no frozen ranking")


if __name__ == "__main__":
    unittest.main()
