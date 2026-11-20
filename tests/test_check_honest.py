import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import tiny_llama, write_sharded

ROOT = Path(__file__).resolve().parents[1]


class TestCheckHonest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.models = self.root / "models"
        base = tiny_llama(np.random.default_rng(7), layers=3)
        write_sharded(self.models / "base", [base])
        suspect = dict(base)
        for name, (dtype, w) in base.items():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                suspect[name] = (dtype, w + 1 / 16)  # a constant matrix has rank one
        write_sharded(self.models / "honest", [suspect])
        q = "model.layers.0.self_attn.q_proj.weight"
        suspect[q] = (base[q][0], base[q][1] + 1 / 16)
        write_sharded(self.models / "flag", [suspect])

    def gate(self, pairs, out=None):
        path = self.root / "pairs.txt"
        path.write_text(pairs, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_honest.py"), "--models", str(self.models),
             "--pairs-file", str(path), "--out", str(out or self.root / "out")],
            cwd=self.root, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True, text=True,
        )

    def assert_row(self, result, suspect, counts, verdict):
        rows = [line.split("\t") for line in result.stdout.splitlines() if line.startswith(suspect + "\t")]
        self.assertEqual(len(rows), 1, result.stdout)
        _, *cells, seconds, actual = rows[0]
        self.assertEqual((cells, actual), (counts.split(), verdict))
        self.assertGreaterEqual(float(seconds), 0)

    def test_rank_one_edits_are_clean(self):
        result = self.gate("# release gate\n\nbase honest # rank-one edits\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_row(result, "honest", "6 0 0 0 6", "CLEAN")
        self.assertTrue((self.root / "out/honest-vs-base/delta.csv").is_file())

    def test_q_proj_is_flagged(self):
        result = self.gate("base flag")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assert_row(result, "flag", "7 0 1 0 6", "FLAGGED")

    def test_missing_pair_continues_and_wins_over_flag(self):
        result = self.gate("base absent\nbase flag")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assert_row(result, "absent", "- - - - -", "MISSING")
        self.assert_row(result, "flag", "7 0 1 0 6", "FLAGGED")
        self.assertLess(result.stdout.index("absent\t"), result.stdout.index("flag\t"))
        self.assertIn("CLEAN=0 FLAGGED=1 MISSING=1 ERROR=0", result.stdout.splitlines()[-1])

    def test_no_changed_tensors_is_flagged(self):
        result = self.gate("base base")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assert_row(result, "base", "0 0 0 0 0", "FLAGGED")

    def test_output_inside_models_is_refused(self):
        out = self.models / "results"
        result = self.gate("base honest", out)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertTrue(result.stdout.startswith("check_honest:"))
        self.assertFalse(out.exists())

    def test_failed_rerun_with_old_clean_ranking_is_error(self):
        first = self.gate("base honest")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        shard, = (self.models / "honest").glob("*.safetensors")
        shard.write_bytes(b"truncated")
        result = self.gate("base honest")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assert_row(result, "honest", "- - - - -", "ERROR")
        diagnostics = [s for s in result.stdout.splitlines() if s.startswith("check_honest:")]
        self.assertEqual(len(diagnostics), 1, result.stdout)
        self.assertIn("llmscan exited", diagnostics[0])
