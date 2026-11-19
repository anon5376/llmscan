import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import write_safetensors
from llmscan.checkpoint import open_checkpoint
from llmscan.direction import alignment_scores, analyze_direction
from test_core import to_bf16

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_directions.py"


def unit(rng, size):
    v = rng.standard_normal(size)
    return v / np.linalg.norm(v)


def projection(rng, axis, width):
    base = to_bf16(rng.standard_normal((len(axis), width)) * 0.02)
    return base, to_bf16(base - axis[:, None] * (axis @ base))


def write_projection_pair(root):
    """Small BF16 fixture for the measurement CLI, including an unchanged mixed-dtype control."""
    rng = np.random.default_rng(73)
    axis = unit(rng, 64)
    for directory in ("base", "suspect"):
        (root / directory).mkdir()

    def write_pair(name, a, b, dtype="BF16"):
        shard = name + ".safetensors"
        write_safetensors(root / "base" / shard, {name: ("BF16", a)})
        write_safetensors(root / "suspect" / shard, {name: (dtype, b)})

    write_pair("model.layers.0.self_attn.q_proj.weight", np.eye(64), np.eye(64), "F32")
    for layer in (0, 14, 27):
        for module, width in (("self_attn.o_proj", 64), ("mlp.down_proj", 160)):
            name = f"model.layers.{layer}.{module}.weight"
            write_pair(name, *projection(rng, axis, width))


class TestDirection(unittest.TestCase):
    def test_one_other_direction_after_bf16_save(self):
        rng = np.random.default_rng(17)
        shared, other = unit(rng, 64), unit(rng, 64)
        vectors = []
        for i in range(13):
            base, edited = projection(rng, shared if i < 12 else other, 96)
            stat = analyze_direction(base, edited, "down_proj")
            vectors.append((-1) ** i * stat.vector)  # an SVD can choose either sign
        scores = alignment_scores(vectors)
        self.assertGreaterEqual(min(scores[:12]), 0.99)
        self.assertLess(scores[12], 0.20)

    def test_slow_rotation_has_no_isolated_layer(self):
        rng = np.random.default_rng(22)
        plane, _ = np.linalg.qr(rng.standard_normal((32, 2)))
        vectors = []
        for angle in np.linspace(-0.6, 0.6, 25):
            direction = plane @ [np.cos(angle), np.sin(angle)]
            base, edited = projection(rng, direction, 48)
            vectors.append(analyze_direction(base, edited, "o_proj").vector)
        scores = alignment_scores(vectors)
        self.assertGreater(min(scores), 0.94)
        self.assertLess(np.ptp(scores), 0.02)

    def test_residual_side_follows_kind_and_survives_transposed_aspect_ratio(self):
        rng = np.random.default_rng(8)
        for shape in ((9, 23), (23, 9)):
            left, right = (unit(rng, n) for n in shape)
            change = np.outer(left, right)
            for kind, expected in (("o_proj", left), ("down_proj", left), ("embed_tokens", right),
                                   ("attn_output", left), ("ffn_down", left), ("token_embd", right)):
                with self.subTest(shape=shape, kind=kind):
                    stat = analyze_direction(np.zeros(shape), change, kind)
                    self.assertEqual(stat.vector.shape, expected.shape)
                    self.assertAlmostEqual(abs(stat.vector @ expected), 1.0)

    def test_scaled_block_gram_matches_svd(self):
        rng = np.random.default_rng(3)
        for shape, scale in (((17, 11), 1.0), ((531, 7), 1e-180), ((7, 531), 1e180)):
            with self.subTest(shape=shape, scale=scale):
                a, b = rng.standard_normal((2, *shape)) * scale
                delta = (b - a) / scale
                u, s, _ = np.linalg.svd(delta, full_matrices=False)
                stat = analyze_direction(a, b, "down_proj")
                self.assertAlmostEqual(abs(stat.vector @ u[:, 0]), 1.0, places=10)
                self.assertAlmostEqual(stat.energy_share, s[0] ** 2 / (s @ s), places=11)
                self.assertAlmostEqual(stat.delta_norm / scale, np.linalg.norm(s), places=10)

    def test_unusable_deltas_do_not_vote_or_receive_scores(self):
        a = np.zeros((8, 5))
        for value, kind, reason in ((0, "o_proj", "zero delta"), (np.nan, "o_proj", "non-finite delta"),
                                    (np.inf, "o_proj", "non-finite delta"), (-np.inf, "o_proj", "non-finite delta"),
                                    (1, "q_proj", "unknown residual side")):
            with self.subTest(value=value, kind=kind):
                b = a.copy()
                b[-1, -1] = value
                stat = analyze_direction(a, b, kind)
                self.assertIsNone(stat.vector)
                self.assertTrue(np.isnan(stat.energy_share))
                self.assertEqual(stat.reason, reason)
                self.assertTrue(np.isnan(alignment_scores([np.eye(8)[0], stat.vector])).all())

    def test_near_half_and_equal_split(self):
        a, b = np.eye(2)
        scores = alignment_scores([a] * 7 + [b] * 6 + [np.ones(1)])
        np.testing.assert_allclose(scores[:7], 1)
        np.testing.assert_allclose(scores[7:13], 0)
        self.assertTrue(np.isnan(scores[13]))  # no peer with this residual width
        np.testing.assert_allclose(alignment_scores([a] * 6 + [b] * 6), 0)


class TestDirectionCLI(unittest.TestCase):
    def test_changed_only_csv_and_independent_bf16_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_projection_pair(root)
            cmd = [sys.executable, str(SCRIPT), "--base", str(root / "base"), "--suspect", str(root / "suspect"),
                   "--out", str(root / "report"), "--probe-layers", "0", "14", "27"]
            run = subprocess.run(cmd, check=True, capture_output=True, text=True)
            with (root / "report" / "directions.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertIn(row["kind"], ("o_proj", "down_proj"))
                self.assertGreater(float(row["alignment"]), 0.99)
                self.assertGreater(float(row["energy_share"]), 0.99)
            trials = [line.split() for line in run.stdout.splitlines() if line.startswith("probe ")]
            self.assertEqual({p[1] for p in trials}, {r["name"] for r in rows})
            # Reconstruct the first saved replacement independently from its two checkpoints.
            name = "model.layers.0.mlp.down_proj.weight"
            with open_checkpoint(root / "base") as a, open_checkpoint(root / "suspect") as b:
                w = a.to_float64(name)
                norm = np.linalg.norm(b.to_float64(name) - w)
            rng = np.random.default_rng(0)
            edit = norm * np.outer(unit(rng, w.shape[0]), unit(rng, w.shape[1]))
            s = np.linalg.svd(to_bf16(w + edit) - w, compute_uv=False)
            for trial in trials:
                fields = dict(s.split("=") for s in trial[2:])
                self.assertEqual(fields["rank"], "1/6")
                self.assertLess(float(fields["alignment"]), 0.50)
                self.assertGreater(float(fields["other_min"]), 0.99)
                self.assertAlmostEqual(float(fields["saved_norm/original"]), 1.0, delta=0.01)
                if trial[1] == name:
                    self.assertAlmostEqual(float(fields["saved_norm/original"]), np.linalg.norm(s) / norm, places=7)
                    self.assertAlmostEqual(float(fields["energy_share"]), s[0] ** 2 / (s @ s), places=7)

    def test_refuses_model_output_including_file_and_directory_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_projection_pair(root)
            (root / "alias").symlink_to(root / "base", target_is_directory=True)
            destinations = [root / "alias" / "report", root / "suspect"]
            if (root / "BASE").is_dir():  # case-insensitive volume
                destinations.append(root / "BASE" / "report")
            shard = next((root / "suspect").glob("*.safetensors"))
            for out in destinations:
                with self.subTest(out=out):
                    cmd = [sys.executable, str(SCRIPT), "--base", str(root / "base"),
                           "--suspect", str(shard), "--out", str(out)]
                    run = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("--out is inside the model directory", run.stderr)
                    self.assertFalse((out / "directions.csv").exists())


if __name__ == "__main__":
    unittest.main()
