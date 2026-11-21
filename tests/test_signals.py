"""Tests for magnitude, spectral shape, row participation and residual direction."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import gguf_float, write_gguf, write_safetensors
from llmscan import core, rank
from llmscan.signals import Measurements
from test_core import to_bf16
from test_rank import layer_name, row, run_cli


class TestProjectionMeasurements(unittest.TestCase):
    def test_direction_and_magnitude_on_both_sides_of_the_gram(self):
        rng = np.random.default_rng(4)
        for n, d in [(71, 17), (17, 71), (17, 17)]:
            with self.subTest(shape=(n, d)):
                w = rng.normal(size=(n, d))
                axis = rng.normal(size=n)
                axis /= np.linalg.norm(axis)
                change = -np.outer(axis, axis @ w)
                stats = core.analyze_delta(w, w + change, 1)
                self.assertAlmostEqual(abs(stats.direction @ axis), 1.)
                self.assertAlmostEqual(stats.projection_gain, 1.)
                self.assertAlmostEqual(stats.frobenius, np.linalg.norm(change))
                energy = np.sum(change ** 2, axis=0)
                self.assertAlmostEqual(stats.column_fraction, energy.sum() ** 2 / (d * (energy @ energy)))

    def test_degenerate_leading_space_has_no_direction_or_gain(self):
        stats = core.analyze_delta(np.zeros((32, 32)), np.eye(32), 1)
        self.assertIsNone(stats.direction)
        self.assertIsNone(stats.projection_gain)
        self.assertAlmostEqual(stats.second_tail_share, 1 / 31)


class TestAnomalySignals(unittest.TestCase):
    def setUp(self):
        self.rows = [row(layer_name(i)) for i in range(10)]
        self.stats = {r.name: Measurements(1.005, 1., 1.002, 1., 0.03) for r in self.rows}
        self.last = self.rows[-1].name

    def flagged(self, **kwargs):
        return {r.name: r.reason for r in rank.rank_tensors(self.rows, self.stats, **kwargs) if r.tier == "T3"}

    def test_gain_outlier_with_lower_effective_rank(self):
        self.stats[self.last] = Measurements(1.0001, 4., 4., 4., 0.03)
        self.assertIn("projection_gain", self.flagged()[self.last])
        self.stats[self.last] = self.stats[self.last]._replace(projection_gain=1.)
        self.assertEqual(self.flagged(), {})  # large natural variation in the base projection

    def test_magnitude_fallback_for_legacy_csv_or_missing_axis(self):
        self.stats[self.last] = Measurements(1.0, 4., 4.)
        self.assertIn("frobenius", self.flagged()[self.last])
        self.stats = {n: m[:2] for n, m in self.stats.items()}
        self.assertIn("sv_001", self.flagged()[self.last])

    def test_subtle_second_mode_and_diffuse_noise(self):
        diagonal = np.zeros(32)
        diagonal[:2] = [1., 0.2]

        def measure():
            # The full spectrum drives the metric even if the report saves one SV.
            s = core.analyze_delta(np.eye(32), np.eye(32) + np.diag(diagonal), 1)
            return Measurements(s.effective_rank, s.singular_values[0], s.frobenius, s.projection_gain, s.second_tail_share)

        self.stats[self.last] = measure()
        self.assertIn("second mode", self.flagged()[self.last])
        diagonal[1:] = 0.2 / np.sqrt(31)  # identical non-leading energy spread across 31 modes
        self.stats[self.last] = measure()
        self.assertEqual(self.flagged(), {})
        self.stats[self.last] = self.stats[self.last]._replace(second_tail_share=0.)
        self.assertEqual(self.flagged(), {})

    def test_direction_only_sign_invariance_and_no_consensus(self):
        eye = np.eye(12)
        axes = {r.name: (-1) ** i * eye[0] for i, r in enumerate(self.rows)}
        axes[self.last] = eye[2]
        self.assertEqual(set(self.flagged(directions=axes)), {self.last})
        axes = {r.name: eye[i] for i, r in enumerate(self.rows)}
        self.assertEqual(self.flagged(directions=axes), {})

    def test_dense_embedding_compared_to_projection_layers(self):
        name = "model.embed_tokens.weight"
        self.rows.append(row(name))
        self.stats[name] = Measurements(1., 1., 1., 1., 0., 0.4)
        axes = {r.name: np.array([1., 0.]) for r in self.rows}
        axes[name] = np.array([0., 1.])
        self.assertEqual(set(self.flagged(directions=axes)), {name})

    def test_separate_stacks_and_widths_do_not_vote_against_one_another(self):
        axes = {}
        self.rows = []
        for stack, direction in [("encoder", [1., 0.]), ("decoder", [0., 1.]), ("vision", [0., 0., 1.])]:
            for i in range(10):
                r = row(f"{stack}.layers.{i}.o_proj.weight")
                self.rows.append(r)
                axes[r.name] = np.array(direction)
                self.stats[r.name] = Measurements(1., 1.)
        ranked = rank.rank_tensors(self.rows, self.stats, directions=axes)
        self.assertTrue(all(r.direction_cosine == 1 and r.tier == "T4" for r in ranked))

    def test_single_embedding_has_independent_magnitude_and_concentration_checks(self):
        for kind in ("embed_tokens", "token_embd"):
            name = f"{kind}.weight"
            self.rows = [row(name)]
            self.stats[name] = Measurements(1., 4., 4., 5., 0., 0.4)
            self.assertIn("full projection", self.flagged()[name])
            self.stats[name] = self.stats[name]._replace(projection_gain=1., token_row_fraction=0.01)
            self.assertIn("token-row", self.flagged()[name])
            self.stats[name] = self.stats[name]._replace(token_row_fraction=0.4)
            self.assertEqual(self.flagged(), {})

    def test_local_gain_bound_survives_a_shifted_median(self):
        self.stats = {r.name: Measurements(1.0, 4., 4., 4.) for r in self.rows}
        self.assertEqual(set(self.flagged()), set(self.stats))


class TestControlCheckpoints(unittest.TestCase):
    def test_honest_bf16_projections_and_embedding_axis_through_cli(self):
        rng = np.random.default_rng(42)
        axis = np.ones(16) / 4
        for fmt in ("safetensors", "gguf"):
            with self.subTest(format=fmt), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                embed = "model.embed_tokens.weight" if fmt == "safetensors" else "token_embd.weight"
                base = {embed: to_bf16(rng.normal(size=(128, 16)))}
                for i in range(9):
                    n = layer_name(i) if fmt == "safetensors" else f"blk.{i}.ffn_down.weight"
                    base[n] = to_bf16(rng.normal(size=(16, 48)) * (1 + i))
                edited = {}
                for name, w in base.items():
                    if name == embed:
                        edited[name] = to_bf16(w - np.outer(w @ axis, axis))
                    else:
                        edited[name] = to_bf16(w - np.outer(axis, axis @ w))
                for label, weights in [("base", base), ("suspect", edited)]:
                    path = tmp / f"{label}.{fmt}"
                    if fmt == "safetensors":
                        write_safetensors(path, {n: ("BF16", a) for n, a in weights.items()})
                    else:
                        write_gguf(path, {n: gguf_float("BF16", a) for n, a in weights.items()})
                rc, stdout, stderr = run_cli("--base", str(tmp / f"base.{fmt}"), "--suspect", str(tmp / f"suspect.{fmt}"),
                                             "--out", str(tmp / "out"), "-k", "1")
                self.assertEqual(rc, 0, stdout + stderr)
                with (tmp / "out" / "ranking.csv").open() as f:
                    ranked = list(csv.DictReader(f))
                self.assertEqual(len(ranked), len(base))
                self.assertTrue(all(r["tier"] == "T4" and float(r["direction_cosine"]) > 0.99 for r in ranked))
                with (tmp / "out" / "delta.csv").open() as f:
                    measured = {r["name"]: r for r in csv.DictReader(f)}
                self.assertGreater(float(measured[embed]["token_row_fraction"]), 0.1)
