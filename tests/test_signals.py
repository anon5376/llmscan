"""Tests for magnitude, spectral shape, row participation and residual direction."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import gguf_float, write_gguf, write_safetensors
from llmscan import core, rank
from llmscan.names import stack_of
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
        self.assertIsNone(stats.projection_error)
        self.assertAlmostEqual(stats.second_tail_share, 1 / 31)

    def test_signed_partial_projection_fit_for_wide_and_tall_weights(self):
        rng = np.random.default_rng(5)
        for shape in [(8, 15), (15, 8)]:
            w = rng.normal(size=shape)
            axis = rng.normal(size=shape[0])
            axis /= np.linalg.norm(axis)
            projected = np.outer(axis, axis @ w)
            for strength, expected in [(0.3, 0.), (1., 0.), (1.5, 1 / 3), (-1., 1.)]:
                with self.subTest(shape=shape, strength=strength):
                    stats = core.analyze_delta(w, w - strength * projected, 1)
                    self.assertAlmostEqual(stats.projection_error, expected)

    def test_incompatible_coefficients_at_unit_gain(self):
        stats = core.analyze_delta(np.array([[1., 0.]]), np.array([[1., 1.]]), 1)
        self.assertEqual(stats.projection_gain, 1.)
        self.assertEqual(stats.projection_error, 1.)

    def test_local_column_error_with_dense_background(self):
        w = np.ones((1, 16384))
        w[:, 1] = 4.  # an unusually large but genuine base coefficient
        clean = np.zeros_like(w)
        self.assertLess(core.analyze_delta(w, clean.copy(), 1).column_spike, 1e-10)
        clean[:, -1] = 3.
        measured = core.analyze_delta(w, clean, 1)
        self.assertGreater(measured.column_fraction, 0.9)
        self.assertLess(measured.projection_error, 0.05)
        self.assertGreater(measured.column_spike, 1.)
        name = "model.embed_tokens.weight"
        stats = Measurements(measured.effective_rank, measured.singular_values[0], measured.frobenius,
                             measured.projection_gain, measured.second_tail_share, measured.column_fraction,
                             measured.projection_error, measured.column_spike)
        finding, = rank.rank_tensors([row(name)], {name: stats})
        self.assertEqual(finding.tier, "T3")
        self.assertIn("token-row residual spike", finding.reason)


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

    def test_gain_compared_to_peer_dispersion_even_in_older_reports(self):
        self.stats[self.last] = Measurements(1.001, 1.25, 1.25, 1.25)
        self.assertIn("projection_gain", self.flagged()[self.last])
        self.stats[self.last] = self.stats[self.last]._replace(projection_gain=1.)
        self.assertEqual(self.flagged(), {})  # low effective rank alone is consistent

    def test_projection_law_compared_to_nonleading_energy(self):
        self.stats[self.last] = Measurements(1., 1., 1., 1., projection_error=0.5)
        self.assertIn("projection law", self.flagged()[self.last])
        self.stats[self.last] = self.stats[self.last]._replace(frobenius=1.1)
        self.assertEqual(self.flagged(), {})

    def test_invalid_new_stats_are_unusable_but_a_perfect_fit_is_valid(self):
        for changes in [dict(projection_error=-0.01), dict(projection_error=1.01),
                        dict(token_row_spike=float("inf")), dict(token_row_spike=-1.)]:
            with self.subTest(changes=changes):
                self.stats[self.last] = Measurements(1., 1., **changes)
                self.assertIn("unusable", self.flagged()[self.last])
        self.stats[self.last] = Measurements(1., 1., projection_error=0., token_row_spike=0.)
        self.assertEqual(self.flagged(), {})

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

    def test_hf_and_gguf_embedding_stack_names(self):
        for embed, writer, namespace in [("model.embed_tokens.weight", layer_name(0), "model"),
                                          ("token_embd.weight", "blk.0.ffn_down.weight", "")]:
            self.assertEqual(stack_of(embed), namespace)
            self.assertEqual(stack_of(writer), namespace)

    def test_root_output_head_uses_only_an_unambiguous_stack(self):
        head = row("lm_head.weight")
        self.rows.append(head)
        self.stats[head.name] = Measurements(1., 1.)
        axes = {r.name: np.array([1., 0.]) for r in self.rows}
        axes[head.name] = np.array([0., 1.])
        def result():
            return next(r for r in rank.rank_tensors(self.rows, self.stats, directions=axes) if r.name == head.name)
        self.assertEqual((result().tier, result().direction_cosine), ("T2", 0.))
        for i in range(4):
            extra = row(f"encoder.layers.{i}.o_proj.weight")
            self.rows.append(extra)
            self.stats[extra.name] = Measurements(1., 1.)
            axes[extra.name] = np.array([0., 1.])
        self.assertIsNone(result().direction_cosine)

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
    def test_cli_reports_external_heads_residual_axis_and_location(self):
        rng = np.random.default_rng(6)
        head = "lm_head.weight"
        w = rng.normal(size=(64, 8))
        base = {head: ("F32", w)}
        suspect = {head: ("F32", w.copy())}
        suspect[head][1][:, 0] = 0
        for i in range(4):
            name = layer_name(i)
            w = rng.normal(size=(8, 12))
            base[name] = ("F32", w)
            edited = w.copy()
            edited[0] = 0
            suspect[name] = ("F32", edited)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_safetensors(tmp / "base.safetensors", base)
            write_safetensors(tmp / "suspect.safetensors", suspect)
            rc, stdout, stderr = run_cli("--base", str(tmp / "base.safetensors"),
                                         "--suspect", str(tmp / "suspect.safetensors"), "--out", str(tmp / "out"))
            self.assertEqual(rc, 0, stdout + stderr)
            with (tmp / "out" / "ranking.csv").open() as f:
                finding = next(r for r in csv.DictReader(f) if r["name"] == head)
            self.assertEqual(finding["tier"], "T2")
            self.assertAlmostEqual(float(finding["direction_cosine"]), 1.)

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
