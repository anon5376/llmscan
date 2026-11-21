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

    def test_rank_one_excess_is_scale_invariant_at_normal_f64_extremes(self):
        for scale in [1e-250, 1., 1e250]:
            with self.subTest(scale=scale):
                w = np.diag([scale, scale, 0.])
                stats = core.analyze_delta(w, np.zeros_like(w), 1)
                self.assertAlmostEqual(core.rank_one_excess(w, stats, "F64", "F64"), 1 / np.sqrt(2))

    def test_rank_one_excess_allows_subnormal_rounding(self):
        unit = float(np.finfo(np.float16).smallest_subnormal)
        w = np.eye(4) * unit
        stats = core.analyze_delta(w, np.zeros_like(w), 1)
        self.assertEqual(core.rank_one_excess(w, stats, "F16", "F32"), 0.)
        self.assertEqual(core.rank_one_excess(w, stats, "F32", "F16"), 0.)

    def test_f64_subnormal_projection_uses_scaled_tail_energy(self):
        # Work in integer multiples of the smallest representable weight.
        units = np.zeros((3, 3))
        units[0] = [3, 11, 17]
        axis = np.array([1., 2., 3.]) / np.sqrt(14.)
        projected = np.rint(units - np.outer(axis, axis @ units))
        step = np.finfo(np.float64).smallest_subnormal
        base = units * step
        stats = core.analyze_delta(base, projected * step, 1)
        self.assertEqual(stats.singular_values[0] / step, 5.)
        self.assertEqual(stats.frobenius / step, 6.)
        self.assertEqual(core.rank_one_excess(base, stats, "F64", "F64"), 0.)
        # Both parts of the signed fit use that same scaled spectrum, even
        # when the norms written in physical units round to different steps.
        fit = Measurements(stats.effective_rank, stats.singular_values[0], stats.frobenius,
                           projection_error=stats.projection_error, tail_norm_fraction=stats.tail_norm_fraction)
        regular = core.analyze_delta(units, projected.copy(), 1)
        expected = regular.projection_error * regular.singular_values[0] / regular.frobenius
        self.assertAlmostEqual(fit.projection_residual().leading, expected)
        self.assertAlmostEqual(fit.projection_residual().tail, regular.tail_norm_fraction)

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

    def test_contraction_accounts_for_rounding_of_both_checkpoints(self):
        # A reflection preserves norms in exact arithmetic. These independently
        # rounded checkpoints exhibit *positive* raw growth on the fit probe.
        for dtype, scale, seed in [("BF16", 1., 2), ("F16", 1., 0), ("F16", 1e-5, 1),
                                   ("F64", 8 * np.finfo(np.float64).smallest_subnormal, 0)]:
            with self.subTest(dtype=dtype, scale=scale):
                rng = np.random.default_rng(seed)
                w = rng.normal(size=(8, 11))
                axis = rng.normal(size=len(w))
                axis /= np.linalg.norm(axis)
                base, edited = w * scale, (w - np.outer(axis, axis @ w)) * scale
                if dtype == "BF16":
                    base, edited = to_bf16(base), to_bf16(edited)
                elif dtype == "F16":
                    base, edited = [x.astype(np.float16).astype(np.float32) for x in (base, edited)]
                stats = core.analyze_delta(base, edited, 1)
                z = stats.fit_residual / np.linalg.norm(stats.fit_residual)
                x = (base.astype(float) / scale) @ z
                diff = (edited.astype(float) / scale) @ z
                self.assertGreater(np.linalg.norm(x + 2 * diff), np.linalg.norm(x))
                self.assertEqual(core.contraction_excess(base, edited, stats, dtype, dtype), 0.)
                self.assertEqual(core.reciprocity_excess(base, edited, stats, dtype, dtype), 0.)

    def test_reciprocity_holds_for_symmetric_edits_at_extreme_scales(self):
        base = np.array([[2., 1., 3., 0.5], [1., 0., 2., -1.]])
        suspect = np.array([[0.25, 0.125], [0.125, 0.75]]) @ base
        for scale in [1e-200, 1., 1e200]:
            with self.subTest(scale=scale):
                b, s = scale * base, scale * suspect
                stats = core.analyze_delta(b, s, 1)
                self.assertGreater(stats.projection_error, 0.01)
                self.assertEqual(core.reciprocity_excess(b, s, stats, "F64", "F64"), 0.)

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

    def test_tied_modes_preserve_reciprocity_after_rounding(self):
        rng = np.random.default_rng(101)
        for shape, dim in [((12, 32), 2), ((32, 12), 3)]:
            left = np.linalg.qr(rng.normal(size=(shape[0], dim)))[0]
            right = np.linalg.qr(rng.normal(size=(shape[1], dim)))[0]
            base = rng.normal(size=shape)
            unchanged = base - left @ (left.T @ base)
            base = unchanged + left @ right.T
            for dtype, scale in [("BF16", 1.), ("F64", 1e-220), ("F64", 1e220)]:
                with self.subTest(shape=shape, dtype=dtype, scale=scale):
                    b, s = base * scale, unchanged * scale
                    if dtype == "BF16":
                        b, s = to_bf16(b), to_bf16(s)
                    stats = core.analyze_delta(b, s, 1)
                    self.assertIsNone(stats.direction)
                    self.assertEqual(stats.output_subspace.shape, left.shape)
                    self.assertEqual(core.reciprocity_excess(b, s, stats, dtype, dtype), 0.)

    def test_reciprocity_detects_skew_with_no_separated_axis(self):
        b = np.eye(6)
        # A cyclic permutation has tied singular values and nonzero skew.
        s = b + np.roll(b, 1, axis=0)
        m = core.analyze_delta(b, s, 1)
        self.assertIsNone(m.direction)
        self.assertIsNone(m.output_subspace)  # spans the entire output
        excess = core.reciprocity_excess(b, s, m, "F64", "F64")
        self.assertGreater(excess, 0.99)
        name = layer_name(0)
        finding, = rank.rank_tensors([row(name)], {name: Measurements(6., 1., reciprocity_excess=excess)})
        self.assertIn("multi-mode change violates projection reciprocity", finding.reason)

    def test_removal_certificate_allows_independent_rounding_of_both_checkpoints(self):
        rng = np.random.default_rng(412)
        for modes in [1, 4]:
            b = rng.normal(size=(17, 8))
            u = np.linalg.qr(rng.normal(size=(len(b), modes)))[0]
            p = (u * np.linspace(1., 0.25, modes)) @ u.T
            s = b - p @ b
            for dtype, save in [("BF16", to_bf16), ("F16", np.float16)]:
                with self.subTest(modes=modes, dtype=dtype):
                    stored_b, stored_s = save(b), save(s)
                    form = stored_s.astype(float).T @ (stored_s.astype(float) - stored_b)
                    self.assertGreater(np.linalg.eigvalsh((form + form.T) / 2)[-1], 0.)
                    self.assertEqual(core.removal_excess(stored_b, stored_s, dtype, dtype), 0.)

    def test_removal_certificate_checks_both_ends_of_strength_interval(self):
        for scale in [1e-250, 1., 1e250]:
            b = scale * np.eye(4)
            for strength in [-0.25, 0., 0.5, 1., 1.25]:
                with self.subTest(scale=scale, strength=strength):
                    excess = core.removal_excess(b, (1 - strength) * b, "F64", "F64")
                    if 0 <= strength <= 1:
                        self.assertEqual(excess, 0.)
                    else:
                        self.assertGreater(excess, 0.99)
        unit = np.finfo(float).smallest_subnormal
        # Rounded half-removal from 3 subnormal units to 2.
        self.assertEqual(core.removal_excess(np.eye(3) * 3 * unit, np.eye(3) * 2 * unit, "F64", "F64"), 0.)
        self.assertEqual(core.removal_excess(np.eye(3) * unit, np.zeros((3, 3)), "F64", "F16"), 0.)


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

    def test_systematic_departure_from_projection_law(self):
        for name in self.stats:
            self.stats[name] = Measurements(1., 1., 1., 0.7, projection_error=0.6, reciprocity_excess=0.5)
        self.assertEqual(self.flagged(), {})
        self.stats[self.last] = self.stats[self.last]._replace(projection_error=0.9)
        self.assertIn("projection law", self.flagged()[self.last])

    def test_multimode_reciprocity_against_a_reference_population(self):
        self.stats = {r.name: Measurements(3., 1., reciprocity_excess=0.4) for r in self.rows}
        self.assertEqual(self.flagged(), {})
        self.stats[self.last] = self.stats[self.last]._replace(reciprocity_excess=0.8)
        self.assertEqual(set(self.flagged()), {self.last})
        self.assertIn("projection reciprocity", self.flagged()[self.last])

    def test_removal_violation_can_be_independent_of_the_leading_fit(self):
        self.stats = {r.name: Measurements(1.1, 1., projection_error=0., removal_excess=0.4) for r in self.rows}
        self.assertEqual(self.flagged(), {})
        self.stats[self.last] = self.stats[self.last]._replace(removal_excess=0.9)
        self.assertEqual(set(self.flagged()), {self.last})

        # Recurrent departures can coexist with a majority below the rounding
        # bound. Clipping at zero gives this cohort a zero MAD.
        for i, name in enumerate(self.stats):
            self.stats[name] = self.stats[name]._replace(removal_excess=0.4 if i % 3 == 0 else 0.)
        self.assertEqual(self.flagged(), {})

    def test_contraction_witness_with_and_without_a_reference_population(self):
        # Some published editing methods systematically depart from partial
        # removal. An unusual signed fit and a contraction violation corroborate
        # one another when a reference distribution exists.
        self.stats = {r.name: Measurements(1.1, 1., 1.1, projection_error=0.1, contraction_excess=0.01)
                      for r in self.rows}
        self.assertEqual(self.flagged(), {})
        self.stats[self.last] = self.stats[self.last]._replace(projection_error=0.11)
        self.assertIn("contraction", self.flagged()[self.last])
        self.rows = self.rows[:3]
        self.assertEqual(set(self.flagged()), {r.name for r in self.rows})

    def test_large_fit_error_proportional_to_diffuse_noise(self):
        # Both residual components grow together in a noisier layer.
        self.stats = {r.name: Measurements(1., 1., np.hypot(1., 0.08), projection_error=0.02)
                      for r in self.rows}
        self.stats[self.last] = Measurements(1., 1., np.hypot(1., 0.4), projection_error=0.1)
        self.assertEqual(self.flagged(), {})

    def test_small_projection_misfit_scales_with_peer_noise(self):
        # Representative BF16 fit and spectral tail from Round 6. Reduce both
        # noise and signal together to exercise a more precise cohort as well.
        for dtype, scale in (("BF16", 1.), ("F16", 0.1)):
            with self.subTest(dtype=dtype):
                tail = 0.047 * scale
                for i, r in enumerate(self.rows):
                    r.base_dtype = r.suspect_dtype = dtype
                    self.stats[r.name] = Measurements(1.0044, 1., np.hypot(1., tail),
                                                      projection_error=(0.005 + 0.00004 * i) * scale)
                self.stats[self.last] = self.stats[self.last]._replace(projection_error=0.0481 * scale)
                flagged = self.flagged()
                self.assertEqual(set(flagged), {self.last})
                self.assertIn("residual norm", flagged[self.last])
                # The same small error has insufficient support on its own.
                self.rows[-1].suspect_dtype = "F32"
                self.assertEqual(self.flagged(), {})

    def test_peer_noise_includes_typical_fit_error_and_dispersion(self):
        # An honest attention layer can exceed three times the median alone.
        # Its error stays within the same multiple of the upper noise level.
        for i, r in enumerate(self.rows):
            self.stats[r.name] = Measurements(1.005, 1., np.hypot(1., 0.05),
                                              projection_error=0.005 + i * 0.0001)
        for error in (0.01, 0.0166):
            with self.subTest(error=error):
                self.stats[self.last] = self.stats[self.last]._replace(projection_error=error)
                self.assertEqual(self.flagged(), {})

    def test_peer_share_from_numerical_residuals_is_insufficient(self):
        self.stats = {r.name: Measurements(1., 1., 1., projection_error=1e-16, tail_norm_fraction=1e-14)
                      for r in self.rows}
        self.stats[self.last] = self.stats[self.last]._replace(projection_error=1e-14)
        self.assertEqual(self.flagged(), {})

    def test_residual_share_needs_peers_at_matching_precision(self):
        self.stats = {r.name: Measurements(1., 1., 1.02, projection_error=0.01) for r in self.rows}
        self.stats[self.last] = self.stats[self.last]._replace(projection_error=0.2)
        self.assertIn("residual norm", self.flagged()[self.last])
        self.rows[-1].suspect_dtype = "F16"
        self.assertEqual(self.flagged(), {})

    def test_exact_fits_do_not_supply_residual_share_measurements(self):
        self.stats = {r.name: Measurements(1., 1., 1., projection_error=0.) for r in self.rows}
        self.stats[self.last] = Measurements(1., 1., 1.02, projection_error=0.1)
        self.assertEqual(self.flagged(), {})

    def test_invalid_new_stats_are_unusable_but_a_perfect_fit_is_valid(self):
        for changes in [dict(projection_error=-0.01), dict(projection_error=1.01),
                        dict(token_row_spike=float("inf")), dict(token_row_spike=-1.),
                        dict(rank_one_excess=-0.1), dict(rank_one_excess=1.1),
                        dict(tail_norm_fraction=-0.1), dict(tail_norm_fraction=1.1),
                        dict(contraction_excess=-0.1), dict(contraction_excess=1.1),
                        dict(reciprocity_excess=-0.1), dict(reciprocity_excess=1.1),
                        dict(removal_excess=-0.1), dict(removal_excess=1.1)]:
            with self.subTest(changes=changes):
                self.stats[self.last] = Measurements(1., 1., **changes)
                self.assertIn("unusable", self.flagged()[self.last])
        self.stats[self.last] = Measurements(1., 1., projection_error=0., token_row_spike=0.)
        self.assertEqual(self.flagged(), {})

    def test_excess_tail_adapts_to_a_shared_multidirectional_method(self):
        self.stats = {r.name: Measurements(2., 1., rank_one_excess=0.6) for r in self.rows}
        self.assertEqual(self.flagged(), {})
        self.stats[self.last] = self.stats[self.last]._replace(rank_one_excess=0.8)
        self.assertIn("rank-one", self.flagged()[self.last])

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

    def test_direction_outlier_scales_with_same_kind_dispersion(self):
        # The same held-out writer is unusual in one cohort and ordinary
        # in the other. Attention writers provide a coherent stack in both.
        rows = [row(layer_name(i, kind)) for i in range(16) for kind in ("mlp.down_proj", "self_attn.o_proj")]
        stats = {r.name: Measurements(1., 1.) for r in rows}
        target = layer_name(7)
        for spread, expected in [(0.001, {target}), (0.1, set())]:
            with self.subTest(spread=spread):
                axes = {}
                for r in rows:
                    angle = spread * (-1) ** r.layer if r.kind == "down_proj" else 0.
                    if r.name == target:
                        angle = 0.12
                    axes[r.name] = (-1) ** r.layer * np.array([np.cos(angle), np.sin(angle)])
                ranked = rank.rank_tensors(rows, stats, directions=axes)
                self.assertEqual({r.name for r in ranked if r.tier == "T3"}, expected)
                if expected:
                    finding = next(r for r in ranked if r.name == target)
                    self.assertIn("orthogonal component", finding.reason)
                    self.assertGreater(finding.direction_cosine, 0.9)

    def test_coherent_direction_drift_at_end_of_stack(self):
        # A strict majority shares one direction. The last layers follow a
        # smooth rotation, reaching well past the sparse-cohort cosine cutoff.
        rows = [row(layer_name(2 * i)) for i in range(32)]
        stats = {r.name: Measurements(1., 1.) for r in rows}
        axes = {}
        for i, r in enumerate(rows):
            angle = 0.2 * max(0, i - 25)
            axes[r.name] = (-1) ** i * np.array([np.cos(angle), np.sin(angle)])
        ranked = rank.rank_tensors(rows, stats, directions=axes)
        self.assertLess(next(r.direction_cosine for r in ranked if r.name == rows[-1].name), 0.5)
        self.assertTrue(all(r.tier == "T4" for r in ranked))

    def test_subspace_comparison_allows_drift_and_abstains_without_consensus(self):
        rows = [row(layer_name(i)) for i in range(24)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(25)
        for coherent in [True, False]:
            bases = {}
            for i, r in enumerate(rows):
                angle = max(0, i - 17) * 0.2
                second = np.cos(angle) * eye[1] + np.sin(angle) * eye[2] if coherent else eye[i + 1]
                bases[r.name] = np.column_stack((eye[0], second))
            with self.subTest(coherent=coherent):
                ranked = rank.rank_tensors(rows, stats, subspaces=bases)
                self.assertTrue(all(r.tier == "T4" for r in ranked))

    def test_sparse_direction_cohort_counts_independent_layers(self):
        # Several experts in one layer do not supply additional layer votes.
        rows = [row(layer_name(i)) for i in range(8)]
        rows += [row(layer_name(0, f"mlp.experts.{i}.down_proj")) for i in range(8)]
        stats = {r.name: Measurements(1., 1.) for r in rows}
        axes = {r.name: np.array([1., 0.]) for r in rows}
        axes[layer_name(7)] = np.array([0.8, 0.6])
        ranked = rank.rank_tensors(rows, stats, directions=axes)
        self.assertTrue(all(r.tier == "T4" for r in ranked))
        extra = row(layer_name(8))
        rows.append(extra)
        stats[extra.name] = Measurements(1., 1.)
        axes[extra.name] = np.array([1., 0.])
        ranked = rank.rank_tensors(rows, stats, directions=axes)
        self.assertEqual([r.name for r in ranked if r.tier == "T3"], [layer_name(7)])

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
    def test_symmetric_cross_terms_outside_the_leading_axis_through_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            name = layer_name(0)
            base = np.eye(20)
            base[0, 0], base[-1, -1] = 4., 256.
            edited = base.copy()
            edited[0, 0] = 0.  # exact leading projection
            edited[1, 2] = edited[2, 1] = 0.25
            write_safetensors(tmp / "b.safetensors", {name: ("BF16", base)})
            write_safetensors(tmp / "s.safetensors", {name: ("BF16", edited)})
            rc, out, err = run_cli("--base", str(tmp / "b.safetensors"), "--suspect", str(tmp / "s.safetensors"),
                                   "--out", str(tmp / "out"), "-k", "1")
            self.assertEqual(rc, 0, out + err)
            m = rank.read_delta_csv(tmp / "out" / "delta.csv")[name]
            self.assertEqual((m.projection_error, m.contraction_excess, m.reciprocity_excess), (0., 0., 0.))
            self.assertGreater(m.removal_excess, 0.9)
            with (tmp / "out" / "ranking.csv").open() as f:
                finding, = csv.DictReader(f)
            self.assertEqual(finding["tier"], "T3")
            self.assertIn("positive inner product", finding["reason"])

    def test_rank_three_shared_subspace_through_cli(self):
        # Every change is a genuine removal with three tied modes. The bases
        # rotate within the same space except for one different output axis.
        rng = np.random.default_rng(27)
        for shape in [(8, 12), (12, 8)]:
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                base, suspect = {}, {}
                for i in range(12):
                    left = np.eye(shape[0])[:, [0, 1, 3 if i == 6 else 2]]
                    right = np.linalg.qr(rng.normal(size=(shape[1], 3)))[0]
                    untouched = np.eye(*shape)
                    untouched -= left @ (left.T @ untouched)
                    base[layer_name(i)] = ("BF16", to_bf16(untouched + left @ right.T))
                    suspect[layer_name(i)] = ("BF16", untouched)
                write_safetensors(tmp / "b.safetensors", base)
                write_safetensors(tmp / "s.safetensors", suspect)
                rc, out, err = run_cli("--base", str(tmp / "b.safetensors"), "--suspect", str(tmp / "s.safetensors"),
                                       "--out", str(tmp / "out"), "-k", "1")
                self.assertEqual(rc, 0, out + err)
                with (tmp / "out" / "ranking.csv").open() as f:
                    flagged = {r["name"]: r["reason"] for r in csv.DictReader(f) if r["tier"] != "T4"}
                self.assertEqual(set(flagged), {layer_name(6)})
                self.assertIn("3-dimensional output subspace", flagged[layer_name(6)])

    def test_two_input_reciprocity_through_cli(self):
        # The second coordinate has a large base response. A one-input norm
        # check permits this pair, but swapping the two coordinate probes
        # must give the same cross response for any symmetric removal.
        for shape in [(4, 8), (8, 4)]:
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                name = layer_name(0)
                base = np.zeros(shape, dtype=np.float32)
                base[0, 0], base[1, 1], base[2, 2] = 1., 64., 128.
                suspect = base.copy()
                suspect[0, :2] = [0., 0.5]
                suspect[3, 2] = 0.25
                write_safetensors(tmp / "b.safetensors", {name: ("BF16", base)})
                write_safetensors(tmp / "s.safetensors", {name: ("BF16", suspect)})
                rc, out, err = run_cli("--base", str(tmp / "b.safetensors"), "--suspect", str(tmp / "s.safetensors"),
                                       "--out", str(tmp / "out"), "-k", "1")
                self.assertEqual(rc, 0, out + err)
                m = rank.read_delta_csv(tmp / "out" / "delta.csv")[name]
                self.assertEqual(m.contraction_excess, 0.)
                self.assertEqual(m.rank_one_excess, 0.)
                self.assertGreater(m.reciprocity_excess, 0.)
                with (tmp / "out" / "ranking.csv").open() as f:
                    finding, = csv.DictReader(f)
                self.assertEqual(finding["tier"], "T3")
                self.assertIn("reciprocity", finding["reason"])

    def test_singleton_contraction_with_coherent_tail_through_cli(self):
        # Large base energy in an untouched coordinate makes the global
        # Frobenius rounding bound loose. Along the second input coordinate
        # the base maps to zero, so growth there contradicts any removal.
        for shape in [(4, 8), (8, 4)]:
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                name = layer_name(0)
                w = np.zeros(shape, dtype=np.float32)
                w[0, 0], w[2, 2] = 1., 128.
                edited = w.copy()
                edited[0, :2] = [0., 0.5]
                edited[1, 3] = 0.25
                write_safetensors(tmp / "b.safetensors", {name: ("BF16", w)})
                write_safetensors(tmp / "s.safetensors", {name: ("BF16", edited)})
                rc, out, err = run_cli("--base", str(tmp / "b.safetensors"), "--suspect", str(tmp / "s.safetensors"),
                                       "--out", str(tmp / "out"), "-k", "1")
                self.assertEqual(rc, 0, out + err)
                m = rank.read_delta_csv(tmp / "out" / "delta.csv")[name]
                self.assertEqual(m.rank_one_excess, 0.)
                residual = m.projection_residual()
                self.assertLess(residual.leading, rank.DEFAULT_OUTLIER_FACTOR * residual.tail)
                self.assertGreater(m.contraction_excess, 0.4)
                with (tmp / "out" / "ranking.csv").open() as f:
                    finding, = csv.DictReader(f)
                self.assertEqual(finding["tier"], "T3")
                self.assertIn("contraction", finding["reason"])

    def test_signed_fit_under_checkpoint_rounding(self):
        axis = np.ones(16) / 4
        for dtype, scale in [("BF16", 1.), ("F16", 1.), ("F16", 1e-5)]:
            with self.subTest(dtype=dtype, scale=scale), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                rng = np.random.default_rng(17)
                base, suspect = {}, {}
                for i in range(10):
                    weights = scale * rng.normal(size=(16, 32))
                    w = to_bf16(weights) if dtype == "BF16" else weights.astype(np.float16).astype(np.float32)
                    w[:, -1] = 0.
                    if scale < 1:
                        self.assertLess(np.max(np.abs(w)), np.finfo(np.float16).smallest_normal)
                    strength = 0.125 if scale == 1 else 1.
                    edited = w - strength * np.outer(axis, axis @ w)
                    if i == 9 and dtype == "BF16":
                        # The zero base column is orthogonal to the true right
                        # coefficients. This discrepancy has the same left axis.
                        edited[:, -1] = axis / 16
                    base[layer_name(i)] = (dtype, w)
                    suspect[layer_name(i)] = (dtype, to_bf16(edited) if dtype == "BF16" else edited)
                write_safetensors(tmp / "b.safetensors", base)
                write_safetensors(tmp / "s.safetensors", suspect)
                rc, out, err = run_cli("--base", str(tmp / "b.safetensors"), "--suspect", str(tmp / "s.safetensors"),
                                       "--out", str(tmp / "out"), "-k", "1")
                self.assertEqual(rc, 0, out + err)
                with (tmp / "out" / "ranking.csv").open() as f:
                    flagged = {r["name"]: r["reason"] for r in csv.DictReader(f) if r["tier"] != "T4"}
                if dtype == "BF16":
                    self.assertEqual(set(flagged), {layer_name(9)})
                    self.assertIn("residual norm", flagged[layer_name(9)])
                    stats = rank.read_delta_csv(tmp / "out" / "delta.csv")[layer_name(9)]
                    residual = stats.projection_residual()
                    self.assertLess(residual.leading, rank.DEFAULT_OUTLIER_FACTOR * residual.tail)
                    self.assertEqual(stats.rank_one_excess, 0.)
                else:
                    self.assertEqual(flagged, {})

    def test_top_one_scan_identifies_spread_tail_and_degenerate_singleton(self):
        names = [layer_name(i) for i in range(10)]
        embed = "model.embed_tokens.weight"
        base = {name: ("F32", np.eye(16)) for name in [*names, embed]}
        edited = np.eye(16)
        edited[0, 0] = 0
        suspect = {name: ("F32", edited) for name in names}
        # The same leading removal, with additional modes sharing the tail.
        suspect[names[-1]] = ("F32", edited - np.diag([0.] + [0.25] * 7 + [0.] * 8))
        # Repeated leading singular values and energy on half of the token rows.
        suspect[embed] = ("F32", np.diag([0.] * 8 + [1.] * 8))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_safetensors(tmp / "b.safetensors", base)
            write_safetensors(tmp / "s.safetensors", suspect)
            rc, out, err = run_cli("--base", str(tmp / "b.safetensors"), "--suspect", str(tmp / "s.safetensors"),
                                   "--out", str(tmp / "out"), "-k", "1")
            self.assertEqual(rc, 0, out + err)
            with (tmp / "out" / "delta.csv").open() as f:
                stats = {r["name"]: r for r in csv.DictReader(f)}
            self.assertLess(float(stats[names[-1]]["second_tail_share"]), 0.25)
            self.assertEqual(stats[embed]["projection_gain"], "")
            self.assertAlmostEqual(float(stats[embed]["token_row_fraction"]), 0.5)
            with (tmp / "out" / "ranking.csv").open() as f:
                flagged = {r["name"]: r["reason"] for r in csv.DictReader(f) if r["tier"] == "T3"}
            self.assertEqual(set(flagged), {names[-1], embed})
            self.assertTrue(all("rank-one" in reason for reason in flagged.values()))

    def test_degenerate_rounded_partial_projection_is_consistent(self):
        # The same tiny subtraction in each element rounds differently at 1 and 2.
        w = np.array([[1., 2.], [2., 1.]])
        axis = np.ones(2) / np.sqrt(2.)
        partial = to_bf16(w - 0.002 * np.outer(axis, axis @ w))
        s = core.analyze_delta(w, partial, 1)
        self.assertIsNone(s.direction)
        self.assertAlmostEqual(s.effective_rank, 2.)
        excess = core.rank_one_excess(w, s, "BF16", "BF16")
        self.assertEqual(excess, 0.)
        embed = row("token_embd.weight")
        ranked, = rank.rank_tensors([embed], {embed.name: Measurements(s.effective_rank, s.singular_values[0],
                                                                        rank_one_excess=excess)})
        self.assertEqual(ranked.tier, "T4")

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
