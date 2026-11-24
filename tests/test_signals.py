"""Tests for magnitude, spectral shape, row participation and residual direction."""

import csv
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np

from helpers import gguf_float, write_gguf, write_safetensors
from llmscan import core, rank
from llmscan.interval_fit import _box_excess, fit_excess
from llmscan.names import stack_of
from llmscan.removal import joint_excess, promising_symmetric_response, skew_excess
from llmscan.signals import Measurements, _endpoint_references, _isolated_cliques, _trajectory_groups, subspace_outliers
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

    def test_single_removal_exports_only_the_line_above_the_rounding_budget(self):
        rng = np.random.default_rng(938)
        for shape in [(40, 19), (19, 40)]:
            base = rng.standard_normal(shape)
            axis = np.linalg.qr(rng.normal(size=(len(base), 1)))[0]
            edited = to_bf16(base - axis @ (axis.T @ base))
            stored = to_bf16(base)
            stats = core.analyze_delta(stored, edited, 2, core.rounding_budget(stored, "BF16", "BF16"))
            with self.subTest(shape=shape):
                self.assertGreater(stats.singular_values[1], 0.)
                self.assertIsNotNone(stats.direction)
                np.testing.assert_array_equal(stats.output_subspace, stats.direction[:, None])
                weak = core.analyze_delta(np.zeros(shape), axis @ np.ones((1, shape[1])), 2, 2 * np.sqrt(shape[1]))
                self.assertIsNotNone(weak.direction)
                self.assertIsNone(weak.output_subspace)

    def test_resolved_column_space_can_include_every_input_of_a_tall_matrix(self):
        b = np.eye(9, 3) * [1., .3, .08]
        m = core.analyze_delta(b, np.zeros_like(b), 1, core.rounding_budget(b, "F16", "F16"))
        self.assertEqual(m.output_subspace.shape, (9, 3))
        np.testing.assert_allclose(m.output_subspace @ m.output_subspace.T, np.diag([1.] * 3 + [0.] * 6))

    def test_removal_certificate_allows_independent_rounding_of_both_checkpoints(self):
        rng = np.random.default_rng(412)
        for modes in [1, 4]:
            b = rng.normal(size=(17, 8))
            u = np.linalg.qr(rng.normal(size=(len(b), modes)))[0]
            p = (u * np.linspace(1., 0.25, modes)) @ u.T
            s = b - p @ b
            for dtype, save in [("BF16", to_bf16), ("F16", np.float16), ("F32", np.float32)]:
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

    def test_joint_removal_certificate_detects_small_gain_on_multiple_axes(self):
        rng = np.random.default_rng(42)
        b = rng.normal(size=(48, 96))
        axes = np.linalg.qr(rng.normal(size=(48, 3)))[0]
        s = b + axes @ (axes.T @ b) / 64
        self.assertGreater(core.removal_excess(to_bf16(b), to_bf16(s), "BF16", "BF16"), 0.)

    def test_joint_certificate_allows_coherent_errors_at_opposite_interval_ends(self):
        # Both centers differ from the same identity matrix. Perturbing just
        # one checkpoint cannot remove the positive form on any unit input.
        radius = np.eye(8) / 64
        b, s = np.eye(8) - radius, np.eye(8) + radius
        x = np.ones(8) / np.sqrt(8)
        value = (s @ x) @ ((s - b) @ x)
        self.assertEqual(joint_excess(b, s, radius, radius, x, value), 0.)

    def test_skew_certificate_charges_both_storage_boxes(self):
        rng = np.random.default_rng(521)
        for shape in [(6, 13), (17, 7)]:
            with self.subTest(shape=shape):
                exact = rng.normal(size=shape)
                rb, rs = rng.uniform(0.01, 0.05, (2, *shape))
                b = exact + rb * rng.choice([-1, 1], shape)
                s = exact - rs * rng.choice([-1, 1], shape)
                self.assertGreater(skew_excess(b, s, rb * 0, rs * 0), 0.99)
                self.assertEqual(skew_excess(b, s, rb, rs), 0.)

    def test_skew_and_joint_certificates_keep_the_stronger_margin(self):
        b = np.eye(6)
        s = b * (1 + 0.001)
        for skew, joint in [(0.02, 0.4), (0.4, 0.02)]:
            with self.subTest(skew=skew), patch.object(core, "skew_probe", return_value=(skew, False)), \
                    patch.object(core, "joint_excess", return_value=joint):
                self.assertEqual(core.removal_excess(b, s, "BF16", "BF16"), 0.4)

    def test_skew_can_start_joint_fit_with_no_positive_quadratic_seed(self):
        b = np.array([[1., 1.], [1., -1.]])
        s = b / 2
        s[0, 1] += 3 / 256
        form = s.T @ (s - b)
        self.assertLess(np.linalg.eigvalsh((form + form.T) / 2).max(), 0.)
        # Either checkpoint alone has too little latitude on the skew plane.
        # The two-box fit still has to certify a violation before reporting it.
        with patch.object(core, "fit_excess", wraps=fit_excess) as fit:
            self.assertEqual(core.removal_excess(b, s, "BF16", "BF16"), 0.)
            fit.assert_called_once()

    def test_joint_witness_uses_one_correction_for_symmetry_and_quadratic_response(self):
        # F = Sᵀ(S−B) has one free entry F01 in [-1, 1]. Symmetry
        # requires F01 = 1, where the (1, 1) response becomes positive.
        b, s = np.array([[1.5, 0.], [-1., 1.5]]), np.eye(2)
        rb, rs = np.array([[0., 1.], [0., 0.]]), np.zeros((2, 2))
        factor = np.ones((2, 1)) / np.sqrt(2.)
        skew = np.array([[0., -.5], [.5, 0.]])
        self.assertEqual(_box_excess(b, s, rb, rs, factor, 0 * skew), 0.)
        self.assertEqual(_box_excess(b, s, rb, rs, 0 * factor, skew), 0.)
        self.assertGreater(_box_excess(b, s, rb, rs, factor, skew), 0.)
        self.assertGreater(fit_excess(b, s, rb, rs), 0.)
        rb[1, 0] = 1.  # F01 = F10 = 0 now satisfies both constraints.
        self.assertEqual(fit_excess(b, s, rb, rs), 0.)

    def test_bf16_fit_boxes_match_adjacent_storage_midpoints(self):
        values = (np.arange(1 << 16, dtype=np.uint32) << 16).view(np.float32)
        levels = np.unique(values[np.isfinite(values)]).astype(float)
        # Sweep both signs, zero, the subnormal/normal transition and every
        # interior binade boundary. Repeat after a common rescaling.
        for exponent in (0, 11):
            with self.subTest(exponent=exponent):
                scaled = np.ldexp(levels, -exponent)
                center, radius = core._fit_storage_box(scaled[1:-1], "BF16", exponent, 0.)
                midpoints = (scaled[1:] + scaled[:-1]) / 2
                np.testing.assert_array_equal(center - radius, midpoints[:-1])
                np.testing.assert_array_equal(center + radius, midpoints[1:])

    def test_wide_fit_screen_uses_symmetric_response_and_storage(self):
        b = np.eye(5)
        s = b.copy()
        s[0, 0] += 1 / 16
        s[2, 3] += 0.5
        z = b[0]
        form = s.T @ (s - b)
        self.assertLess(z @ form @ z, np.linalg.norm(form - form.T))
        radius = np.full_like(b, 1 / 1024)
        self.assertTrue(promising_symmetric_response(b, s, radius, radius, z))
        self.assertFalse(promising_symmetric_response(b, s, radius, radius, b[1]))
        self.assertFalse(promising_symmetric_response(b, s, 100 * radius, 100 * radius, z))

        # A positive symmetric seed still qualifies for the legacy fit
        # when the modeled storage scale overestimates its measured noise.
        s[2, 3] = 0.
        self.assertTrue(promising_symmetric_response(b, s, 100 * radius, 100 * radius, z))

    def test_quadratic_box_certificate_is_bounded_by_the_exact_minimum(self):
        b, s = np.array([[0.8]]), np.ones((1, 1))
        rb, rs = np.array([[0.1]]), np.array([[0.08]])
        # Charging the linear and mixed errors separately consumes the gap.
        gap = 0.2
        self.assertGreater(1.2 * rs.item() + rb.item() + (rs * rb).item(), gap)
        excess = _box_excess(b, s, rb, rs, s, np.zeros_like(s))
        self.assertGreater(excess, 0.)
        self.assertLessEqual(excess * gap, 0.92 * (0.92 - 0.9))
        # The enlarged intervals touch at a common exact weight.
        self.assertEqual(_box_excess(b, s, rb, np.array([[0.1]]), s, np.zeros_like(s)), 0.)

    def test_interval_fit_with_coherent_errors_on_rectangular_matrices(self):
        rng = np.random.default_rng(432)
        for shape in [(13, 18), (18, 13)]:
            b = rng.normal(size=shape)
            axes = np.linalg.qr(rng.normal(size=(len(b), 3)))[0]
            radius = np.abs(b) / 256
            error = radius * rng.choice([-1., 1.], size=shape)
            for strengths, gain in [([0.03, 0.02, 0.], True), ([-1., -0.1, -0.1], False)]:
                with self.subTest(shape=shape, strengths=strengths):
                    s = b + (axes * strengths) @ (axes.T @ b)
                    excess = fit_excess(b + error, s - error, radius, radius)
                    self.assertEqual(excess > 0., gain)


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
                ranked = rank.rank_tensors(rows, stats, subspaces=bases, directions={r.name: eye[0] for r in rows})
                self.assertTrue(all(r.tier == "T4" for r in ranked))

    def test_secondary_switch_uses_geometry_under_a_shared_leading_axis(self):
        eye = np.eye(7)
        rows = [row(layer_name(i)) for i in range(17)]
        for minority in ({4, 5, 6, 7}, {1, 4, 10, 15}, {13, 14, 15, 16}):
            for shifted in (False, True):
                with self.subTest(minority=minority, shifted=shifted):
                    axes, spaces, stats = {}, {}, {}
                    for r in rows:
                        member = r.layer in minority
                        axes[r.name] = (-1) ** r.layer * eye[0]
                        # Third mode changes. Rotate the reported basis order
                        # independently of the separately measured leading axis.
                        basis = eye[:, [0, 1, 3 if member else 2]]
                        spaces[r.name] = np.roll(basis, r.layer % 3, axis=1)
                        stats[r.name] = Measurements(1.6 if shifted and member else 1.5, 1.)
                    ranked = rank.rank_tensors(rows, stats, subspaces=spaces, directions=axes)
                    flagged = {r.layer for r in ranked if r.tier == "T3"}
                    self.assertEqual(flagged, minority)
                    self.assertTrue(all("secondary subspace switch" in r.reason for r in ranked if r.tier == "T3"))

    def test_multiple_mode_switches_need_observed_boundaries(self):
        eye = np.eye(6)
        both = {1, 2, 3, 7, 8, 9}
        for missing, shift, expected in [(None, 0., both), (6, 0., {1, 2, 3}), (None, .1, both)]:
            with self.subTest(missing=missing, shift=shift):
                rows = [row(layer_name(i)) for i in range(17) if i != missing]
                stats = {r.name: Measurements(1.5, 1.) for r in rows}
                spaces, axes = {}, {}
                for r in rows:
                    axis = 2 if 7 <= r.layer <= 9 else 1
                    if r.layer in (1, 2, 3):
                        axis = 3
                    theta = (axis - 1) * shift
                    axes[r.name] = np.cos(theta) * eye[0] + np.sin(theta) * eye[4]
                    spaces[r.name] = np.column_stack((axes[r.name], eye[axis]))
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces, directions=axes)
                self.assertEqual({r.layer for r in ranked if r.tier != "T4"}, expected)

    def test_secondary_switches_at_different_angular_scales_and_both_ends(self):
        rows = [row(layer_name(i)) for i in range(30)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        eye = np.eye(4)
        # The first two minority concepts border each other and the second
        # recurs. Their separation is smaller than the usual coherence radius.
        groups = [{0, 1, 2}, {3, 4, 5, 16, 17, 18}, {27, 28, 29}]
        axes, spaces = {}, {}
        for r in rows:
            angle = next((a for a, g in zip((8., 18., 65.), groups) if r.layer in g), 0.)
            theta = np.radians(angle + .1 * (r.layer % 3))
            secondary = np.cos(theta) * eye[1] + np.sin(theta) * eye[2]
            axes[r.name] = (-1) ** r.layer * eye[0]
            spaces[r.name] = np.column_stack((eye[0], secondary))
        ranked = rank.rank_tensors(rows[::-1], stats, subspaces=spaces, directions=axes)
        self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, set.union(*groups))

    def test_trajectory_partition_checks_drift_in_layer_order(self):
        # The same unordered angles describe either a slow rotation or
        # repeated abrupt jumps. Only the first provides a reference path.
        ids = list(range(24))
        smooth = list(range(0, 60, 3))
        shuffled = [value for pair in zip(smooth[:10], smooth[10:]) for value in pair]
        for cover in [smooth, shuffled]:
            with self.subTest(cover=cover):
                theta = np.radians(cover[:10] + [90] * 4 + cover[10:])
                angles = np.abs(theta[:, None] - theta)
                result = next(_trajectory_groups(ids, angles, len(ids), 3., 1e-6), None)
                if cover == smooth:
                    labels, step = result
                    self.assertEqual(set(np.flatnonzero(labels != labels[0])), set(range(10, 14)))
                    self.assertAlmostEqual(np.degrees(step), 3.)
                else:
                    self.assertIsNone(result)

    def test_secondary_switches_against_a_smooth_majority_trajectory(self):
        rows = [row(layer_name(i)) for i in range(32)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        eye = np.eye(6)
        runs = [range(4, 8), range(18, 21)]
        for switching, primary_jump in [(False, False), (True, False), (True, True)]:
            with self.subTest(switching=switching, primary_jump=primary_jump):
                spaces, axes = {}, {}
                for r in rows:
                    theta = np.radians(2.6 * r.layer)
                    secondary = np.cos(theta) * eye[1] + np.sin(theta) * eye[2]
                    primary = eye[0]
                    for k, run in enumerate(runs):
                        if switching and r.layer in run:
                            turn = np.radians(60. + k * 30.)
                            secondary = np.cos(turn) * secondary + np.sin(turn) * eye[3 + k]
                            if primary_jump:
                                primary = np.cos(.12) * eye[0] + np.sin(.12) * eye[5]
                    axes[r.name] = (-1) ** r.layer * primary
                    spaces[r.name] = np.column_stack((primary, secondary))
                ranked = rank.rank_tensors(rows[::-1], stats, directions=axes, subspaces=spaces)
                expected = set().union(*runs) if switching and not primary_jump else set()
                self.assertEqual({r.layer for r in ranked if r.tier != "T4"}, expected)
                self.assertTrue(all("step per layer" in r.reason for r in ranked if r.tier == "T3"))

    def test_switches_with_an_exceptional_step_in_the_majority_trajectory(self):
        rows = [row(layer_name(i)) for i in range(24)]
        eye = np.eye(4)
        ordinary = list(range(0, 60, 3))
        ordinary[3], ordinary[7] = ordinary[7], ordinary[3]
        angles = ordinary[:10] + [90] * 4 + ordinary[10:]
        for shared, primary in [(False, False), (True, False), (True, True)]:
            with self.subTest(shared=shared, primary=primary):
                spaces, directions = {}, {}
                for r, angle in zip(rows, np.radians(angles)):
                    basis = np.cos(angle) * eye[:, :2] + np.sin(angle) * eye[:, 2:]
                    if shared:
                        basis[:, 0] = eye[0]
                    spaces[r.name] = basis
                    if primary:
                        directions[r.name] = eye[0]
                findings = subspace_outliers(rows, spaces, 3., directions=directions)
                for i in range(10, 14):
                    self.assertIn("step per layer", findings[layer_name(i)])

    def test_isolated_run_compares_both_boundary_steps_with_other_layers(self):
        # An ordinary 40-degree step enters a pause. An exceptional exit
        # alone does not make the pause an anomalous regime.
        n = 23
        ids = list(range(n))
        run = np.arange(10, 14)
        for entry, expected in [(40., False), (70., True)]:
            with self.subTest(entry=entry):
                steps = np.full(n - 1, 40.)
                steps[run[:-1]] = 0.
                steps[run[0] - 1], steps[run[-1]] = entry, 68.
                # Each step takes a fresh orthogonal direction. Distant
                # layers don't revisit an earlier point on the walk.
                axes = np.eye(n)
                for i, angle in enumerate(np.radians(steps), 1):
                    axes[i] = np.cos(angle) * axes[i-1] + np.sin(angle) * axes[i]
                angles = np.arccos(np.clip(np.abs(axes @ axes.T), 0., 1.))
                result = _isolated_cliques(ids, angles, n, 3., 1e-6, (0, n - 1))
                if expected:
                    self.assertEqual(set(np.flatnonzero(result[0])), set(run))
                else:
                    self.assertIsNone(result)

    def test_no_isolated_run_among_unrelated_directions_with_incidental_pairs(self):
        rng = np.random.default_rng(204)
        axes = rng.normal(size=(28, 48))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        # There are several copies, but no three consecutive observations.
        axes[[3, 11, 12]] = axes[2]
        angles = np.arccos(np.clip(np.abs(axes @ axes.T), 0., 1.))
        ids = list(range(len(axes)))
        self.assertIsNone(_isolated_cliques(ids, angles, len(ids), 3., 1e-6, (ids[0], ids[-1])))

    def test_clique_without_measured_boundaries_preserves_other_groups_reference(self):
        n = 33
        eye = np.eye(n + 4)
        walk = eye[1:n + 1].copy()
        angle = np.radians(40.)
        for i in range(1, n):
            walk[i] = np.cos(angle) * walk[i - 1] + np.sin(angle) * walk[i]
        rows = [row(layer_name(i)) for i in range(n)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        first = set(range(1, 9))
        second = set(range(17, 25))
        # If both cliques qualify, sixteen of the thirty-two measured planes
        # belong to them. A missing boundary cannot turn a member into a peer.
        split = set(range(1, 5)) | set(range(11, 15))
        for line, group, expected in [(0, first, second), (9, first, second), (0, split, set())]:
            with self.subTest(line=line, group=group):
                spaces = {}
                for r in rows:
                    i = r.layer
                    secondary = walk[i]
                    if i in group:
                        secondary = eye[-1]
                    if i in second:
                        secondary = eye[-2]
                    spaces[r.name] = eye[:, :1] if i == line else np.column_stack((eye[0], secondary))
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, expected)

    def test_measured_end_of_incomplete_run_can_contradict_a_recurring_switch(self):
        n = 30
        eye = np.eye(n + 3)
        walk = eye[1:n + 1].copy()
        step = np.radians(40.)
        for i in range(1, n):
            walk[i] = np.cos(step) * walk[i - 1] + np.sin(step) * walk[i]
        rows = [row(layer_name(i)) for i in range(n)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        complete = set(range(16, 20))
        for degrees, expected in [(39, set()), (80, complete)]:
            with self.subTest(degrees=degrees):
                spaces = {}
                for r in rows:
                    i = r.layer
                    secondary = eye[-1] if 1 <= i <= 4 or i in complete else walk[i]
                    if i == 5:
                        angle = np.radians(degrees)
                        secondary = np.cos(angle) * eye[-1] + np.sin(angle) * walk[i]
                    spaces[r.name] = eye[:, :1] if i == 0 else np.column_stack((eye[0], secondary))
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, expected)

    def test_isolated_secondary_runs_under_a_shared_leading_axis(self):
        n = 30
        eye = np.eye(n + 4)
        walk = eye[1:n+1].copy()
        for i in range(1, n):
            angle = np.radians(39 + i % 5)
            walk[i] = np.cos(angle) * walk[i-1] + np.sin(angle) * walk[i]
        runs = (set(range(3)), set(range(16, 21)))
        for switching, primary_jump, missing, expected in [
            (False, False, None, set()), (True, False, None, set.union(*runs)),
            (True, True, None, set()), (True, False, 15, set()),
            (True, False, 3, runs[1]),
        ]:
            with self.subTest(switching=switching, primary_jump=primary_jump, missing=missing):
                rows = [row(layer_name(i)) for i in range(n) if i != missing]
                axes, spaces = {}, {}
                for r in rows:
                    primary, secondary = eye[0], walk[r.layer]
                    for k, run in enumerate(runs):
                        if switching and r.layer in run:
                            angle = np.radians(r.layer % 3)
                            secondary = np.cos(angle) * eye[-k-1] + np.sin(angle) * secondary
                            if primary_jump:
                                primary = np.cos(.13) * eye[0] + np.sin(.13) * eye[-3]
                    axes[r.name] = (-1) ** r.layer * primary
                    spaces[r.name] = np.column_stack((primary, secondary))
                stats = {r.name: Measurements(1.5, 1.) for r in rows}
                ranked = rank.rank_tensors(rows[::-1], stats, directions=axes, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier != "T4"}, expected)
                self.assertTrue(all("within-clique angle" in r.reason for r in ranked if r.tier == "T3"))

    def test_recurring_leading_change_leaves_separate_secondary_run_visible(self):
        n = 36
        eye = np.eye(n + 4)
        rows = [row(layer_name(i)) for i in range(n)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        run = set(range(12, 16))
        recurring = {4, 21, 32}
        axes, spaces = {}, {}
        walk = eye[1]
        for r in rows:
            axis = eye[0]
            if r.layer in recurring:
                axis = np.cos(.1) * axis + np.sin(.1) * eye[-1]
                secondary = eye[-2]
            elif r.layer in run:
                secondary = eye[-3]
            else:
                if r.layer:
                    step = np.radians(38 + r.layer % 4)
                    walk = np.cos(step) * walk + np.sin(step) * eye[1 + r.layer]
                secondary = walk
            axes[r.name] = axis
            spaces[r.name] = np.column_stack((axis, secondary))
        findings = subspace_outliers(rows, spaces, 3., stats, axes)
        self.assertEqual(set(findings), {layer_name(i) for i in run})

    def test_mixed_rank_compares_only_paired_modes_in_either_rank_order(self):
        rows = [row(layer_name(i)) for i in range(19)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        run = set(range(6, 11))
        eye = np.eye(8)
        rng = np.random.default_rng(416)
        for (small, large), degrees, reverse in product([(1, 2), (2, 3), (2, 4)], [0, 35, 90], [False, True]):
            with self.subTest(ranks=(small, large), degrees=degrees, reverse=reverse):
                theta = np.radians(degrees)
                plane = eye[:, :large]
                narrow = np.cos(theta) * plane[:, :small] + np.sin(theta) * eye[:, large:large+small]
                spaces = {}
                for r in rows:
                    q = narrow if (r.layer in run) != reverse else plane
                    rotation, _ = np.linalg.qr(rng.normal(size=(q.shape[1], q.shape[1])))
                    spaces[r.name] = q @ rotation
                ranked = rank.rank_tensors(rows[::-1], stats, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, run if degrees else set())
                self.assertTrue(all("whole output subspace" in r.reason for r in ranked if r.tier == "T3"))

    def test_contained_rank_changes_can_change_which_axis_leads(self):
        rows = [row(layer_name(i)) for i in range(18)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        eye = np.eye(8)
        for run, reverse, sparse in product([{7}, {6, 7, 8, 9}], [False, True], [False, True]):
            with self.subTest(run=run, reverse=reverse, sparse=sparse):
                spaces = {r.name: eye[:, 1:2] if (r.layer in run) != reverse else eye[:, :2] for r in rows}
                axes = {r.name: spaces[r.name][:, 0] for r in rows if not sparse or r.layer in run or r.layer < 4}
                ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                self.assertTrue(all(r.tier == "T4" for r in ranked))

    def test_expansion_uses_edit_strength_in_the_unpaired_directions(self):
        rows = [row(layer_name(i)) for i in range(17)]
        run = set(range(7, 11))
        stats = {r.name: Measurements(2., 1.) for r in rows}
        rng = np.random.default_rng(128)
        axes, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        for (p, q), extra, subset, tall in product([(1, 2), (2, 3), (2, 4)], [1.1, 4., 12.],
                                                   [False, True], [False, True]):
            with self.subTest(ranks=(p, q), extra=extra, subset=subset, tall=tall):
                # Keep only one singular value for CSV output. The projection
                # still needs every resolved mode in the original output space.
                spaces, spectra = {}, {}
                right = np.eye(6 if tall else 9)
                for r in rows:
                    wide = (r.layer in run) != subset
                    strengths = [8.] * p + ([extra] * (q-p) if wide else [])
                    d = len(strengths)
                    edit = (axes[:, :d] * strengths) @ right[:d]
                    m = core.analyze_delta(np.zeros_like(edit), edit, 1, 1.)
                    self.assertEqual(m.output_subspace.shape, (7, d))
                    spaces[r.name], spectra[r.name] = m.output_subspace, m.output_spectrum
                findings = subspace_outliers(rows, spaces, 3., stats, spectra=spectra)
                expected = run if not subset and extra > 2 else set()
                self.assertEqual(set(findings), {layer_name(i) for i in expected})
                self.assertTrue(all("subspace expansion" in reason for reason in findings.values()))

    def test_expansion_requires_an_abrupt_measured_boundary(self):
        rows = [row(layer_name(i)) for i in range(17)]
        run = set(range(4, 10))
        for gradual, missing, omitted in product([False, True], [False, True], [0., 3.]):
            with self.subTest(gradual=gradual, missing=missing, omitted=omitted):
                spaces, spectra = {}, {}
                for r in rows:
                    if missing and r.layer == 3:
                        continue
                    if r.layer in run:
                        strength = 4. + (r.layer - min(run)) * 2. if gradual else 4.
                        weights = [8., strength]
                    else:
                        weights = [8.]
                    spaces[r.name] = np.eye(5)[:, :len(weights)]
                    spectra[r.name] = core.OutputSpectrum(np.array(weights), omitted, 1., 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                expected = run if not (gradual or missing or omitted) else set()
                self.assertEqual(set(findings), {layer_name(i) for i in expected})

    def test_expansions_allow_multiple_directions_and_short_runs(self):
        rows = [row(layer_name(i)) for i in range(22)]
        eye = np.eye(10)
        layouts = [range(7, 13), (0, 3, 6, 9, 12, 21), (2, 3, 9, 10, 16, 17)]
        for layout, directions in product(layouts, [2, 6]):
            with self.subTest(layout=layout, directions=directions):
                spaces, spectra = {}, {}
                for r in rows:
                    axes, strengths = [0, 1], [8., 7.]
                    if r.layer in layout:
                        axes.append(2 + layout.index(r.layer) * directions // len(layout))
                        strengths.append(5.)
                    spaces[r.name] = eye[:, axes]
                    spectra[r.name] = core.OutputSpectrum(np.array(strengths), .1, 1., 1.)
                findings = subspace_outliers(rows[::-1], spaces, 3., spectra=spectra)
                self.assertEqual(set(findings), {layer_name(i) for i in layout})

    def test_expansion_relative_to_a_drifting_majority_space(self):
        rows = [row(layer_name(i)) for i in range(28)]
        eye = np.eye(12)
        target = range(11, 17)
        for rank, rate, primary, stepped in product([1, 2, 3], [1., 3.], [10., 150.], [False, True]):
            with self.subTest(rank=rank, rate=rate, primary=primary, stepped=stepped):
                spaces, spectra = {}, {}
                for r in rows:
                    angle = np.radians(rate * r.layer)
                    line = np.cos(angle) * eye[0] + np.sin(angle) * eye[3]
                    axes, values = [line, *eye[1:rank]], [primary, *([10.] * (rank-1))]
                    if r.layer in target:
                        axes.append(eye[4 + target.index(r.layer)])
                        values.append(2.7 if stepped and r.layer == target.start else 7.)
                    # Neither basis order nor singular-vector sign defines a trajectory.
                    spaces[r.name] = (-1)**r.layer * np.column_stack(axes[::-1])
                    noise = 2.5 if r.layer == 2 else 1.
                    spectra[r.name] = core.OutputSpectrum(np.array(values[::-1]), 0., noise, 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                self.assertEqual(set(findings), {layer_name(i) for i in target})
                self.assertTrue(all("local majority trajectory" in why for why in findings.values()))

    def test_expansion_past_either_end_of_a_drifting_majority(self):
        rows = [row(layer_name(i)) for i in range(29)]
        eye = np.eye(12)
        for reverse, varying, stepped in product([False, True], repeat=3):
            with self.subTest(reverse=reverse, varying=varying, stepped=stepped):
                spaces, spectra, expected = {}, {}, set()
                for r in rows:
                    depth = 28 - r.layer if reverse else r.layer
                    angle = np.radians(2.3 * depth)
                    axis = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
                    axes, values = [axis, eye[2]], [12., 9.]
                    if depth >= 22:
                        axes.append(eye[3 + (depth-22 if varying else 0)])
                        values.append(3. if stepped and depth == 22 else 8.)
                        expected.add(r.name)
                    spaces[r.name] = (-1)**depth * np.column_stack(axes[::-1])
                    noise = 3. if depth == 4 else 1.
                    spectra[r.name] = core.OutputSpectrum(np.array(values[::-1]), 0., noise, 1.)
                found = subspace_outliers(rows, spaces, 3., spectra=spectra)
                self.assertEqual(set(found), expected)
                self.assertTrue(all("local majority trajectory" in reason for reason in found.values()))

    def test_spaced_endpoint_anchors_use_equally_spaced_majority_observations(self):
        eye = np.eye(5)
        rows = [row(layer_name(i)) for i in range(27)]
        for reverse, drops in product([False, True], [{19}, {17, 19}]):
            with self.subTest(reverse=reverse, drops=drops):
                spaces, spectra, expected = {}, {}, set()
                for r in rows:
                    depth = 26 - r.layer if reverse else r.layer
                    angle = np.radians(2. * depth)
                    axes = [np.cos(angle) * eye[0] + np.sin(angle) * eye[1]]
                    values = [12.]
                    if depth not in drops:
                        axes.append(eye[2])
                        values.append(9.)
                    if depth > 20:
                        axes.append(eye[3])
                        values.append(8.)
                        expected.add(r.name)
                    spaces[r.name] = np.column_stack(axes)
                    spectra[r.name] = core.OutputSpectrum(np.array(values), 0., 1., 1.)
                self.assertEqual(set(subspace_outliers(rows, spaces, 3., spectra=spectra)), expected)

    @staticmethod
    def wandering_axes(eye, count, high_dimensional):
        if not high_dimensional:
            angles = np.radians([8. * abs(i % 10 - 5) for i in range(count)])
            return [np.cos(a) * eye[0] + np.sin(a) * eye[1] for a in angles]
        # Step into a fresh coordinate on the sphere, occasionally retracing
        # the last step. No fixed small ambient plane reproduces these axes.
        step = np.radians(7.)
        axes = [eye[0]]
        for i in range(1, count):
            axes.append(axes[-2] if i % 5 == 0 else np.cos(step) * axes[-1] + np.sin(step) * eye[i])
        return axes

    def test_expansions_along_turning_and_high_dimensional_paths(self):
        rows = [row(layer_name(i)) for i in range(31)]
        eye = np.eye(42)
        for wandering, start, stepped in product([False, True], [0, 12, 25], [False, True]):
            with self.subTest(wandering=wandering, start=start, stepped=stepped):
                spaces, spectra = {}, {}
                run = range(start, start + 6)
                axes = self.wandering_axes(eye, len(rows), wandering)
                for r, axis in zip(rows, axes):
                    directions = [eye[31], axis]
                    values = [9., 12. if wandering else 100.]
                    if r.layer in run:
                        # Two simultaneous extra axes, one changing every layer.
                        directions.extend([eye[32], eye[33 + run.index(r.layer)]])
                        shoulder = run[-1] if start == 0 else run[0]
                        values.extend([4. if stepped and r.layer == shoulder else 11., 3.])
                    spaces[r.name] = np.column_stack(directions[::-1])
                    noise = 4. if r.layer == 7 else 1.
                    spectra[r.name] = core.OutputSpectrum(np.array(values[::-1]), 0., noise, 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                self.assertEqual(set(findings), {layer_name(i) for i in run})
                method = "tracked majority space" if wandering else "majority ambient span"
                self.assertTrue(all(method in why for why in findings.values()), findings)

    def test_honest_wandering_with_constant_and_gradually_resolving_rank(self):
        rows = [row(layer_name(i)) for i in range(31)]
        eye = np.eye(33)
        for wandering, ramp, reverse in product([False, True], repeat=3):
            with self.subTest(wandering=wandering, ramp=ramp, reverse=reverse):
                spaces, spectra = {}, {}
                axes = self.wandering_axes(eye, len(rows), wandering)
                for r, axis in zip(rows, axes):
                    depth = 30 - r.layer if reverse else r.layer
                    extra = 2. * ((depth + 1) / 18.)**4 if ramp else 0.
                    budget = 3.5
                    edit = 32. * np.outer(axis, axis) + extra * np.outer(eye[-1], eye[-1])
                    m = core.analyze_delta(0. * eye, edit, 1, budget)
                    spaces[r.name], spectra[r.name] = m.output_subspace, m.output_spectrum
                self.assertEqual({s.shape[1] for s in spaces.values()}, {1, 2} if ramp else {1})
                self.assertEqual(subspace_outliers(rows, spaces, 3., spectra=spectra), {})

    def test_only_the_extra_modes_component_outside_the_envelope_counts(self):
        eye = np.eye(5)
        extra = .8 * eye[2] + .6 * eye[3]
        basis = np.column_stack((eye[0], eye[1], extra))
        spectrum = core.OutputSpectrum(np.array([20., 10., 5.]), 0., 1., 1.)
        self.assertAlmostEqual(spectrum.outside_bounds(eye[:, :3], basis, 2)[2], 3.)
        self.assertAlmostEqual(spectrum.outside_bounds(eye[:, :4], basis, 2)[2], 0.)

    def test_endpoint_forecast_error_at_the_same_physical_horizon(self):
        ids = [*range(11), 11, 14, 22]
        eye = np.eye(3)
        bases = []
        for depth in ids:
            angle = np.radians(1.8 * depth + .05 * depth**2)
            axis = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
            bases.append(axis[:, None] if depth <= 10 else np.column_stack((axis, eye[2])))
        refs = dict(_endpoint_references(ids, bases, np.arange(11), 1e-7, np.arccos(.9)))
        self.assertEqual({ids[i] for i in refs}, {11, 14})
        for i, ref in refs.items():
            h = ids[i] - 10
            # Continuing the last step of a quadratic misses the true angle
            # by the same amount after every held-out endpoint at horizon h.
            error = np.radians(.05 * h * (h+1))
            self.assertAlmostEqual(ref.uncertainty, np.tan(error))

    def test_curving_trajectory_with_a_terminal_change_in_rank_resolution(self):
        eye = np.eye(4)
        rows = [row(layer_name(i)) for i in range(30)]
        for reverse, ramp in product([False, True], repeat=2):
            with self.subTest(reverse=reverse, ramp=ramp):
                spaces, spectra = {}, {}
                for r in rows:
                    depth = 29 - r.layer if reverse else r.layer
                    angle = np.radians(1.4 * depth + .045 * depth**2)
                    axis = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
                    secondary = -np.sin(angle) * eye[0] + np.cos(angle) * eye[1]
                    strength = .1 * (depth + 1) if ramp else 2.
                    noise = 2.2 if ramp or depth < 22 else .5
                    edit = 40. * np.outer(axis, axis) + strength * np.outer(secondary, secondary)
                    m = core.analyze_delta(0. * eye, edit, 1, noise)
                    spaces[r.name], spectra[r.name] = m.output_subspace, m.output_spectrum
                self.assertEqual({q.shape[1] for q in spaces.values()}, {1, 2})
                self.assertEqual(subspace_outliers(rows, spaces, 3., spectra=spectra), {})

    def test_drifting_ramp_and_changes_of_rank_resolution_stay_clear(self):
        rows = [row(layer_name(i)) for i in range(32)]
        eye = np.eye(5)
        for ramp, reverse in product([False, True], [False, True]):
            with self.subTest(ramp=ramp, reverse=reverse):
                spaces, spectra = {}, {}
                for r in rows:
                    depth = 31 - r.layer if reverse else r.layer
                    angle = np.radians(2.5 * depth)
                    line = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
                    strength = 0.15 * (depth + 1) if ramp else 4.
                    noise = 5. if depth < 12 or depth >= 22 else .3
                    edit = 12 * np.outer(line, line) + strength * np.outer(eye[2], eye[2])
                    m = core.analyze_delta(0. * eye, edit, 1, noise)
                    spaces[r.name], spectra[r.name] = m.output_subspace, m.output_spectrum
                self.assertEqual(subspace_outliers(rows, spaces, 3., spectra=spectra), {})

    def test_drifting_ramp_reaches_resolved_strength_without_an_abrupt_boundary(self):
        eye = np.eye(5)
        rows = [row(layer_name(i)) for i in range(32)]
        for abrupt in [False, True]:
            with self.subTest(abrupt=abrupt):
                spaces, spectra = {}, {}
                for r in rows:
                    theta = np.radians(2.5 * r.layer)
                    axis = np.cos(theta) * eye[0] + np.sin(theta) * eye[1]
                    depth = 1 + min(r.layer, 31 - r.layer)
                    if abrupt and depth >= 10:
                        depth = 16
                    strength = 2. * (depth / 10)**4
                    edit = 20. * np.outer(axis, axis) + strength * np.outer(eye[3], eye[3])
                    m = core.analyze_delta(0. * eye, edit, 1, 1.9)
                    spaces[r.name], spectra[r.name] = m.output_subspace, m.output_spectrum
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                self.assertEqual(set(findings), {layer_name(i) for i in range(9, 23)} if abrupt else set())

    def test_observed_acceleration_away_from_an_expansion_preserves_its_reference(self):
        eye = np.eye(4)
        rows = [row(layer_name(i)) for i in range(36)]
        for reverse in [False, True]:
            with self.subTest(reverse=reverse):
                spaces, spectra, expected = {}, {}, set()
                for r in rows:
                    depth = 35 - r.layer if reverse else r.layer
                    angle = np.radians(depth + 0.25 * max(0, depth - 24)**2)
                    axis = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
                    expanded = 9 <= depth < 14
                    spaces[r.name] = np.column_stack((axis, eye[2])) if expanded else axis[:, None]
                    values = [10., 5.] if expanded else [10.]
                    spectra[r.name] = core.OutputSpectrum(np.array(values), 0., 1., 1.)
                    if expanded:
                        expected.add(r.name)
                self.assertEqual(set(subspace_outliers(rows, spaces, 3., spectra=spectra)), expected)

    def test_an_abrupt_change_between_majority_regimes_is_not_a_drifting_reference(self):
        eye = np.eye(5)
        rows = [row(layer_name(i)) for i in range(30)]
        spaces, spectra = {}, {}
        for r in rows:
            angle = np.radians(r.layer + (40 if r.layer >= 12 else 0))
            axis = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
            expanded = 8 <= r.layer < 14
            spaces[r.name] = np.column_stack((axis, eye[2])) if expanded else axis[:, None]
            values = [8., 6.] if expanded else [8.]
            spectra[r.name] = core.OutputSpectrum(np.array(values), 0., 1., 1.)
        findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
        self.assertFalse(any("subspace expansion" in why for why in findings.values()))

    def test_short_expansion_can_have_one_resolved_internal_change_of_level(self):
        rows = [row(layer_name(i)) for i in range(24)]
        for strengths, expected in [([4., 8., 8., 8.], True), ([8., 8., 4., 4., 4., 4.], True),
                                    ([2.4, 3.6, 3.5, 3.55, 3.45, 3.4], True),
                                    ([4., 6., 8., 10., 12., 14.], False)]:
            for reverse in [False, True]:
                with self.subTest(strengths=strengths, reverse=reverse):
                    spaces, spectra = {}, {}
                    target = range(5, 5 + len(strengths))
                    levels = strengths[::-1] if reverse else strengths
                    for r in rows:
                        values = [20., levels[target.index(r.layer)]] if r.layer in target else [20.]
                        spaces[r.name] = np.eye(4)[:, :len(values)]
                        spectra[r.name] = core.OutputSpectrum(np.array(values), 0., 1., 1.)
                    findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                    self.assertEqual(set(findings), {layer_name(i) for i in target} if expected else set())

    def test_strength_trajectory_rejects_a_ramp_even_as_extra_axes_rotate(self):
        rows = [row(layer_name(i)) for i in range(19)]
        for strengths, expected in [([1.1] * 5, []), ([1.1, 3., 5., 3., 1.1], []),
                                    ([1.1, 7., 7., 7., 1.1], [7, 8, 9])]:
            with self.subTest(strengths=strengths):
                spaces, spectra = {}, {}
                for r in rows:
                    axes, weights = [0], [8.]
                    if 6 <= r.layer <= 10:
                        axes.append(r.layer - 5)
                        weights.append(strengths[r.layer - 6])
                    spaces[r.name] = np.eye(7)[:, axes]
                    spectra[r.name] = core.OutputSpectrum(np.array(weights), 0., 1., 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                self.assertEqual(set(findings), {layer_name(i) for i in expected})

    def test_tilt_of_a_strong_paired_axis_cannot_certify_a_weak_extra_mode(self):
        eye = np.eye(5)
        tilt = np.cos(.2) * eye[0] + np.sin(.2) * eye[1]
        basis = np.column_stack((tilt, eye[2]))
        spectrum = core.OutputSpectrum(np.array([100., 1.1]), 0., 1., 1.)
        lower, upper, unpaired = spectrum.outside_bounds(eye[:, :1], basis)
        self.assertAlmostEqual(lower, 100. * np.sin(.2))
        self.assertAlmostEqual(upper, lower)
        self.assertAlmostEqual(unpaired, 1.1)

    def test_expansion_boundary_and_ordinary_steps_use_the_same_complement(self):
        rows = [row(layer_name(i)) for i in range(19)]
        eye = np.eye(8)
        run = set(range(6, 11))
        for tilt_strength in (5., 12.):
            with self.subTest(tilt_strength=tilt_strength):
                spaces, spectra = {}, {}
                for r in rows:
                    basis, weights = eye[:, :1], [100.]
                    if r.layer in run:
                        sine = (tilt_strength + 2 * (r.layer % 2)) / 100.
                        tilt = np.sqrt(1. - sine**2) * eye[0] + sine * eye[1]
                        basis, weights = np.column_stack((tilt, eye[r.layer - 4])), [100., 4.]
                    spaces[r.name] = basis
                    spectra[r.name] = core.OutputSpectrum(np.array(weights), 0., 1., 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                expanded = {name for name, reason in findings.items() if "subspace expansion" in reason}
                expected = {layer_name(i) for i in run} if tilt_strength > 6 else set()
                self.assertEqual(expanded, expected)

    def test_stepped_expansion_attributes_only_independently_elevated_layers(self):
        rows = [row(layer_name(i)) for i in range(26)]
        eye = np.eye(4)
        for (shoulders, attributed), reverse in product([([2.4], [2.4]), ([2.4, 2.8], [2.4, 2.8]),
                                                        ([1.1, 2.4], []), ([1.1], [])], [False, True]):
            with self.subTest(shoulders=shoulders, reverse=reverse):
                values = [0.] * 7 + shoulders + [6.] * 4
                values += [0.] * (len(rows) - len(values))
                if reverse:
                    values.reverse()
                spaces, spectra = {}, {}
                for r, value in zip(rows, values):
                    # Less precise reference intervals overlap both
                    # intermediate levels. The majority still separates.
                    noise = 2. if r.layer in (0, 25) else 1.
                    strengths = [9., value] if value else [9.]
                    spaces[r.name] = eye[:, :len(strengths)]
                    spectra[r.name] = core.OutputSpectrum(np.array(strengths), 0., noise, 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                # 2.4 clears the majority, but its interval overlaps a 1.1
                # neighbor. Only a measured outer boundary supports attribution.
                expected = {r.name for r, value in zip(rows, values) if value in [6., *attributed]}
                self.assertEqual(set(findings), expected)

    def test_flat_core_cannot_hide_motion_among_transition_layers(self):
        rows = [row(layer_name(i)) for i in range(36)]
        for first, reverse in product([2.5, 4.], [False, True]):
            with self.subTest(first=first, reverse=reverse):
                values = [0.] * 10 + [first, first + 1.] + [14.] * 10 + [0.] * 14
                if reverse:
                    values.reverse()
                spaces, spectra = {}, {}
                for r, extra in zip(rows, values):
                    strengths = [20., extra] if extra else [20.]
                    spaces[r.name] = np.eye(4)[:, :len(strengths)]
                    noise = 5. if r.layer in (0, 35) else 1.
                    spectra[r.name] = core.OutputSpectrum(np.array(strengths), 0., noise, 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                expected = {r.name for r, v in zip(rows, values) if v == 14. or (v and first > 3.)}
                self.assertEqual(set(findings), expected)

    def test_expansion_step_reference_tolerates_isolated_but_not_typical_motion(self):
        rows = [row(layer_name(i)) for i in range(26)]
        eye = np.eye(4)
        for varying in [{3}, set(range(0, 26, 2))]:
            with self.subTest(varying=varying):
                spaces, spectra = {}, {}
                target = set(range(9, 14))
                for r in rows:
                    if r.layer in target:
                        basis, values = eye[:, [0, 2]], [20., 5.5]
                    else:
                        angle = .12 if r.layer in varying else 0.
                        axis = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
                        basis, values = axis[:, None], [20.]
                    spaces[r.name] = basis
                    spectra[r.name] = core.OutputSpectrum(np.array(values), 0., .5, 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                expanded = {name for name, reason in findings.items() if "subspace expansion" in reason}
                expected = target if len(varying) == 1 else set()
                self.assertEqual(expanded, {layer_name(i) for i in expected})

    def test_continuous_strength_ramp_across_rank_and_rounding_thresholds(self):
        rows = [row(layer_name(i)) for i in range(32)]
        eye = np.eye(5)
        for reverse in [False, True]:
            with self.subTest(reverse=reverse):
                spaces, spectra = {}, {}
                for r in rows:
                    depth = len(rows) - r.layer - 1 if reverse else r.layer
                    strength = (depth + 1) / 8
                    noise = 3. if depth < 20 else .5
                    edit = np.diag([10., strength, 0., 0., 0.])
                    measured = core.analyze_delta(0. * eye, edit, 1, noise)
                    spaces[r.name], spectra[r.name] = measured.output_subspace, measured.output_spectrum
                self.assertEqual(subspace_outliers(rows, spaces, 3., spectra=spectra), {})

    def test_expansion_compares_absolute_strength_when_layer_budgets_differ(self):
        rows = [row(layer_name(i)) for i in range(17)]
        for scale, extra, run in product([1e-220, 1., 1e220], [4., 30.], [set(range(6, 11)), {4, 9, 10}]):
            with self.subTest(scale=scale, extra=extra, run=run):
                spaces, spectra = {}, {}
                for r in rows:
                    strength = extra if r.layer in run else 4.
                    budget = .5 if r.layer in run else 5.
                    edit = np.diag([20., strength, 0., 0., 0.]) * scale
                    m = core.analyze_delta(np.zeros_like(edit), edit, 1, budget * scale)
                    spaces[r.name], spectra[r.name] = m.output_subspace, m.output_spectrum
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                self.assertEqual(set(findings), {layer_name(i) for i in run} if extra > 4 else set())

    def test_terminal_expansion_requires_a_continuous_measured_trajectory(self):
        rows = [row(layer_name(i)) for i in range(19)]
        for run, measured in product([{11, 12}, {17, 18}], [False, True]):
            with self.subTest(run=run, measured=measured):
                spaces, spectra = {}, {}
                for r in rows:
                    strengths = [8.]
                    if r.layer in run or r.layer == 3:
                        strengths.append(6. if r.layer in run else 1.1)
                    spaces[r.name] = np.eye(4)[:, :len(strengths)]
                    if measured or r.layer != 3:
                        spectra[r.name] = core.OutputSpectrum(np.array(strengths), 0., 1., 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                expected = run if measured or 18 not in run else set()
                self.assertEqual(set(findings), {layer_name(i) for i in expected})

    def test_only_confirmed_interior_expansions_support_a_separate_terminal_run(self):
        rows = [row(layer_name(i)) for i in range(24)]
        for strength in [2.5, 6.]:
            with self.subTest(strength=strength):
                spaces, spectra = {}, {}
                for r in rows:
                    extra = {2: 1.1, 8: strength, 21: 2.5, 22: 6., 23: 6.}.get(r.layer)
                    values = [10.] if extra is None else [10., extra]
                    spaces[r.name] = np.eye(4)[:, :len(values)]
                    if r.layer != 2:
                        noise = 2. if r.layer == 0 else 1.
                        spectra[r.name] = core.OutputSpectrum(np.array(values), 0., noise, 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                expected = {8, 21, 22, 23} if strength == 6. else set()
                self.assertEqual(set(findings), {layer_name(i) for i in expected})

    def test_interior_core_survives_an_unsupported_terminal_extension(self):
        rows = [row(layer_name(i)) for i in range(24)]
        for gap, reverse in product([False, True], [False, True]):
            with self.subTest(gap=gap, reverse=reverse):
                spaces, spectra = {}, {}
                for r in rows:
                    depth = 23 - r.layer if reverse else r.layer
                    extra = {2: 1.1, 20: 6., 21: 6., 22: 2.5, 23: 2.5}.get(depth)
                    strengths = [10.] if extra is None else [10., extra]
                    spaces[r.name] = np.eye(4)[:, :len(strengths)]
                    if not gap or depth != 2:
                        noise = 2. if depth == 0 else 1.
                        spectra[r.name] = core.OutputSpectrum(np.array(strengths), 0., noise, 1.)
                findings = subspace_outliers(rows, spaces, 3., spectra=spectra)
                expected = range(20, 22 if gap else 24)
                self.assertEqual(set(findings), {layer_name(23 - i if reverse else i) for i in expected})

    def test_mixed_ranks_share_an_intersection_and_count_independent_moving_modes(self):
        rows = [row(layer_name(i)) for i in range(21)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(9)
        run = set(range(7, 12))
        for moving in [1, 2]:
            with self.subTest(moving=moving):
                spaces = {}
                for r in rows:
                    axes = [0, 1, 2]
                    if r.layer in run:
                        axes[-moving:] = [3, 4][:moving]
                        axes.extend([5, 6][:1 + r.layer % 2])
                    spaces[r.name] = eye[:, axes]
                findings = subspace_outliers(rows, spaces, 3., stats)
                self.assertEqual(set(findings), {layer_name(i) for i in run})
                self.assertTrue(all(f"shared output dimension {3-moving}" in why for why in findings.values()))
                # An orthogonal leading-axis change contradicts the common
                # intersection. Only a second measured moving mode suffices.
                axes = {r.name: spaces[r.name][:, 2] for r in rows}
                findings = subspace_outliers(rows, spaces, 3., stats, axes)
                expected = {layer_name(i) for i in run} if moving == 2 else set()
                self.assertEqual(set(findings), expected)

    def test_contained_lines_supply_no_evidence_about_a_planes_second_mode(self):
        rows = [row(layer_name(i)) for i in range(23)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(6)
        for target, lines in [({7}, {5, 6, 8, 9}), (set(range(6, 11)), {3})]:
            with self.subTest(target=target, lines=lines):
                spaces = {}
                for r in rows:
                    modes = [0, 2] if r.layer in target else [0, 1]
                    spaces[r.name] = eye[:, [0] if r.layer in lines else modes]
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, target)

    def test_secondary_measurement_needed_at_the_boundary_but_not_at_every_outside_layer(self):
        rows = [row(layer_name(i)) for i in range(23)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(6)
        terminal = set(range(18, 23))
        for lines, measured in [(range(5, 8), True), (range(15, 18), False)]:
            with self.subTest(lines=lines):
                spaces, axes = {}, {}
                for r in rows:
                    modes = [0, 2] if r.layer in terminal else [0, 1]
                    q = eye[:, [3] if r.layer in lines else modes]
                    spaces[r.name] = q
                    theta = np.radians(3 * (-1)**r.layer)
                    axes[r.name] = q[:, 0] if r.layer in lines else q @ [np.cos(theta), np.sin(theta)]
                ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                flagged = {r.layer for r in ranked if r.tier == "T3" and r.layer in terminal}
                self.assertEqual(flagged, terminal if measured else set())

    def test_sparse_planes_corroborate_only_runs_with_secondary_boundary_measurements(self):
        rows = [row(layer_name(i)) for i in range(22)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        eye = np.eye(5)
        first, second = {5, 6, 7}, {13, 14, 15}
        angle = np.radians(8)
        for planes, expected in [({4, 8}, first), ({12, 16}, second), ({4, 16}, set()),
                                 ({4, 8, 12, 16}, first | second)]:
            with self.subTest(planes=planes):
                axes, spaces = {}, {}
                for r in rows:
                    i = r.layer
                    member = i in first or i in second
                    q = eye[:, [0, 2 if member else 1]]
                    measured = member or i in planes
                    axis = q[:, 0] if measured else q @ [np.cos(angle), np.sin(angle)]
                    axes[r.name] = axis
                    spaces[r.name] = q if measured else axis[:, None]
                ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, expected)

    def test_cohort_endpoints_require_observations_of_the_switched_modes(self):
        rows = [row(layer_name(i)) for i in range(21)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(6)
        for reverse, measured in product([True, False], repeat=2):
            with self.subTest(reverse=reverse, measured=measured):
                spaces, run = {}, set()
                for r in rows:
                    i = len(rows) - 1 - r.layer if reverse else r.layer
                    modes = [0, 1]
                    if 13 <= i < 18:
                        modes = [0, 2]
                        run.add(r.layer)
                    elif i >= 18 and not measured:
                        modes = [0]
                    spaces[r.name] = eye[:, modes]
                found = subspace_outliers(rows, spaces, 3., stats)
                self.assertEqual(set(found), {layer_name(i) for i in run} if measured else set())

    def test_missing_boundary_affects_only_its_own_run(self):
        rows = [row(layer_name(i)) for i in range(30)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(5)
        for reverse, lines in product([False, True], [{0, 1, 2}, {7}, {17}, {7, 17}, set()]):
            with self.subTest(reverse=reverse, lines=lines):
                spaces, expected = {}, set()
                runs = [range(3, 7), range(18, 22)]
                for r in rows:
                    i = 29 - r.layer if reverse else r.layer
                    modes = [0] if i in lines else [0, 1]
                    for run in runs:
                        if i in run:
                            modes = [0, 2]
                            if run.start - 1 not in lines and run.stop not in lines:
                                expected.add(r.layer)
                    spaces[r.name] = eye[:, modes]
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, expected)

    def test_higher_rank_outlier_needs_departure_in_its_additional_modes(self):
        rows = [row(layer_name(i)) for i in range(21)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(6)
        for secondary in (2, 3):
            with self.subTest(secondary=secondary):
                spaces, axes = {}, {}
                for r in rows:
                    modes = [0] if r.layer < 12 else [1, 2]
                    if r.layer == 20:
                        modes = [0, secondary]
                    spaces[r.name] = eye[:, modes]
                    axes[r.name] = eye[:, modes[0]]
                ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                last = next(r for r in ranked if r.layer == 20)
                self.assertEqual(last.tier, "T4" if secondary == 2 else "T3")

    def test_two_moving_modes_with_an_intermittently_resolved_shared_mode(self):
        rows = [row(layer_name(i)) for i in range(19)]
        run = set(range(6, 11))
        eye = np.eye(7)
        spaces = {}
        for r in rows:
            axes = [2, 3, 4] if r.layer in run else [0, 1]
            if r.layer % 2:
                axes.append(5)
            spaces[r.name] = eye[:, axes]
        findings = subspace_outliers(rows, spaces, 3.)
        self.assertEqual(set(findings), {layer_name(i) for i in run})
        self.assertTrue(all("2 output modes" in why for why in findings.values()))

    def test_whole_space_switches_include_interior_terminal_and_recurring_groups(self):
        eye = np.eye(8)
        n = 29
        # The modes inside each plane can swap order or rotate freely.
        # Effective rank is identical across all layers.
        groups = [{0, 1, 2}, {7, 8, 9, 10}, {15, 19, 25}]
        for missing in [None, 6]:
            with self.subTest(missing=missing):
                rows = [row(layer_name(i)) for i in range(n) if i != missing]
                stats = {r.name: Measurements(2., 1.) for r in rows}
                spaces = {}
                for r in rows:
                    k = next((j for j, g in enumerate(groups, 1) if r.layer in g), 0)
                    theta = .7 * r.layer
                    c, s = np.cos(theta), np.sin(theta)
                    spaces[r.name] = eye[:, 2*k:2*k+2] @ np.array([[c, -s], [s, c]])
                ranked = rank.rank_tensors(rows[::-1], stats, subspaces=spaces)
                expected = set.union(*groups) if missing is None else groups[2]
                self.assertEqual({r.layer for r in ranked if r.tier != "T4"}, expected)
                self.assertTrue(all("smallest cross-group principal angle" in r.reason
                                    for r in ranked if r.tier == "T3"))

    def test_whole_space_switch_with_two_nearby_minor_regimes(self):
        rows = [row(layer_name(i)) for i in range(24)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(6)
        spaces = {}
        for r in rows:
            if 9 <= r.layer < 15:
                angle = np.radians(12 if r.layer >= 12 else 0)
                spaces[r.name] = np.cos(angle) * eye[:, 2:4] + np.sin(angle) * eye[:, 4:]
            else:
                spaces[r.name] = eye[:, :2]
        findings = subspace_outliers(rows, spaces, 3., stats)
        self.assertEqual(set(findings), {layer_name(i) for i in range(9, 15)})
        self.assertTrue(all("whole output subspace switch in 6 layers" in why
                            for why in findings.values()))

    def test_whole_and_shared_space_switches_need_an_abrupt_boundary(self):
        eye = np.eye(4)
        rows = [row(layer_name(i)) for i in range(24)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        minority = {r.layer for r in rows[-4:]}
        rotations = [(0., 60.), (10., 10.), (30., 60.), (45., 45.), (45., 90.), (50., 90.), (90., 90.)]
        for rotation, drift in product(rotations, (True, False)):
            with self.subTest(rotation=rotation, drift=drift):
                spaces = {}
                for r in rows:
                    t = np.clip((r.layer - 8) / 12, 0., 1.) if drift else float(r.layer in minority)
                    angles = np.radians(rotation) * t
                    spaces[r.name] = eye[:, :2] * np.cos(angles) + eye[:, 2:] * np.sin(angles)
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier != "T4"}, set() if drift else minority)

    def test_separate_switch_boundaries_are_excluded_from_ordinary_steps(self):
        rows = [row(layer_name(i)) for i in range(14)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(6)
        first, second = {1, 3, 5}, {7, 9, 11}
        spaces = {}
        for r in rows:
            basis = eye[:, :2]
            if r.layer in first:
                basis = (basis + eye[:, 2:4]) / np.sqrt(2)
            elif r.layer in second:
                basis = eye[:, 4:]
            spaces[r.name] = basis
        found = subspace_outliers(rows, spaces, 3., stats)
        self.assertEqual(set(found), {layer_name(i) for i in first | second})

    def test_alternating_coherent_spaces_without_consecutive_outside_layers(self):
        rows = [row(layer_name(i)) for i in range(15)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(4)
        spaces = {r.name: eye[:, 2:] if r.layer % 2 else eye[:, :2] for r in rows}
        found = subspace_outliers(rows, spaces, 3., stats)
        self.assertEqual(set(found), {r.name for r in rows if r.layer % 2})

    def test_distant_similar_leading_axes_with_orthogonal_boundary_axes(self):
        rows = [row(layer_name(i)) for i in range(20)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        eye = np.eye(4)
        group = set(range(7, 12))
        rotation = np.radians([16., 80.])
        changed = eye[:, :2] * np.cos(rotation) + eye[:, 2:] * np.sin(rotation)
        spaces = {r.name: changed if r.layer in group else eye[:, :2] for r in rows}
        axes = {r.name: spaces[r.name][:, int(r.layer in (6, 12))] for r in rows}
        for observed in [range(20), [3, 6, 9, 12, 18], [4, 9, 15], [4, 8, 9, 15]]:
            with self.subTest(observed=observed):
                samples = {layer_name(i): axes[layer_name(i)] for i in observed}
                found = subspace_outliers(rows, spaces, 3., stats, samples)
                self.assertEqual(set(found), {layer_name(i) for i in group})

    def test_sparse_leading_samples_across_one_or_two_multimode_boundaries(self):
        rows = [row(layer_name(i)) for i in range(24)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        eye = np.eye(6)
        rotation = np.radians([16., 80.])
        second = eye[:, :2] * np.cos(rotation) + eye[:, 2:4] * np.sin(rotation)
        first_run = {4, 5, 6}
        for start in (7, 12):
            with self.subTest(start=start):
                second_run = set(range(start, start + 3))
                spaces = {r.name: eye[:, 4:] for r in rows}
                spaces.update({layer_name(i): eye[:, :2] for i in first_run})
                spaces.update({layer_name(i): second for i in second_run})
                axes = {layer_name(i): spaces[layer_name(i)][:, 0]
                        for i in (4, 5, start + 1, start + 2)}
                found = subspace_outliers(rows, spaces, 3., stats, axes)
                expected = first_run | second_run
                self.assertEqual(set(found), {layer_name(i) for i in expected})

    def test_within_run_drift_preserves_two_abrupt_mode_changes(self):
        rows = [row(layer_name(i)) for i in range(22)]
        stats = {r.name: Measurements(1.6, 1.) for r in rows}
        group = set(range(8, 14))
        eye = np.eye(4)
        for interior, edge in [(62., 58.), (58., 62.)]:
            with self.subTest(interior=interior, edge=edge):
                spaces = {r.name: eye[:, :2] for r in rows}
                for i in group:
                    angles = np.radians([20., edge if i == max(group) else interior])
                    spaces[layer_name(i)] = eye[:, :2] * np.cos(angles) + eye[:, 2:] * np.sin(angles)
                axes = {layer_name(i): spaces[layer_name(i)][:, 0] for i in (11, 12, 15, 16)}
                found = subspace_outliers(rows, spaces, 3., stats, axes)
                self.assertEqual(set(found), {layer_name(i) for i in group})

    def test_sparse_boundary_axes_use_the_resolution_of_the_output_spaces(self):
        rows = [row(layer_name(i)) for i in range(24)]
        stats = {r.name: Measurements(1.6, 1.) for r in rows}
        group = set(range(9, 14))
        eye = np.eye(4)
        for weak, strong in [(0., 60.), (8., 50.), (18., 72.), (20., 90.), (32., 80.), (45., 45.)]:
            angles = np.radians([weak, strong])
            rotated = eye[:, :2] * np.cos(angles) + eye[:, 2:] * np.sin(angles)
            spaces = {r.name: rotated if r.layer in group else eye[:, :2] for r in rows}
            for samples in [(), (8, 9), (7, 8, 9), range(7, 11), range(5, 13), range(24)]:
                with self.subTest(weak=weak, strong=strong, samples=samples):
                    axes = {layer_name(i): spaces[layer_name(i)][:, 0] for i in samples}
                    ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                    self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, group)

    def test_two_moving_modes_with_a_retained_dimension(self):
        # One dimension is shared, but replacing one leading axis in a
        # three-dimensional space would retain two dimensions.
        rows = [row(layer_name(i)) for i in range(21)]
        group = {6, 7, 8, 9}
        stats = {r.name: Measurements(1.6, 1.) for r in rows}
        eye = np.eye(6)
        angles = np.radians([0., 18., 75.])
        rotated = eye[:, :3] * np.cos(angles) + eye[:, 3:] * np.sin(angles)
        for leading in (1, 2):
            with self.subTest(leading=leading):
                spaces, axes = {}, {}
                for r in rows:
                    basis = rotated if r.layer in group else eye[:, :3]
                    spaces[r.name] = np.roll(basis, r.layer % 3, axis=1)
                    axes[r.name] = (-1) ** r.layer * basis[:, leading]
                found = subspace_outliers(rows, spaces, 3., stats, axes)
                self.assertEqual(set(found), {layer_name(i) for i in group})
                self.assertTrue(all("2 output modes" in reason for reason in found.values()))

    def test_resolved_primary_replacement_with_a_retained_complement(self):
        rows = [row(layer_name(i)) for i in range(24)]
        stats = {r.name: Measurements(1.6, 1.) for r in rows}
        eye = np.eye(5)
        for dim, samples in product((2, 3), [(7, 8), range(6, 10), range(4, 12), range(24)]):
            with self.subTest(dim=dim, samples=samples):
                spaces, axes = {}, {}
                for r in rows:
                    angle = np.radians(35. if 8 <= r.layer < 14 else 0.)
                    primary = np.cos(angle) * eye[0] + np.sin(angle) * eye[-1]
                    basis = np.column_stack((primary, eye[:, 1:dim]))
                    spaces[r.name] = np.roll(basis, r.layer % dim, axis=1)
                    if r.layer in samples:
                        axes[r.name] = (-1) ** r.layer * primary
                ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                self.assertTrue(all(r.tier == "T4" for r in ranked))

    def test_separate_edit_bands_have_no_observed_subspace_boundary(self):
        # Independently edited depth bands can have distinct, constant planes.
        # Identical weights between bands supply no edit-space observations.
        rows = [row(layer_name(i), status="identical" if i in (6, 13) else "changed")
                for i in range(23)]
        edited = [r for r in rows if r.status == "changed"]
        stats = {r.name: Measurements(1.6, 1.) for r in edited}
        eye = np.eye(4)
        for rotation in [(0., 65.), (18., 75.), (90., 90.)]:
            with self.subTest(rotation=rotation):
                angles = np.radians(rotation)
                band = eye[:, :2] * np.cos(angles) + eye[:, 2:] * np.sin(angles)
                spaces = {r.name: band if 6 < r.layer < 13 else eye[:, :2] for r in edited}
                axes = {layer_name(i): spaces[layer_name(i)][:, 0] for i in (5, 7)}
                ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                self.assertEqual(len(ranked), len(edited))
                self.assertTrue(all(r.tier == "T4" for r in ranked))

    def test_whole_space_switch_compares_weakest_boundary_mode_with_outside_steps(self):
        rows = [row(layer_name(i)) for i in range(32)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        eye = np.eye(len(rows) + 5)
        group = {7, 10, 11, 12, 19}
        for angle, expected in [(32., set()), (55., group)]:
            with self.subTest(angle=angle):
                walk = eye[2]
                tilt = np.radians(angle)
                first = np.cos(tilt) * eye[0] + np.sin(tilt) * eye[1]
                spaces = {}
                for r in rows:
                    if r.layer in group:
                        spaces[r.name] = np.column_stack((first, eye[-1]))
                    else:
                        step = np.radians(36 + r.layer % 3)
                        walk = np.cos(step) * walk + np.sin(step) * eye[3 + r.layer]
                        spaces[r.name] = np.column_stack((eye[0], walk))
                findings = subspace_outliers(rows, spaces, 3., stats)
                self.assertEqual(set(findings), {layer_name(i) for i in expected})

    def test_unresolved_shared_space_with_rotated_bases_and_multiple_groups(self):
        rows = [row(layer_name(i)) for i in range(32)]
        eye = np.eye(12)
        rng = np.random.default_rng(53)
        groups = [{0, 1, 2}, {8, 11, 16}, {27, 28, 29, 30, 31}]
        for k in (1, 2):
            with self.subTest(k=k):
                spaces = {}
                for r in rows:
                    rest = eye[:, 2:4]
                    for j, group in enumerate(groups):
                        if r.layer in group:
                            rest = .5 * rest + np.sqrt(.75) * eye[:, 4+2*j:6+2*j]
                    basis = np.column_stack((eye[:, :k], rest))
                    rotation, _ = np.linalg.qr(rng.normal(size=(k+2, k+2)))
                    spaces[r.name] = basis @ rotation
                stats = {r.name: Measurements(2., 1.) for r in rows}
                findings = subspace_outliers(rows[::-1], spaces, 3., stats)
                self.assertEqual(set(findings), {layer_name(i) for i in set.union(*groups)})
                self.assertTrue(all(f"shared output dimension {k}" in s for s in findings.values()))

    def test_unresolved_shared_axis_with_rotating_or_wandering_peers(self):
        n = 32
        eye = np.eye(n + 3)
        rows = [row(layer_name(i)) for i in range(n)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        group = {5, 11, 12, 13, 18}
        for wandering, switched in product((False, True), repeat=2):
            with self.subTest(wandering=wandering, switched=switched):
                spaces = {}
                secondary = eye[1]
                for r in rows:
                    if switched and r.layer in group:
                        axis = .5 * eye[1] + np.sqrt(.75) * eye[-1]
                    elif wandering:
                        if r.layer:
                            theta = np.radians(38 + r.layer % 3)
                            secondary = np.cos(theta) * secondary + np.sin(theta) * eye[r.layer+1]
                        axis = secondary
                    else:
                        theta = np.radians(2 * r.layer)
                        axis = np.cos(theta) * eye[1] + np.sin(theta) * eye[2]
                    spaces[r.name] = np.column_stack((eye[0], axis))
                findings = subspace_outliers(rows, spaces, 3., stats)
                expected = group if switched else set()
                self.assertEqual(set(findings), {layer_name(i) for i in expected})

    def test_pairwise_intersections_rotate_without_a_common_axis(self):
        rows = [row(layer_name(i)) for i in range(32)]
        eye = np.eye(3)
        spaces = {}
        for r in rows:
            # Each background plane meets the minority plane in a line, but
            # that line rotates across the majority. No common line survives.
            theta = np.radians(3 * r.layer)
            axis = np.cos(theta) * eye[0] + np.sin(theta) * eye[1]
            spaces[r.name] = eye[:, :2] if 12 <= r.layer < 17 else np.column_stack((axis, eye[2]))
        stats = {r.name: Measurements(2., 1.) for r in rows}
        self.assertEqual(subspace_outliers(rows, spaces, 3., stats), {})

    def test_nearly_shared_axis_also_changes_at_the_boundary(self):
        rows = [row(layer_name(i)) for i in range(20)]
        eye = np.eye(5)
        spaces = {}
        for r in rows:
            member = r.layer >= 16
            angle = np.radians(3 * member)
            axis = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
            step = np.radians(2 * (r.layer % 2))
            secondary = np.cos(step) * eye[3 if member else 2] + np.sin(step) * eye[4]
            spaces[r.name] = np.column_stack((axis, secondary))
        stats = {r.name: Measurements(2., 1.) for r in rows}
        self.assertEqual(subspace_outliers(rows, spaces, 3., stats), {})

    def test_separation_must_exceed_reconstructed_shared_space_spread(self):
        rows = [row(layer_name(i)) for i in range(16)]
        eye = np.eye(3)
        spaces = {}
        tilt = np.radians(20.)
        for r in rows:
            # Each pair intersects exactly. The estimated common line varies
            # by 8 degrees as the background planes turn around their cone.
            azimuth = np.radians(12.) * np.sin(2 * np.pi * r.layer / 11)
            line = np.cos(azimuth) * eye[0] + np.sin(azimuth) * eye[1]
            perp = -np.sin(azimuth) * eye[0] + np.cos(azimuth) * eye[1]
            basis = np.column_stack((line, np.cos(tilt) * perp + np.sin(tilt) * eye[2]))
            spaces[r.name] = basis if r.layer < 12 else eye[:, :2]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        self.assertEqual(subspace_outliers(rows, spaces, 3., stats), {})

    def test_partially_resolved_leading_axes_can_contradict_shared_modes(self):
        rows = [row(layer_name(i)) for i in range(20)]
        eye = np.eye(5)
        spaces = {r.name: eye[:, [0, 1, 3 if r.layer >= 16 else 2]] for r in rows}
        stats = {r.name: Measurements(2., 1.) for r in rows}
        for axis in (0, 1, 3):
            with self.subTest(axis=axis):
                # Some layers resolve a leading axis. A change inside the
                # common plane and a change out of it both contradict stability.
                directions = {r.name: eye[axis if r.layer >= 16 else 0] for r in rows if r.layer % 2}
                findings = subspace_outliers(rows, spaces, 3., stats, directions)
                expected = {layer_name(i) for i in range(16, 20)} if axis == 0 else set()
                self.assertEqual(set(findings), expected)

    def test_sparse_leading_measurements_use_observed_within_group_drift(self):
        rows = [row(layer_name(i)) for i in range(20)]
        eye = np.eye(5)
        spaces = {r.name: eye[:, [0, 1, 3 if r.layer >= 14 else 2]] for r in rows}
        stats = {r.name: Measurements(2., 1.) for r in rows}
        for jump in (0, 6):
            with self.subTest(jump=jump):
                directions = {}
                for i in [2, 6, 10, 13, 14, 18]:
                    theta = np.radians(i + (jump if i >= 14 else 0))
                    directions[layer_name(i)] = np.cos(theta) * eye[0] + np.sin(theta) * eye[1]
                findings = subspace_outliers(rows, spaces, 3., stats, directions)
                expected = {layer_name(i) for i in range(14, 20)} if jump == 0 else set()
                self.assertEqual(set(findings), expected)

    def test_leading_axes_without_a_drift_sample_use_switched_angle(self):
        rows = [row(layer_name(i)) for i in range(20)]
        eye = np.eye(4)
        stats = {r.name: Measurements(2., 1.) for r in rows}
        for switch, jump, detected in [(15, 2, True), (15, 7, False),
                                       (15, 20, False), (45, 7, True)]:
            with self.subTest(switch=switch, jump=jump):
                angle = np.radians(switch)
                changed = np.cos(angle) * eye[2] + np.sin(angle) * eye[3]
                spaces = {r.name: np.column_stack((eye[0], eye[1], changed if r.layer >= 14 else eye[2]))
                          for r in rows}
                theta = np.radians(jump)
                axes = {layer_name(13): eye[0], layer_name(14): np.cos(theta) * eye[0] + np.sin(theta) * eye[1]}
                findings = subspace_outliers(rows, spaces, 3., stats, axes)
                expected = {layer_name(i) for i in range(14, 20)} if detected else set()
                self.assertEqual(set(findings), expected)

    def test_recurring_clique_with_matched_rank_among_wandering_peers(self):
        rows = [row(layer_name(i)) for i in range(30)]
        stats = {r.name: Measurements(2., 1.) for r in rows}
        minority = {3, 9, 14, 16, 26}
        eye = np.eye(2 * len(rows) + 2)
        for shared_axis in [True, False]:
            with self.subTest(shared_axis=shared_axis):
                bg = eye[:, :2]
                spaces = {}
                axes = {r.name: eye[:, 0] for r in rows} if shared_axis else {}
                for r in rows:
                    if r.layer in minority:
                        basis = eye[:, -2:]
                    else:
                        if r.layer:
                            theta = np.radians(38 + r.layer % 4)
                            bg = bg * np.cos(theta) + eye[:, 2*r.layer:2*r.layer+2] * np.sin(theta)
                        basis = bg
                    spaces[r.name] = np.column_stack((eye[:, 0], basis[:, 1])) if shared_axis else basis
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces, directions=axes)
                self.assertEqual({r.layer for r in ranked if r.tier != "T4"}, minority)

    def test_gradual_transitions_between_three_plateaus(self):
        eye = np.eye(3)
        angles = [0.] * 20 + list(range(5, 35, 5)) + [30.] * 3 + list(range(35, 65, 5)) + [60.] * 3
        rows = [row(layer_name(i)) for i in range(len(angles))]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        axes, spaces = {}, {}
        for r, theta in zip(rows, np.radians(angles)):
            axes[r.name] = eye[0]
            spaces[r.name] = np.column_stack((eye[0], np.cos(theta) * eye[1] + np.sin(theta) * eye[2]))
        ranked = rank.rank_tensors(rows, stats, subspaces=spaces, directions=axes)
        self.assertTrue(all(r.tier == "T4" for r in ranked))

    def test_terminal_switch_separates_all_secondary_modes_relative_to_peer_spread(self):
        eye = np.eye(6)
        rows = [row(layer_name(i)) for i in range(16)]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        axes = {r.name: (-1) ** r.layer * eye[0] for r in rows}
        for early, dim, angle in product([False, True], [2, 3], [40., 60., 90.]):
            with self.subTest(early=early, dim=dim, angle=angle):
                minority = rows[:4] if early else rows[-4:]
                spaces = {}
                for r in rows:
                    # Within-group spread gives a measured angular resolution.
                    theta = np.radians((angle if r in minority else 0.) + .1 * (r.layer % 2))
                    secondary = np.cos(theta) * eye[:, 1:dim] + np.sin(theta) * eye[:, 3:dim+2]
                    basis = np.column_stack((eye[0], secondary))
                    spaces[r.name] = np.roll(basis, r.layer % dim, axis=1)
                ranked = rank.rank_tensors(rows, stats, directions=axes, subspaces=spaces)
                flagged = {r.name for r in ranked if r.tier == "T3"}
                self.assertEqual(flagged, {r.name for r in minority})
                self.assertTrue(all("minimum secondary angle" in r.reason for r in ranked if r.tier == "T3"))

    def test_terminal_switch_calibrates_the_closest_secondary_mode(self):
        eye = np.eye(5)
        rows = [row(layer_name(i)) for i in range(16)]
        directions = {r.name: eye[0] for r in rows}
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        for step, expected in [(12., set()), (40., {12, 13, 14, 15})]:
            with self.subTest(step=step):
                spaces = {}
                for r in rows:
                    # One secondary axis jumps 90 degrees. The other varies
                    # by 5 degrees within groups and changes by a smaller angle.
                    terminal = r.layer >= 12
                    theta = np.radians((step if terminal else 0.) + 5 * (r.layer % 2))
                    axis = np.cos(theta) * eye[1] + np.sin(theta) * eye[2]
                    spaces[r.name] = np.column_stack((eye[0], axis, eye[4 if terminal else 3]))
                ranked = rank.rank_tensors(rows, stats, directions=directions, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, expected)

    def test_terminal_primary_jump_compared_with_neighbors_under_global_drift(self):
        rows = [row(layer_name(i)) for i in range(16)]
        eye = np.eye(5)
        for jump, expected in [(0, {12, 13, 14, 15}), (7, set())]:
            with self.subTest(jump=jump):
                axes, spaces = {}, {}
                for r in rows:
                    terminal = r.layer >= 12
                    # An isolated excursion in the background must not raise
                    # the tolerance for a primary-axis jump at the boundary.
                    theta = np.radians(r.layer + (jump if terminal else 0) + (6 if r.layer == 4 else 0))
                    axes[r.name] = np.cos(theta) * eye[0] + np.sin(theta) * eye[1]
                    spaces[r.name] = np.column_stack((axes[r.name], eye[3 if terminal else 2]))
                stats = {r.name: Measurements(1.5, 1.) for r in rows}
                result = subspace_outliers(rows, spaces, 3., stats, axes)
                self.assertEqual(set(result), {layer_name(i) for i in expected})

    def test_terminal_switch_needs_an_observed_boundary(self):
        eye = np.eye(4)
        for missing, resolved, shift in [(11, True, 0.), (5, True, 0.), (None, False, 0.), (None, True, .1)]:
            with self.subTest(missing=missing, resolved=resolved, shift=shift):
                rows = [row(layer_name(i)) for i in range(16) if i != missing]
                spaces, directions = {}, {}
                for r in rows:
                    angle = shift if r.layer >= 12 else 0.
                    primary = np.cos(angle) * eye[0] + np.sin(angle) * eye[3]
                    spaces[r.name] = np.column_stack((primary, eye[2 if r.layer >= 12 else 1]))
                    if resolved:
                        directions[r.name] = primary
                stats = {r.name: Measurements(1.5, 1.) for r in rows}
                expected = {layer_name(i) for i in range(12, 16)} if missing is None else set()
                self.assertEqual(set(subspace_outliers(rows, spaces, 3., stats, directions)), expected)

    def test_change_of_primary_concept_does_not_corroborate_secondary_switch(self):
        # These edit spaces intersect in two axes. The dominant direction
        # changes between groups, even though it lies in their intersection.
        eye = np.eye(5)
        rows = [row(layer_name(i)) for i in range(17)]
        spaces, directions = {}, {}
        for r in rows:
            member = 7 <= r.layer <= 10
            spaces[r.name] = eye[:, [0, 1, 3 if member else 2]]
            angle = .1 if member else 0.
            directions[r.name] = np.cos(angle) * eye[0] + np.sin(angle) * eye[1]
        stats = {r.name: Measurements(1.5, 1.) for r in rows}
        self.assertEqual(subspace_outliers(rows, spaces, 3., stats, directions), {})

    def test_recurring_subspace_needs_a_switch_or_spectral_corroboration(self):
        rows = [row(layer_name(i)) for i in range(15)]
        axes = np.eye(17)
        for selected, background_shared, group_rank, expected in [
            ({0, 4, 8}, False, 1.3, {0, 4, 8}),
            ({0, 4, 8}, False, 1.7, {0, 4, 8}),
            ({0, 4, 8}, False, 1.5, set()),   # one concept recurs at typical strength
            ({0, 1, 2}, False, 1.3, set()),   # contiguous editing regime
            ({0, 4, 8}, True, 1.5, {0, 4, 8}),  # observed switches under a shared axis
            ({4, 5, 6}, True, 1.3, {4, 5, 6}),
            ({0, 4, 8}, True, 1.3, {0, 4, 8}),
        ]:
            with self.subTest(selected=selected, shared=background_shared, rank=group_rank):
                spaces, stats = {}, {}
                for i, r in enumerate(rows):
                    second = 2 if background_shared else 2 + i
                    if i in selected:
                        second = 1
                    elif i == 11 and not background_shared:
                        second = 2 + 10  # one incidental pair is insufficient to form another cohort
                    spaces[r.name] = axes[:, [0, second]]
                    stats[r.name] = Measurements(group_rank if i in selected else 1.5, 1.)
                ranked = rank.rank_tensors(rows, stats, subspaces=spaces)
                self.assertEqual({r.layer for r in ranked if r.tier == "T3"}, expected)

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
    def test_resolved_line_against_a_plane_through_cli(self):
        # A run drops one measurable mode. Only its angle to the plane
        # distinguishes the two cases, with the same per-layer spectra.
        for contained in [False, True]:
            with self.subTest(contained=contained), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                base, suspect = {}, {}
                run = set(range(5, 10))
                for i in range(18):
                    w = np.eye(8)
                    edited = w.copy()
                    modes = [0, 1]
                    if i in run:
                        modes = [0 if contained else 2]
                    edited[modes] = 0
                    base[layer_name(i)] = ("F32", w)
                    suspect[layer_name(i)] = ("F32", edited)
                write_safetensors(tmp / "base.safetensors", base)
                write_safetensors(tmp / "suspect.safetensors", suspect)
                rc, out, err = run_cli("--base", str(tmp / "base.safetensors"),
                                       "--suspect", str(tmp / "suspect.safetensors"), "--out", str(tmp / "out"))
                self.assertEqual(rc, 0, out + err)
                with (tmp / "out" / "ranking.csv").open() as f:
                    flagged = {int(r["layer"]): r["reason"] for r in csv.DictReader(f) if r["tier"] != "T4"}
                self.assertEqual(set(flagged), set() if contained else run)
                self.assertTrue(all("whole output subspace" in why for why in flagged.values()))

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
            self.assertIn("inner product constraints", finding["reason"])

    def test_tied_modes_switch_output_space_through_cli(self):
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory() as tmp:
                self._check_tied_modes_switch(Path(tmp), shared)

    def _check_tied_modes_switch(self, tmp, shared):
        base, suspect = {}, {}
        for i in range(16):
            w = np.eye(8, dtype=np.float32)
            edited = w.copy()
            modes = [0 if shared else 2, 3] if 5 <= i <= 8 else [0, 1]
            edited[modes] = 0
            base[layer_name(i)] = ("F32", w)
            suspect[layer_name(i)] = ("F32", edited)
        write_safetensors(tmp / "b.safetensors", base)
        write_safetensors(tmp / "s.safetensors", suspect)
        rc, out, err = run_cli("--base", str(tmp / "b.safetensors"), "--suspect", str(tmp / "s.safetensors"),
                               "--out", str(tmp / "out"), "-k", "1")
        self.assertEqual(rc, 0, out + err)
        with (tmp / "out" / "ranking.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["name"] for r in rows if r["tier"] == "T3"}, {layer_name(i) for i in range(5, 9)})
        evidence = "shared output dimension 1" if shared else "whole output subspace switch"
        self.assertTrue(all(evidence in r["reason"] for r in rows if r["tier"] == "T3"))

    def test_rank_three_shared_subspace_through_cli(self):
        # Both tied and separated spectra have the same output space except
        # for one layer's third axis. Keep only one CSV singular value.
        rng = np.random.default_rng(27)
        for shape, strengths in product([(8, 12), (12, 8)], [(1., 1., 1.), (1., .32, .11)]):
            with self.subTest(shape=shape, strengths=strengths), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                base, suspect = {}, {}
                for i in range(12):
                    left = np.eye(shape[0])[:, [0, 1, 3 if i == 6 else 2]]
                    right = np.linalg.qr(rng.normal(size=(shape[1], 3)))[0]
                    untouched = np.eye(*shape)
                    untouched -= left @ (left.T @ untouched)
                    base[layer_name(i)] = ("BF16", to_bf16(untouched + (left * strengths) @ right.T))
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

    def test_inner_product_certificate_on_a_strict_contraction_through_cli(self):
        # The strongest changed coordinate is an ordinary half removal.
        # Every scalar quadratic response is negative. Only the joint
        # cross responses exclude a symmetric partial removal.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            name = layer_name(0)
            b = np.eye(12, 9, dtype=np.float32)
            b[0, 0] = 4.
            s = b / 2
            s[3, 4] = 0.125
            form = s.T @ (s - b)
            self.assertLess(np.linalg.eigvalsh((form + form.T) / 2)[-1], 0.)
            write_safetensors(tmp / "b.safetensors", {name: ("BF16", b)})
            write_safetensors(tmp / "s.safetensors", {name: ("BF16", s)})
            rc, out, err = run_cli("--base", str(tmp / "b.safetensors"),
                                   "--suspect", str(tmp / "s.safetensors"), "--out", str(tmp / "out"))
            self.assertEqual(rc, 0, out + err)
            m = rank.read_delta_csv(tmp / "out" / "delta.csv")[name]
            self.assertEqual(m.reciprocity_excess, 0.)
            self.assertGreater(m.removal_excess, 0.)
            with (tmp / "out" / "ranking.csv").open() as f:
                finding, = csv.DictReader(f)
            self.assertEqual(finding["tier"], "T3")
            self.assertIn("inner product constraints", finding["reason"])

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
