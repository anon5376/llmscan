import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import write_safetensors
from llmscan import core, delta, inventory
from llmscan.safetensors_io import Checkpoint


def orthonormal(rng, n, r):
    q, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return q


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bf16 value (ties to even), as a bf16 re-save would."""
    bits = x.astype(np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


class TestAnalyzeDelta(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_known_spectrum_and_participation_ratio(self):
        s = np.array([5.0, 3.0, 1.0])
        dw = orthonormal(self.rng, 64, 3) @ np.diag(s) @ orthonormal(self.rng, 32, 3).T
        base = self.rng.standard_normal((64, 32)).astype(np.float32)
        stats = core.analyze_delta(base.copy(), (base + dw).astype(np.float32), k=8)
        self.assertEqual(len(stats.singular_values), 8)
        np.testing.assert_allclose(stats.singular_values[:3], s, rtol=1e-4)
        np.testing.assert_allclose(stats.singular_values[3:], 0.0, atol=1e-3)
        self.assertAlmostEqual(stats.effective_rank, (s @ s) ** 2 / np.sum(s ** 4), places=3)

    def test_matches_a_full_svd_on_a_dense_change(self):
        base = self.rng.standard_normal((40, 24)).astype(np.float32)
        suspect = self.rng.standard_normal((40, 24)).astype(np.float32)
        expected = np.linalg.svd(suspect.astype(np.float64) - base, compute_uv=False)
        stats = core.analyze_delta(base.copy(), suspect.copy(), k=5)
        np.testing.assert_allclose(stats.singular_values, expected[:5], rtol=1e-6)
        energy = expected ** 2
        self.assertAlmostEqual(stats.effective_rank, energy.sum() ** 2 / np.sum(energy ** 2), places=6)

    def test_abliteration_is_rank_one_even_after_a_bf16_resave(self):
        w = to_bf16(self.rng.standard_normal((256, 768)).astype(np.float32) * 0.02)
        r = orthonormal(self.rng, 256, 1)
        edited = to_bf16((w - r @ (r.T @ w)).astype(np.float32))  # W' = W − r̂ r̂ᵀ W, saved as bf16
        stats = core.analyze_delta(w.copy(), edited, k=4)
        self.assertLess(stats.effective_rank, 1.05)
        self.assertGreater(stats.singular_values[0], 20 * stats.singular_values[1])

    def test_wide_and_tall_give_the_same_answer(self):
        base = self.rng.standard_normal((12, 30)).astype(np.float32)
        suspect = self.rng.standard_normal((12, 30)).astype(np.float32)
        wide = core.analyze_delta(base.copy(), suspect.copy(), k=12)
        tall = core.analyze_delta(base.T.copy(), suspect.T.copy(), k=12)
        np.testing.assert_allclose(wide.singular_values, tall.singular_values, rtol=1e-9)
        self.assertAlmostEqual(wide.effective_rank, tall.effective_rank, places=9)

    def test_row_blocks_do_not_change_the_result(self):
        base = self.rng.standard_normal((50, 9)).astype(np.float32)
        suspect = self.rng.standard_normal((50, 9)).astype(np.float32)
        whole = core.analyze_delta(base.copy(), suspect.copy(), k=9)
        with mock.patch.object(core, "_ROWS_PER_BLOCK", 7):
            blocked = core.analyze_delta(base.copy(), suspect.copy(), k=9)
        np.testing.assert_allclose(blocked.singular_values, whole.singular_values, rtol=1e-9)

    def test_fewer_values_than_k_when_the_matrix_is_small(self):
        base = np.zeros((3, 5), dtype=np.float32)
        stats = core.analyze_delta(base.copy(), np.ones((3, 5), dtype=np.float32), k=32)
        self.assertEqual(len(stats.singular_values), 3)
        self.assertAlmostEqual(stats.singular_values[0], np.sqrt(15), places=5)
        self.assertAlmostEqual(stats.effective_rank, 1.0, places=6)

    def test_zero_change_has_rank_zero(self):
        w = self.rng.standard_normal((6, 4)).astype(np.float32)
        stats = core.analyze_delta(w.copy(), w.copy(), k=2)
        self.assertEqual(stats.effective_rank, 0.0)
        np.testing.assert_array_equal(stats.singular_values, [0.0, 0.0])

    def test_extreme_f64_scales_keep_rank_and_spectrum(self):
        for scale in (1e-150, 1e-100, 1e80, 1e150):
            stats = core.analyze_delta(np.zeros((16, 16)), np.eye(16) * scale, k=3)
            self.assertAlmostEqual(stats.effective_rank, 16.0, places=9, msg=f"scale {scale}")
            np.testing.assert_allclose(stats.singular_values, [scale] * 3, rtol=1e-12)

    def test_non_finite_change_is_reported_as_nan(self):
        base = np.zeros((4, 4), dtype=np.float32)
        suspect = base.copy()
        suspect[1, 2] = np.inf
        stats = core.analyze_delta(base, suspect, k=2)
        self.assertTrue(np.isnan(stats.effective_rank))
        self.assertTrue(np.isnan(stats.singular_values).all())

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            core.analyze_delta(np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32), k=1)
        with self.assertRaises(ValueError):
            core.analyze_delta(np.zeros(4, np.float32), np.zeros(4, np.float32), k=1)
        with self.assertRaises(ValueError):
            core.analyze_delta(np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32), k=0)


class TestScanPrecision(unittest.TestCase):
    def test_f64_change_below_float32_resolution_keeps_its_rank(self):
        base = np.ones((16, 16))
        suspect = base + np.eye(16) * 5e-8  # a full-rank change that float32 would round to zero
        suspect[0, 0] = 1 + 1e-7
        energy = np.linalg.svd(suspect - base, compute_uv=False) ** 2
        expected = energy.sum() ** 2 / np.sum(energy ** 2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            name = "model.layers.0.self_attn.o_proj.weight"
            write_safetensors(tmp / "a.safetensors", {name: ("F64", base)})
            write_safetensors(tmp / "b.safetensors", {name: ("F64", suspect)})
            with Checkpoint(tmp / "a.safetensors") as a, Checkpoint(tmp / "b.safetensors") as b:
                delta.scan(a, b, inventory.compare(a, b), tmp / "delta.csv", k=4, log=lambda _: None)
            (row,) = csv.DictReader(open(tmp / "delta.csv", encoding="utf-8"))
        self.assertGreater(expected, 10)
        self.assertAlmostEqual(float(row["effective_rank"]), expected, places=3)

    def test_tiny_f64_change_through_the_scan_does_not_crash(self):
        name = "model.layers.0.self_attn.o_proj.weight"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_safetensors(tmp / "a.safetensors", {name: ("F64", np.zeros((16, 16)))})
            write_safetensors(tmp / "b.safetensors", {name: ("F64", np.eye(16) * 1e-100)})
            with Checkpoint(tmp / "a.safetensors") as a, Checkpoint(tmp / "b.safetensors") as b:
                delta.scan(a, b, inventory.compare(a, b), tmp / "delta.csv", k=4, log=lambda _: None)
            (row,) = csv.DictReader(open(tmp / "delta.csv", encoding="utf-8"))
        self.assertAlmostEqual(float(row["effective_rank"]), 16.0, places=4)
        self.assertEqual(row["sv_001"], "1e-100")


if __name__ == "__main__":
    unittest.main()
