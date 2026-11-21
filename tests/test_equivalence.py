"""The ΔW core after the runtime and memory work, against the one it replaced (legacy_core.py).

Wherever the eigensolver is the one the old core used, every number must be the same to the last
bit. Where the values-only solver runs instead, the two differ by solver rounding and every
reported number, each singular value included, must agree far more closely than it is printed.
"""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import legacy_core
from helpers import write_safetensors
from llmscan import core, delta, inventory
from llmscan.names import RESIDUAL_WRITERS, TOKEN_MATRICES, canonical_kind
from llmscan.safetensors_io import Checkpoint

SCALARS = ("effective_rank", "frobenius", "column_fraction", "second_tail_share", "projection_gain",
           "projection_error", "column_spike", "tail_norm_fraction")
RELATIVE = 1e-8  # between the two eigensolvers; measured differences stay below 1e-9
SHAPES = [(1, 1), (1, 9), (9, 1), (2, 2), (12, 12), (40, 12), (12, 40), (33, 33)]
KINDS = ["projection", "partial", "amplified", "dense", "low rank", "projection and noise", "two modes",
         "near the separation limit", "repeated leading value"]
DTYPES = ["F32", "BF16", "F16", "F64"]


def stored(x: np.ndarray, dtype: str) -> np.ndarray:
    """The array a checkpoint of that dtype hands to the core."""
    if dtype == "F64":
        return np.array(x, np.float64)
    if dtype == "F16":
        return x.astype(np.float16).astype(np.float32)
    x = x.astype(np.float32)
    return (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32) if dtype == "BF16" else x


def changed(rng, base: np.ndarray, kind: str) -> np.ndarray:
    m, n = base.shape
    u = rng.standard_normal(m)
    u /= np.linalg.norm(u)
    removal = np.outer(u, u @ base)
    if kind == "projection":
        return base - removal
    if kind == "partial":
        return base - rng.uniform(0.1, 0.9) * removal
    if kind == "amplified":
        return base + rng.uniform(0.5, 3) * removal
    if kind == "dense":
        return base + rng.standard_normal((m, n)) * 10.0 ** rng.integers(-5, 0)
    if kind == "low rank":
        r = int(rng.integers(1, 5))
        return base + 0.01 * rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    if kind == "projection and noise":
        return base - removal + rng.standard_normal((m, n)) * 10.0 ** rng.integers(-11, -1)
    if kind == "two modes":
        return base - removal - 10.0 ** rng.integers(-5, 1) * np.outer(rng.standard_normal(m), rng.standard_normal(n))
    r = min(m, n)
    left, _ = np.linalg.qr(rng.standard_normal((m, r)))
    right, _ = np.linalg.qr(rng.standard_normal((n, r)))
    values = np.zeros(r)
    second = 1.0 + rng.choice([-1e-9, 0.0, 1e-9, 1e-4])  # σ₂/σ₁ = 1/2 is where the leading mode stops being separated
    values[:2] = ((2.0, second) if kind == "near the separation limit" else (1.0, 1.0))[:r]
    return base + (left * values) @ right.T


def analyze(base, suspect, k: int) -> tuple[core.DeltaStats, bool]:
    """The new core's answer, and whether its eigenvalues came from the solver the old core used."""
    solved = []

    def solve(gram, reported, original=core._eigenvalues):
        solved.append(original(gram, reported))
        return solved[-1]

    with mock.patch.object(core, "_eigenvalues", solve):
        stats = core.analyze_delta(base, suspect, k)
    return stats, not solved or solved[-1][1] is not None


def weak_tail(rng, shape, strong, weak) -> tuple[np.ndarray, np.ndarray]:
    """An F64 pair whose change has the singular values `strong` followed by `weak`."""
    values = np.array(list(strong) + list(weak))
    left, _ = np.linalg.qr(rng.standard_normal((shape[0], len(values))))
    right, _ = np.linalg.qr(rng.standard_normal((shape[1], len(values))))
    base = rng.standard_normal(shape)
    return base, base + (left * values) @ right.T


class Equivalence(unittest.TestCase):
    def compare(self, old: core.DeltaStats, new: core.DeltaStats, exact: bool) -> None:
        self.assertEqual(len(old.singular_values), len(new.singular_values))
        for name in SCALARS + ("direction",):
            self.assertEqual(getattr(old, name) is None, getattr(new, name) is None, name)
        if exact:
            np.testing.assert_array_equal(old.singular_values, new.singular_values)
            for name in SCALARS:
                if not (name == "effective_rank" and np.isnan(old.effective_rank) and np.isnan(new.effective_rank)):
                    self.assertEqual(getattr(old, name), getattr(new, name), name)
            if old.direction is not None:
                np.testing.assert_array_equal(old.direction, new.direction)
            return
        np.testing.assert_allclose(new.singular_values, old.singular_values, rtol=RELATIVE, atol=0)
        for name in SCALARS:
            if getattr(old, name) is not None:
                self.assertLessEqual(abs(getattr(old, name) - getattr(new, name)), RELATIVE * abs(getattr(old, name)), name)
        if old.direction is not None:  # a singular vector is defined up to its sign
            sign = 1.0 if old.direction @ new.direction > 0 else -1.0
            np.testing.assert_allclose(sign * new.direction, old.direction, rtol=0, atol=1e-10)

    def against_legacy(self, base: np.ndarray, suspect: np.ndarray, k: int, base_dtype="F32", suspect_dtype=None) -> bool:
        """Compare every output of the two cores. True if the values-only eigensolver ran."""
        dtypes = base_dtype, suspect_dtype or base_dtype
        old = legacy_core.analyze_delta(base.copy(), suspect.copy(), k)
        old_excess = legacy_core.rank_one_excess(base, old, *dtypes)
        untouched, delta_w = base.copy(), suspect.copy()
        new, exact = analyze(base, delta_w, k)
        np.testing.assert_array_equal(base, untouched)
        np.testing.assert_array_equal(delta_w, suspect - base)  # as before, the suspect copy becomes ΔW
        self.compare(old, new, exact)
        excess = core.rank_one_excess(base, new, *dtypes)
        # The base norm summed during the projection pass is the one rank_one_excess would sum itself.
        self.assertEqual(excess, core.rank_one_excess(base, new._replace(base_energy=None), *dtypes))
        np.testing.assert_array_equal(base, untouched)
        self.assertEqual(old_excess is None, excess is None)
        if old_excess is not None:
            self.assertLessEqual(abs(old_excess - excess), 0 if exact else RELATIVE * abs(old_excess))
        return not exact


class TestCoreMatchesLegacy(Equivalence):
    def test_every_signal_for_every_kind_of_change(self):
        rng = np.random.default_rng(20261122)
        paths = {False: 0, True: 0}
        separated = 0
        for i, (shape, kind, dtype) in enumerate((s, c, d) for s in SHAPES for c in KINDS for d in DTYPES):
            base = rng.standard_normal(shape) * 10.0 ** rng.integers(-2, 3)
            a, b = stored(base, dtype), stored(changed(rng, base, kind), dtype)
            if np.array_equal(a, b):  # the pipeline never sends an unchanged tensor
                continue
            with self.subTest(shape=shape, kind=kind, dtype=dtype):
                paths[self.against_legacy(a, b, (1, 4, 32)[i % 3], dtype)] += 1
                separated += legacy_core.analyze_delta(a.copy(), b.copy(), 1).direction is not None
        # both eigensolver paths, and both outcomes of the separation test, are well covered
        self.assertGreater(paths[False], 50)
        self.assertGreater(paths[True], 50)
        self.assertGreater(separated, 50)
        self.assertGreater(sum(paths.values()) - separated, 30)

    def test_row_blocks_including_a_short_last_one(self):
        rng = np.random.default_rng(7)
        with mock.patch.object(core, "_ROWS_PER_BLOCK", 5), mock.patch.object(legacy_core, "_ROWS_PER_BLOCK", 5):
            for shape in [(23, 4), (4, 23), (10, 10), (5, 3), (6, 6)]:
                for kind in ("projection", "dense", "two modes"):
                    for dtype in DTYPES:
                        base = rng.standard_normal(shape)
                        with self.subTest(shape=shape, kind=kind, dtype=dtype):
                            self.against_legacy(stored(base, dtype), stored(changed(rng, base, kind), dtype), 3, dtype)

    def test_more_rows_than_one_real_block(self):
        rng = np.random.default_rng(8)
        rows = core._ROWS_PER_BLOCK + 37
        for shape in [(rows, 6), (6, rows)]:
            for kind in ("projection", "dense"):
                base = rng.standard_normal(shape).astype(np.float32)
                with self.subTest(shape=shape, kind=kind):
                    self.against_legacy(base, changed(rng, base, kind).astype(np.float32), 4)

    def test_extreme_f64_scales_and_the_subnormal_floor(self):
        rng = np.random.default_rng(9)
        base = rng.standard_normal((14, 9))
        for scale in (1e-300, 1e-160, 1e150, 1e300):
            for kind in ("projection", "partial", "dense"):
                with self.subTest(scale=scale, kind=kind):
                    self.against_legacy(base * scale, changed(rng, base, kind) * scale, 4, "F64")
        tiny = np.zeros((6, 5))
        tiny[2, 3] = 5e-324
        for suspect in (tiny, np.outer(np.arange(6.), np.arange(5.)) * 5e-324, np.full((6, 5), 2.5e-308)):
            with self.subTest(largest=float(suspect.max())):
                self.against_legacy(np.zeros((6, 5)), suspect, 3, "F64")
                self.against_legacy(np.ones((6, 5)) * 1e-310, suspect + 1e-310, 3, "F64")

    def test_zero_and_non_finite_changes(self):
        base = np.arange(12, dtype=np.float32).reshape(4, 3)
        for bad in (np.nan, np.inf, -np.inf):
            suspect = base.copy()
            suspect[1, 2] = bad
            with self.subTest(bad=bad):
                old = legacy_core.analyze_delta(base.copy(), suspect.copy(), 5)
                new = core.analyze_delta(base, suspect.copy(), 5)
                self.compare(old, new, exact=True)
                self.assertIsNone(core.rank_one_excess(base, new, "F32", "F32"))
        self.against_legacy(base, base.copy(), 5)
        self.against_legacy(base.T.copy(), base.T.copy(), 2)

    def test_mixed_precision_budgets(self):
        rng = np.random.default_rng(10)
        base = rng.standard_normal((16, 11))
        suspect = changed(rng, base, "projection")
        for base_dtype in DTYPES:
            for suspect_dtype in DTYPES:
                wide = "F64" in (base_dtype, suspect_dtype)
                a, b = (np.array(stored(x, d), np.float64 if wide else np.float32)
                        for x, d in ((base, base_dtype), (suspect, suspect_dtype)))
                with self.subTest(base=base_dtype, suspect=suspect_dtype):
                    self.against_legacy(a, b, 4, base_dtype, suspect_dtype)

    def test_strided_and_transposed_views(self):
        rng = np.random.default_rng(11)
        wide = rng.standard_normal((24, 18)).astype(np.float32)
        for name, view in (("transposed", lambda x: x.T), ("every other row", lambda x: x[::2]),
                           ("reversed", lambda x: x[::-1, ::-1]), ("fortran", np.asfortranarray)):
            for kind in ("projection", "dense"):
                suspect = changed(rng, wide, kind).astype(np.float32)
                with self.subTest(view=name, kind=kind):
                    old = legacy_core.analyze_delta(view(wide), view(suspect.copy()), 4)
                    self.compare(old, *analyze(view(wide), view(suspect.copy()), 4))

    def test_a_weak_reported_mode_keeps_the_old_solver(self):
        # Two strong modes pass the test on the energy beyond the first, but values this far below
        # the first are where the two solvers stop agreeing to the printed digits.
        rng = np.random.default_rng(15)
        for shape, k in (((64, 64), 4), ((90, 70), 6), ((30, 24), 6), ((200, 12), 32)):
            for weak in (1e-4, 1e-5, 1e-6, 1e-7):
                base, suspect = weak_tail(rng, shape, (3.0, 2.0), (3 * weak, 2 * weak, weak, weak / 2))
                with self.subTest(shape=shape, k=k, weak=weak):
                    with mock.patch.object(np.linalg, "eigvalsh", wraps=np.linalg.eigvalsh) as values_only:
                        self.assertFalse(self.against_legacy(base, suspect, k, "F64"))
                    if 4 * (k + 8) <= min(shape):  # predicted, not found out after a wasted solve
                        self.assertNotIn(min(shape), [call.args[0].shape[0] for call in values_only.call_args_list])

    def test_weak_modes_beyond_those_reported_do_not_matter(self):
        rng = np.random.default_rng(16)
        used = 0
        for shape in ((64, 64), (90, 70), (30, 24)):
            for weak in (1e-2, 1e-5, 1e-7):
                base, suspect = weak_tail(rng, shape, (3.0, 2.0, 1.0, 0.5), (weak, weak / 2, weak / 4))
                with self.subTest(shape=shape, weak=weak):
                    used += self.against_legacy(base, suspect, 4, "F64")
        self.assertEqual(used, 9)


class TestStoredMatrices(Equivalence):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def pair(self, base: dict, suspect: dict):
        write_safetensors(self.tmp / "base.safetensors", base)
        write_safetensors(self.tmp / "suspect.safetensors", suspect)
        return Checkpoint(self.tmp / "base.safetensors"), Checkpoint(self.tmp / "suspect.safetensors")

    def test_blocks_and_largest_entry_are_those_of_the_loaded_array(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal((11, 7)) * 3
        tensors = {d: (d, stored(x, d)) for d in DTYPES}
        with self.pair(tensors, tensors)[0] as ck:
            for dtype in DTYPES:
                for wide in (False, True):
                    whole = ck.to_float64(dtype) if wide else ck.to_float32(dtype)
                    m = ck.matrix(dtype, wide)
                    with self.subTest(dtype=dtype, wide=wide):
                        self.assertEqual((m.shape, m.T.shape, m.ndim), (whole.shape, whole.T.shape, 2))
                        for sl in (slice(None), slice(0, 4), slice(4, 9), slice(9, 40), slice(11, 15)):
                            for got, want in ((m.rows(sl), whole[sl]), (m.T.rows(sl), whole.T[sl])):
                                self.assertEqual(got.dtype, want.dtype)
                                np.testing.assert_array_equal(got, want)
                        got = m.rows(slice(0, 2))
                        got += 1  # a fresh copy: writable, and the file is untouched
                        np.testing.assert_array_equal(m.rows(slice(0, 2)), whole[:2])
                        self.assertEqual(m.T.largest(), float(np.abs(whole).max()))
                        self.assertEqual(m.largest(), float(np.abs(whole).max()))
                    ck.release(dtype)  # dropped pages are read again from the file
                    np.testing.assert_array_equal(ck.to_float64(dtype) if wide else ck.to_float32(dtype), whole)

    def test_largest_entry_of_a_wide_tensor_is_found_in_pieces(self):
        rng = np.random.default_rng(17)
        tensors = {"wide": ("F32", rng.standard_normal((6, 20)).astype(np.float32)),
                   "one row": ("F32", rng.standard_normal((1, 50)).astype(np.float32)),
                   "tall": ("F32", rng.standard_normal((20, 6)).astype(np.float32))}
        with self.pair(tensors, tensors)[0] as ck:
            for name, (_, x) in tensors.items():
                for turn in (lambda m: m, lambda m: m.T):
                    m = turn(ck.matrix(name))
                    with self.subTest(name=name, shape=m.shape), \
                            mock.patch.object(type(m), "_stored_rows", autospec=True, side_effect=type(m)._stored_rows) as read:
                        self.assertEqual(m.largest(block=10), float(np.abs(x).max()))
                        sizes = [read.side_effect(*call.args).size for call in read.call_args_list]
                        self.assertGreater(len(sizes), 1 if min(x.shape) > 1 else 0)
                        self.assertLessEqual(max(sizes), max(10 * min(x.shape), max(x.shape)))  # 10 rows of the long side

    def test_only_float_matrices_can_be_streamed(self):
        tensors = {"ids": ("I64", np.arange(6).reshape(2, 3)), "bias": ("F32", np.ones(4, np.float32))}
        with self.pair(tensors, tensors)[0] as ck:
            for name in tensors:
                with self.assertRaises(ValueError):
                    ck.matrix(name)

    def test_streaming_both_checkpoints_gives_the_same_bits_as_loading_them(self):
        rng = np.random.default_rng(13)
        base, suspect, kinds = {}, {}, {}
        for shape in [(23, 4), (4, 23), (9, 9)]:
            for kind in ("projection", "partial", "dense", "two modes"):
                for base_dtype, suspect_dtype in (("F32", "F32"), ("BF16", "BF16"), ("F16", "F16"), ("F64", "F64"),
                                                  ("BF16", "F32"), ("F32", "F64"), ("F64", "F16")):
                    name = f"{shape} {kind} {base_dtype} {suspect_dtype}"
                    x = rng.standard_normal(shape)
                    base[name], suspect[name] = (base_dtype, stored(x, base_dtype)), (suspect_dtype, stored(changed(rng, x, kind), suspect_dtype))
                    kinds[name] = base_dtype, suspect_dtype
        a, b = self.pair(base, suspect)
        with a, b, mock.patch.object(core, "_ROWS_PER_BLOCK", 5):
            for name, dtypes in kinds.items():
                wide = "F64" in dtypes
                load = "to_float64" if wide else "to_float32"
                for transposed in (False, True):
                    turn = (lambda m: m.T) if transposed else (lambda m: m)
                    with self.subTest(name=name, transposed=transposed):
                        loaded_base = turn(getattr(a, load)(name))
                        loaded = core.analyze_delta(loaded_base, turn(getattr(b, load)(name)), 3)
                        stored_base = turn(a.matrix(name, wide))
                        streamed = core.analyze_delta(stored_base, turn(b.matrix(name, wide)), 3)
                        self.compare(loaded, streamed, exact=True)
                        self.assertEqual(loaded.base_energy, streamed.base_energy)
                        mixed = core.analyze_delta(turn(a.matrix(name, wide)), turn(getattr(b, load)(name)), 3)
                        self.compare(loaded, mixed, exact=True)
                        for stats in (loaded, loaded._replace(base_energy=None)):
                            self.assertEqual(core.rank_one_excess(loaded_base, stats, *dtypes),
                                             core.rank_one_excess(stored_base, stats, *dtypes))

    def reference_rows(self, base, suspect, rows, k, analyze, excess_of):
        """delta.csv as the scan wrote it before: both tensors loaded whole, then the core on arrays."""
        out, directions = [], {}
        for r in rows:
            if r.status == inventory.IDENTICAL or r.base_shape.count("x") != 1:
                continue
            load = "to_float64" if "F64" in (r.base_dtype, r.suspect_dtype) else "to_float32"
            a, b = getattr(base, load)(r.name), getattr(suspect, load)(r.name)
            kind = canonical_kind(r.kind)
            token_matrix = kind in TOKEN_MATRICES and r.layer is None
            projection = token_matrix or kind in RESIDUAL_WRITERS
            if token_matrix:
                a, b = a.T, b.T
            stats = analyze(a, b, k)
            excess = excess_of(a, stats, r.base_dtype, r.suspect_dtype)
            if projection and stats.direction is not None:
                directions[r.name] = stats.direction
            row = {"name": r.name, "effective_rank": f"{float(stats.effective_rank):.6g}"}
            values = (stats.frobenius, stats.projection_gain if projection else None, stats.second_tail_share,
                      stats.column_fraction if token_matrix else None, stats.projection_error if projection else None,
                      stats.column_spike if token_matrix else None, excess, stats.tail_norm_fraction)
            row.update({f: f"{v:.8g}" for f, v in zip(delta.EXTRA_FIELDS, values) if v is not None})
            row.update({delta.sv_field(i + 1): f"{float(s):.6g}" for i, s in enumerate(stats.singular_values[:k])})
            out.append(row)
        return out, directions

    def test_scan_writes_what_the_array_pipeline_wrote(self):
        rng = np.random.default_rng(14)
        d, ff, vocab = 8, 16, 40
        layer = "model.layers.{}.{}.weight".format
        plan = [  # name, shape, kind of change, base dtype, suspect dtype
            ("model.embed_tokens.weight", (vocab, d), "dense", "BF16", "BF16"),
            ("lm_head.weight", (vocab, d), "projection", "F32", "F32"),
            (layer(0, "self_attn.o_proj"), (d, d), "projection", "F32", "F32"),
            (layer(0, "mlp.down_proj"), (d, ff), "partial", "BF16", "BF16"),
            (layer(0, "mlp.up_proj"), (ff, d), "dense", "F16", "F16"),
            (layer(1, "self_attn.o_proj"), (d, d), "projection", "F64", "F64"),
            (layer(1, "self_attn.q_proj"), (d, d), "two modes", "BF16", "F32"),
            (layer(1, "mlp.down_proj"), (d, ff), "projection", "F64", "F32"),
            (layer(1, "mlp.gate_proj"), (ff, d), "low rank", "F32", "F16"),
        ]
        base, suspect = {}, {}
        for name, shape, kind, base_dtype, suspect_dtype in plan:
            x = rng.standard_normal(shape)
            turned = name in ("model.embed_tokens.weight", "lm_head.weight")  # residual directions are the columns
            y = changed(rng, x.T, kind).T if turned else changed(rng, x, kind)
            base[name], suspect[name] = (base_dtype, stored(x, base_dtype)), (suspect_dtype, stored(y, suspect_dtype))
        base["model.norm.weight"] = suspect["model.norm.weight"] = ("F32", np.ones(d, np.float32))
        a, b = self.pair(base, suspect)
        with a, b, mock.patch.object(core, "_ROWS_PER_BLOCK", 7), mock.patch.object(legacy_core, "_ROWS_PER_BLOCK", 7):
            rows = inventory.compare(a, b)
            directions = {}
            done = delta.scan(a, b, rows, self.tmp / "delta.csv", k=4, log=lambda s: None, directions=directions)
            self.assertEqual(done, len(plan))
            with open(self.tmp / "delta.csv", newline="", encoding="utf-8") as f:
                written = [{k: v for k, v in row.items() if v != "" and k in want}
                           for row, want in ((row, set(delta.EXTRA_FIELDS) | {"name", "effective_rank"} |
                                              {delta.sv_field(i + 1) for i in range(4)}) for row in csv.DictReader(f))
                           if row["status"] != inventory.IDENTICAL]

            # Reading the checkpoints in row blocks changes nothing at all.
            same, same_directions = self.reference_rows(a, b, rows, 4, core.analyze_delta, core.rank_one_excess)
            self.assertEqual(written, same)
            self.assertEqual(directions.keys(), same_directions.keys())
            for name in directions:
                np.testing.assert_array_equal(directions[name], same_directions[name])

            # And the whole scan reports what the old core reported.
            old, old_directions = self.reference_rows(a, b, rows, 4, legacy_core.analyze_delta, legacy_core.rank_one_excess)
            self.assertEqual(directions.keys(), old_directions.keys())
            self.assertGreater(len(directions), 3)
            for name in directions:
                sign = 1.0 if directions[name] @ old_directions[name] > 0 else -1.0
                np.testing.assert_allclose(sign * directions[name], old_directions[name], rtol=0, atol=1e-10)
            for new_row, old_row in zip(written, old, strict=True):
                self.assertEqual(new_row.keys(), old_row.keys())
                self.assertEqual(new_row["name"], old_row["name"])
                for field in new_row.keys() - {"name"}:
                    with self.subTest(name=new_row["name"], field=field):
                        # the fields are printed to six or eight digits
                        self.assertLessEqual(abs(float(new_row[field]) - float(old_row[field])),
                                             2e-6 * abs(float(old_row[field])) + 1e-9 * float(old_row["sv_001"]))


if __name__ == "__main__":
    unittest.main()
