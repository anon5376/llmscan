import contextlib
import csv
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import tiny_llama, write_safetensors
from llmscan import _blas, atomic, cli, core, delta, inventory
from llmscan.safetensors_io import Checkpoint, CheckpointError


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def make_pair(tmp: Path, changed_layers: tuple[int, ...] = (0, 1, 2, 3), d: int = 16, ff: int = 32):
    rng = np.random.default_rng(7)
    base = tiny_llama(rng, layers=max(changed_layers, default=0) + 1, d=d, ff=ff, vocab=32)
    suspect = dict(base)
    u = rng.standard_normal(d, dtype=np.float32)
    u /= np.linalg.norm(u)
    for layer in changed_layers:
        for suffix in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            name = f"model.layers.{layer}.{suffix}"
            dt, w = base[name]
            suspect[name] = (dt, (w - u[:, None] * (u @ w)).astype(np.float32))
    base_path = tmp / "base.safetensors"
    suspect_path = tmp / "suspect.safetensors"
    write_safetensors(base_path, base)
    write_safetensors(suspect_path, suspect)
    return base_path, suspect_path


class TestMemoryEstimator(unittest.TestCase):
    def test_estimate_accounts_for_decoded_gram_and_block(self):
        row = inventory.InventoryRow(
            name="m.weight", layer=0, kind="o_proj", status=inventory.CHANGED,
            base_dtype="BF16", suspect_dtype="F32", base_shape="1024x2048", suspect_shape="1024x2048", numel=1024 * 2048,
        )
        # m=2048, n=1024, itemsize=4 -> decoded=16 MiB, gram=16 MiB, block=32 MiB -> 64 MiB
        self.assertEqual(delta.estimate_tensor_bytes(row), 64 * 1024 * 1024)

    def test_estimate_uses_float64_when_either_side_is_f64(self):
        row = inventory.InventoryRow(
            name="m.weight", layer=0, kind="o_proj", status=inventory.CHANGED,
            base_dtype="F32", suspect_dtype="F64", base_shape="16x8", suspect_shape="16x8", numel=128,
        )
        # m=16, n=8, itemsize=8 -> decoded=2*128*8=2048, gram=2*64*8=1024, block=2*16*8*8=2048
        self.assertEqual(delta.estimate_tensor_bytes(row), 2048 + 1024 + 2048)

    def test_max_parallelism_bounds_by_jobs_and_memory(self):
        self.assertEqual(delta._max_parallelism([100, 100, 100], jobs=1, max_memory_bytes=1000), 1)
        self.assertEqual(delta._max_parallelism([100], jobs=8, max_memory_bytes=1000), 1)
        self.assertEqual(delta._max_parallelism([500, 500, 500], jobs=8, max_memory_bytes=400), 1)
        self.assertEqual(delta._max_parallelism([100, 100, 100, 100], jobs=8, max_memory_bytes=250), 2)
        self.assertEqual(delta._max_parallelism([1000, 100, 100, 100], jobs=8, max_memory_bytes=350), 3)
        self.assertEqual(delta._max_parallelism([50, 50, 50, 50], jobs=3, max_memory_bytes=1000), 3)


class TestBlasThreadCap(unittest.TestCase):
    def test_caps_and_restores_without_raising_lower_limit(self):
        state = {"a": 8, "b": 1}

        def make_ctrl(key):
            return (lambda: state[key], lambda n: state.__setitem__(key, n))

        with mock.patch.object(_blas, "_discover_controls", return_value=[make_ctrl("a"), make_ctrl("b")]):
            cap = _blas.BlasThreadCap(workers=4)
            cap.apply()
            self.assertEqual(state, {"a": 2, "b": 1})
            cap.restore()
            self.assertEqual(state, {"a": 8, "b": 1})

    def test_workers_one_is_noop(self):
        with mock.patch.object(_blas, "_discover_controls") as discover:
            cap = _blas.BlasThreadCap(workers=1)
            cap.apply()
            cap.restore()
        discover.assert_not_called()


class TestSerialFallbacks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _assert_stays_serial(self, base_path: Path, suspect_path: Path, **scan_kwargs):
        with mock.patch("llmscan.delta.concurrent.futures.ThreadPoolExecutor") as pool_cls, \
             mock.patch.object(delta.BlasThreadCap, "apply") as blas_apply:
            with Checkpoint(base_path) as b, Checkpoint(suspect_path) as s:
                rows = inventory.compare(b, s)
                delta.scan(b, s, rows, self.tmp / "delta.csv", k=4, log=lambda _: None, **scan_kwargs)
        pool_cls.assert_not_called()
        blas_apply.assert_not_called()

    def test_default_jobs_one_leaves_serial_path_alone(self):
        base_path, suspect_path = make_pair(self.tmp, changed_layers=(0, 1))
        self._assert_stays_serial(base_path, suspect_path, jobs=1)

    def test_single_changed_matrix_leaves_serial_path_alone(self):
        rng = np.random.default_rng(2)
        base = tiny_llama(rng, layers=1, d=16, ff=32, vocab=32)
        suspect = dict(base)
        dt, w = base["model.embed_tokens.weight"]
        w2 = w.copy()
        w2[0, 0] += 0.5
        suspect["model.embed_tokens.weight"] = (dt, w2)
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)
        self._assert_stays_serial(self.tmp / "b.safetensors", self.tmp / "s.safetensors", jobs=8)

    def test_identical_checkpoints_leave_serial_path_alone(self):
        base_path, _ = make_pair(self.tmp, changed_layers=())
        self._assert_stays_serial(base_path, base_path, jobs=8)

    def test_tight_memory_budget_degrades_to_serial(self):
        base_path, suspect_path = make_pair(self.tmp, changed_layers=(0, 1))
        self._assert_stays_serial(base_path, suspect_path, jobs=8, max_memory_bytes=1)


class TestConcurrencyBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_output_parity_between_serial_and_concurrent(self):
        base_path, suspect_path = make_pair(self.tmp, changed_layers=(0, 1, 2, 3))
        out_serial = self.tmp / "out-serial"
        out_par = self.tmp / "out-par"
        c1, _, e1 = run_cli("--base", str(base_path), "--suspect", str(suspect_path), "--out", str(out_serial), "-k", "8")
        c2, _, e2 = run_cli("--base", str(base_path), "--suspect", str(suspect_path), "--out", str(out_par), "-k", "8", "-j", "4")
        self.assertEqual(c1, 0, e1)
        self.assertEqual(c2, 0, e2)
        self.assertEqual({p.name for p in out_serial.iterdir()}, {p.name for p in out_par.iterdir()})
        for p in sorted(out_serial.iterdir()):
            q = out_par / p.name
            if p.name == "delta.csv":
                with p.open(encoding="utf-8") as fa, q.open(encoding="utf-8") as fb:
                    ra, rb = list(csv.DictReader(fa)), list(csv.DictReader(fb))
                for row in ra + rb:
                    row.pop("seconds")
                self.assertEqual(ra, rb)
            else:
                self.assertEqual(p.read_bytes(), q.read_bytes(), p.name)

    def test_checkpoint_reads_stay_on_main_thread(self):
        base_path, suspect_path = make_pair(self.tmp, changed_layers=(0, 1, 2))
        main_tid = threading.get_ident()
        seen_tids = []
        orig_f32 = Checkpoint.to_float32
        orig_f64 = Checkpoint.to_float64

        def checked_f32(self_ck, name):
            seen_tids.append(threading.get_ident())
            return orig_f32(self_ck, name)

        def checked_f64(self_ck, name):
            seen_tids.append(threading.get_ident())
            return orig_f64(self_ck, name)

        with mock.patch.object(Checkpoint, "to_float32", checked_f32), \
             mock.patch.object(Checkpoint, "to_float64", checked_f64):
            with Checkpoint(base_path) as b, Checkpoint(suspect_path) as s:
                rows = inventory.compare(b, s)
                done = delta.scan(b, s, rows, self.tmp / "delta.csv", k=4, log=lambda _: None, jobs=4)
        self.assertEqual(done, 6)
        self.assertTrue(seen_tids)
        self.assertTrue(all(tid == main_tid for tid in seen_tids))

    def test_memory_budget_caps_in_flight_tensors_below_worker_count(self):
        rng = np.random.default_rng(3)
        base = {f"model.layers.{i}.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)) for i in range(6)}
        suspect = {k: (dt, w + 0.1) for k, (dt, w) in base.items()}
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)

        lock = threading.Lock()
        active = 0
        max_active = 0
        orig = core.analyze_delta

        def slow_analyze(wb, ws, k):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            try:
                return orig(wb, ws, k)
            finally:
                with lock:
                    active -= 1

        with Checkpoint(self.tmp / "b.safetensors") as b, Checkpoint(self.tmp / "s.safetensors") as s:
            rows = inventory.compare(b, s)
            one_cost = delta.estimate_tensor_bytes(rows[0])
            with mock.patch.object(core, "analyze_delta", slow_analyze):
                done = delta.scan(
                    b, s, rows, self.tmp / "delta.csv", k=4, log=lambda _: None,
                    jobs=6, max_memory_bytes=2 * one_cost,
                )
        self.assertEqual(done, 6)
        self.assertEqual(max_active, 2)

    def test_giant_tensor_runs_alone_between_small_tensors(self):
        rng = np.random.default_rng(4)
        base = {
            "model.layers.0.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
            "model.layers.1.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
            "model.layers.2.self_attn.o_proj.weight": ("F32", rng.standard_normal((128, 128)).astype(np.float32)),
            "model.layers.3.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
            "model.layers.4.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
        }
        suspect = {k: (dt, w + 0.1) for k, (dt, w) in base.items()}
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)

        lock = threading.Lock()
        active = 0
        giant_saw_peers = False
        orig = core.analyze_delta

        def tracked(wb, ws, k):
            nonlocal active, giant_saw_peers
            is_giant = wb.shape == (128, 128)
            with lock:
                active += 1
                if is_giant and active > 1:
                    giant_saw_peers = True
            time.sleep(0.03)
            with lock:
                if is_giant and active > 1:
                    giant_saw_peers = True
                active -= 1
            return orig(wb, ws, k)

        with Checkpoint(self.tmp / "b.safetensors") as b, Checkpoint(self.tmp / "s.safetensors") as s:
            rows = inventory.compare(b, s)
            small_cost = delta.estimate_tensor_bytes(rows[0])
            giant_cost = delta.estimate_tensor_bytes(rows[2])
            budget = 2 * small_cost
            self.assertGreater(giant_cost, budget)
            with mock.patch.object(core, "analyze_delta", tracked):
                done = delta.scan(
                    b, s, rows, self.tmp / "delta.csv", k=4, log=lambda _: None,
                    jobs=4, max_memory_bytes=budget,
                )
        self.assertEqual(done, 5)
        self.assertFalse(giant_saw_peers)

    def test_ordered_incremental_writes_when_later_tensors_finish_first(self):
        rng = np.random.default_rng(5)
        base = {
            "model.layers.0.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
            "model.layers.1.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
            "model.layers.2.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
            "model.layers.3.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)),
        }
        suspect = dict(base)
        for i in (0, 2, 3):
            name = f"model.layers.{i}.self_attn.o_proj.weight"
            dt, w = base[name]
            w2 = w.copy()
            w2[0, 0] += float(i + 1)
            suspect[name] = (dt, w2)
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)

        others_done = threading.Event()
        finished_later = 0
        lock = threading.Lock()
        orig = core.analyze_delta

        def staggered(wb, ws, k):
            nonlocal finished_later
            diff = float(ws[0, 0] - wb[0, 0])
            if abs(diff - 1.0) < 0.1:  # layer 0 waits until layers 2 and 3 finish
                others_done.wait(timeout=5.0)
                return orig(wb, ws, k)
            res = orig(wb, ws, k)
            with lock:
                finished_later += 1
                if finished_later == 2:
                    others_done.set()
            return res

        logs = []
        with Checkpoint(self.tmp / "b.safetensors") as b, Checkpoint(self.tmp / "s.safetensors") as s:
            rows = inventory.compare(b, s)
            with mock.patch.object(core, "analyze_delta", staggered):
                delta.scan(b, s, rows, self.tmp / "delta.csv", k=4, log=logs.append, jobs=3)

        with (self.tmp / "delta.csv").open(encoding="utf-8") as f:
            names = [r["name"] for r in csv.DictReader(f)]
        self.assertEqual(names, [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.2.self_attn.o_proj.weight",
            "model.layers.3.self_attn.o_proj.weight",
        ])
        self.assertTrue(logs[0].startswith("[1/3] model.layers.0."))
        self.assertTrue(logs[1].startswith("[2/3] model.layers.2."))
        self.assertTrue(logs[2].startswith("[3/3] model.layers.3."))


class TestInterruptionAndErrors(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_keyboard_interrupt_keeps_completed_ordered_prefix(self):
        base_path, suspect_path = make_pair(self.tmp, changed_layers=(0, 1, 2, 3))
        out_csv = self.tmp / "delta.csv"
        logged = []

        def interrupt_on_second(msg: str):
            logged.append(msg)
            if len(logged) == 2:
                raise KeyboardInterrupt()

        with Checkpoint(base_path) as b, Checkpoint(suspect_path) as s:
            rows = inventory.compare(b, s)
            with self.assertRaises(KeyboardInterrupt):
                atomic.write(
                    out_csv,
                    lambda tmp: delta.scan(b, s, rows, tmp, k=4, log=interrupt_on_second, jobs=2),
                    keep_partial=True,
                )

        self.assertTrue(out_csv.exists())
        with out_csv.open(encoding="utf-8") as f:
            written = list(csv.DictReader(f))
        changed_written = [r for r in written if r["status"] != inventory.IDENTICAL]
        # The first changed tensor was written before interrupt_on_second raised on the second
        self.assertEqual(len(changed_written), 1)
        expected_first = next(r.name for r in rows if r.status != inventory.IDENTICAL and r.base_shape.count("x") == 1)
        self.assertEqual(changed_written[0]["name"], expected_first)

    def test_keyboard_interrupt_during_wait_cancels_queued_and_preserves_prefix(self):
        rng = np.random.default_rng(11)
        base = {f"model.layers.{i}.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)) for i in range(6)}
        suspect = {k: (dt, w + 0.5) for k, (dt, w) in base.items()}
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)

        release = threading.Event()
        started = []
        lock = threading.Lock()
        orig = core.analyze_delta

        def slow(wb, ws, k):
            with lock:
                started.append(1)
            release.wait(timeout=2.0)
            return orig(wb, ws, k)

        orig_wait = delta.concurrent.futures.wait
        calls = 0

        def interrupt_wait(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls >= 1:
                raise KeyboardInterrupt()
            return orig_wait(*args, **kwargs)

        out_csv = self.tmp / "delta.csv"
        try:
            with Checkpoint(self.tmp / "b.safetensors") as b, Checkpoint(self.tmp / "s.safetensors") as s:
                rows = inventory.compare(b, s)
                with mock.patch.object(core, "analyze_delta", slow), \
                     mock.patch.object(delta.concurrent.futures, "wait", interrupt_wait):
                    with self.assertRaises(KeyboardInterrupt):
                        atomic.write(
                            out_csv,
                            lambda tmp: delta.scan(b, s, rows, tmp, k=4, log=lambda _: None, jobs=2),
                            keep_partial=True,
                        )
            # Only the 2 initially admitted tensors should have been submitted, not all 6
            self.assertLessEqual(len(started), 2)
            self.assertTrue(out_csv.exists())
        finally:
            release.set()

    def test_cli_stops_cleanly_on_not_implemented_with_jobs(self):
        base_path, suspect_path = make_pair(self.tmp, changed_layers=(0, 1))
        out = self.tmp / "out"
        with mock.patch.object(core, "analyze_delta", side_effect=NotImplementedError("stub")):
            code, _, err = run_cli("--base", str(base_path), "--suspect", str(suspect_path), "--out", str(out), "-j", "4")
        self.assertEqual(code, 2)
        self.assertIn("stub", err)
        self.assertTrue((out / "inventory.csv").exists())
        self.assertTrue((out / "ranking.csv").exists())

    def test_error_in_later_tensor_flushes_earlier_completed_tensor_first(self):
        rng = np.random.default_rng(6)
        base = {f"model.layers.{i}.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)) for i in range(3)}
        suspect = {k: (dt, w + float(i + 1)) for i, (k, (dt, w)) in enumerate(base.items())}
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)

        second_failed = threading.Event()
        orig = core.analyze_delta

        def fail_second(wb, ws, k):
            diff = float(ws[0, 0] - wb[0, 0])
            if abs(diff - 2.0) < 0.1:
                second_failed.set()
                raise RuntimeError("boom on layer 1")
            if abs(diff - 1.0) < 0.1:
                second_failed.wait(timeout=5.0)
            return orig(wb, ws, k)

        out_csv = self.tmp / "delta.csv"
        with Checkpoint(self.tmp / "b.safetensors") as b, Checkpoint(self.tmp / "s.safetensors") as s:
            rows = inventory.compare(b, s)
            with mock.patch.object(core, "analyze_delta", fail_second):
                with self.assertRaisesRegex(RuntimeError, "boom on layer 1"):
                    delta.scan(b, s, rows, out_csv, k=4, log=lambda _: None, jobs=2)

        with out_csv.open(encoding="utf-8") as f:
            names = [r["name"] for r in csv.DictReader(f)]
        self.assertEqual(names, ["model.layers.0.self_attn.o_proj.weight"])

    def test_load_error_flushes_earlier_tensor_before_raising(self):
        rng = np.random.default_rng(9)
        base = {f"model.layers.{i}.self_attn.o_proj.weight": ("F32", rng.standard_normal((16, 16)).astype(np.float32)) for i in range(3)}
        suspect = {k: (dt, w + 0.25) for k, (dt, w) in base.items()}
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)

        orig_load = Checkpoint.to_float32

        def fail_layer_1(self_ck, name):
            if "layers.1." in name:
                raise CheckpointError("corrupt layer 1")
            return orig_load(self_ck, name)

        out_csv = self.tmp / "delta.csv"
        with Checkpoint(self.tmp / "b.safetensors") as b, Checkpoint(self.tmp / "s.safetensors") as s:
            rows = inventory.compare(b, s)
            with mock.patch.object(Checkpoint, "to_float32", fail_layer_1):
                with self.assertRaisesRegex(CheckpointError, "corrupt layer 1"):
                    delta.scan(b, s, rows, out_csv, k=4, log=lambda _: None, jobs=2)

        with out_csv.open(encoding="utf-8") as f:
            names = [r["name"] for r in csv.DictReader(f)]
        self.assertEqual(names, ["model.layers.0.self_attn.o_proj.weight"])


class TestCliValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_invalid_jobs_and_max_memory_preserve_previous_scan(self):
        base_path, suspect_path = make_pair(self.tmp, changed_layers=(0,))
        out = self.tmp / "out"
        base_args = ["--base", str(base_path), "--suspect", str(suspect_path), "--out", str(out)]
        self.assertEqual(run_cli(*base_args, "-k", "2")[0], 0)
        before = {p.name: p.read_bytes() for p in out.iterdir()}

        for bad in (
            ["--jobs", "0"],
            ["--jobs", "-3"],
            ["--max-memory", "0"],
            ["--max-memory", "-16"],
            ["--max-memory", "nan"],
            ["--max-memory", "inf"],
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit) as cm:
                    run_cli(*base_args, *bad)
                self.assertEqual(cm.exception.code, 2)
                self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, before)

    def test_help_is_ascii_encodable(self):
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
