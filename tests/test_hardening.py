import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from itertools import permutations
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import write_safetensors, write_sharded
from llmscan import atomic, cli, core, inventory
from llmscan.safetensors_io import MAX_OPEN_MAPS, VALUE_CHUNK, Checkpoint, CheckpointError
from test_pipeline import fake_analyze, run_cli

ROOT = Path(__file__).resolve().parent.parent


def write_raw(path: Path, header: dict | bytes, payload: bytes) -> None:
    """A safetensors file with exactly the header and payload given, valid or not."""
    h = header if isinstance(header, bytes) else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + payload)


def entry(dtype: str, shape: list, begin: int, end: int) -> dict:
    return {"dtype": dtype, "shape": shape, "data_offsets": [begin, end]}


def link_or_skip(make, *args, **kw) -> None:
    """Symlinks and hard links need a privilege or filesystem support that some systems lack."""
    try:
        make(*args, **kw)
    except FileExistsError:
        raise
    except (OSError, NotImplementedError) as e:
        raise unittest.SkipTest(f"cannot create the link here: {e}")


def compare_one(tmp: Path, base: tuple[str, np.ndarray], suspect: tuple[str, np.ndarray]) -> str:
    write_safetensors(tmp / "a.safetensors", {"x": base})
    write_safetensors(tmp / "b.safetensors", {"x": suspect})
    with Checkpoint(tmp / "a.safetensors") as a, Checkpoint(tmp / "b.safetensors") as b:
        return inventory.compare(a, b)[0].status


class TestCrossDtypeComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_f64_difference_below_float32_resolution_is_changed(self):
        status = compare_one(self.tmp, ("F64", np.array([1 + 2.0**-30])), ("F32", np.array([1.0])))
        self.assertEqual(status, inventory.CHANGED)

    def test_f64_value_that_overflows_float32_is_changed(self):
        status = compare_one(self.tmp, ("F64", np.array([1e300])), ("F32", np.array([np.inf])))
        self.assertEqual(status, inventory.CHANGED)

    def test_mixed_float_comparison_across_chunks(self):
        # All four formats represent these values exactly.
        a = np.resize(np.array([0.5, -1.25, 3.0, 2.0**-20]), 4 * VALUE_CHUNK + 3)
        for edit in (None, 0, VALUE_CHUNK - 1, VALUE_CHUNK, 2 * VALUE_CHUNK, a.size - 1):
            b = a.copy()
            if edit is not None:
                b[edit] = -2
            expected = inventory.IDENTICAL if edit is None else inventory.CHANGED
            for left, right in permutations(("BF16", "F16", "F32", "F64"), 2):
                with self.subTest(left=left, right=right, edit=edit):
                    self.assertEqual(compare_one(self.tmp, (left, a), (right, b)), expected)

    def test_difference_in_a_late_chunk_is_found(self):
        a = np.zeros(5000, dtype=np.float32)
        b = a.copy()
        b[-1] = 1
        write_safetensors(self.tmp / "a.safetensors", {"x": ("BF16", a)})
        write_safetensors(self.tmp / "b.safetensors", {"x": ("F32", b)})
        with Checkpoint(self.tmp / "a.safetensors") as ca, Checkpoint(self.tmp / "b.safetensors") as cb:
            self.assertFalse(inventory.values_equal(ca, cb, "x"))
            self.assertEqual([len(c) for c in ca.iter_values("x", chunk=2048)], [2048, 2048, 904])
            self.assertEqual(sum(c.sum() for c in cb.iter_values("x", chunk=2048)), 1)

    def test_value_comparison_memory_is_bounded_by_the_chunk_not_the_tensor(self):
        n = 2_000_000  # whole-tensor float32 decoding of both sides would allocate ~9 bytes per element
        a = np.zeros(n, dtype=np.float32)
        write_safetensors(self.tmp / "a.safetensors", {"x": ("BF16", a)})
        write_safetensors(self.tmp / "b.safetensors", {"x": ("F32", a)})
        with Checkpoint(self.tmp / "a.safetensors") as ca, Checkpoint(self.tmp / "b.safetensors") as cb:
            tracemalloc.start()
            try:
                self.assertTrue(inventory.values_equal(ca, cb, "x"))
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        self.assertLess(peak, 4 * n)  # less than ONE float32 copy of the tensor; whole-tensor decoding needs ~9n

    def test_same_dtype_detects_signed_zero_in_partial_final_chunk(self):
        a = np.zeros(32_769, dtype=np.float32)  # two 64 KiB chunks plus one value
        b = a.copy()
        b[-1] = -0.0
        self.assertEqual(compare_one(self.tmp, ("F32", a), ("F32", b)), inventory.CHANGED)

    def test_same_dtype_byte_comparison_memory_is_bounded_by_the_chunk_not_the_tensor(self):
        n = 8_000_000  # 32 MB per side; reading each tensor whole would allocate at least 32 MB
        a = np.zeros(n, dtype=np.float32)
        write_safetensors(self.tmp / "a.safetensors", {"x": ("F32", a)})
        write_safetensors(self.tmp / "b.safetensors", {"x": ("F32", a)})
        with Checkpoint(self.tmp / "a.safetensors") as ca, Checkpoint(self.tmp / "b.safetensors") as cb:
            tracemalloc.start()
            try:
                self.assertTrue(inventory.bytes_equal(ca, cb, "x"))
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        self.assertLess(peak, 1 << 20)


class TestBoundedOpenMaps(unittest.TestCase):
    def test_touching_many_shards_keeps_few_maps_and_file_descriptors(self):
        tmp = Path(tempfile.mkdtemp())
        shards = [{f"t{i}": ("F32", np.full((2, 2), i, np.float32))} for i in range(40)]
        write_sharded(tmp / "a", shards)
        write_sharded(tmp / "b", shards)
        fds = Path("/dev/fd")
        before = len(os.listdir(fds)) if fds.is_dir() else None
        with Checkpoint(tmp / "a") as a, Checkpoint(tmp / "b") as b:
            rows = inventory.compare(a, b)
            self.assertEqual({r.status for r in rows}, {inventory.IDENTICAL})
            self.assertLessEqual(len(a._maps), MAX_OPEN_MAPS)
            np.testing.assert_array_equal(a.to_float32("t3"), 3)  # a shard evicted earlier maps again
            if before is not None:
                self.assertLessEqual(len(os.listdir(fds)) - before, 2 * MAX_OPEN_MAPS)
        self.assertEqual(len(a._maps), 0)


class TestMalformedHeaders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "m.safetensors"
        self.f4 = np.ones(2, np.float32).tobytes()  # two float32 values: 8 bytes

    def assert_rejected(self, header, payload, pattern):
        write_raw(self.path, header, payload)
        with self.assertRaisesRegex(CheckpointError, pattern):
            Checkpoint(self.path)

    def test_valid_layout_is_accepted(self):
        write_raw(self.path, {"__metadata__": {"format": "pt"}, "a": entry("F32", [2], 0, 8), "b": entry("F32", [], 8, 12),
                              "e": entry("F32", [0], 12, 12)}, self.f4 + bytes(4))
        with Checkpoint(self.path) as c:
            self.assertEqual(sorted(c.tensors), ["a", "b", "e"])

    def test_negative_offsets(self):
        self.assert_rejected({"x": entry("F32", [1], -4, 0)}, bytes(4), "data_offsets")
        self.assert_rejected({"x": entry("F32", [1], 0, -4)}, bytes(4), "data_offsets")

    def test_overlapping_tensors(self):
        self.assert_rejected({"a": entry("F32", [2], 0, 8), "b": entry("F32", [2], 0, 8)}, self.f4, "overlaps")
        self.assert_rejected({"a": entry("F32", [2], 0, 8), "b": entry("F32", [2], 4, 12)}, self.f4 + bytes(4), "overlaps")

    def test_gap_between_tensors(self):
        self.assert_rejected({"a": entry("F32", [1], 0, 4), "b": entry("F32", [1], 8, 12)}, bytes(12), "gap")

    def test_negative_and_non_integer_dims(self):
        self.assert_rejected({"x": entry("F32", [-1, -1], 0, 4)}, bytes(4), "shape")
        self.assert_rejected({"x": entry("F32", [1.5], 0, 4)}, bytes(4), "shape")
        self.assert_rejected({"x": entry("F32", "2", 0, 8)}, self.f4, "shape")

    def test_duplicate_tensor_names_in_the_json(self):
        raw = b'{"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "x": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}'
        self.assert_rejected(raw, self.f4, "duplicate")

    def test_unknown_dtype(self):
        self.assert_rejected({"x": entry("F7", [1], 0, 3)}, bytes(3), "unknown dtype")
        self.assert_rejected({"x": {"shape": [1], "data_offsets": [0, 4]}}, bytes(4), "unknown dtype")

    def test_byte_size_must_match_dtype_and_shape(self):
        self.assert_rejected({"x": entry("F32", [3], 0, 8)}, self.f4, "expected 12")

    def test_trailing_payload_the_header_does_not_describe(self):
        self.assert_rejected({"x": entry("F32", [1], 0, 4)}, self.f4, "not described")

    def test_header_that_is_not_json_or_not_an_object(self):
        self.assert_rejected(b"{not json", b"", "invalid safetensors header")
        self.assert_rejected(b"\xff\xfe", b"", "invalid safetensors header")
        self.assert_rejected(b"[]", b"", "not a JSON object")
        self.assert_rejected({"x": 5}, b"", "no metadata")

    def test_dtype_and_metadata_of_the_wrong_type(self):
        for bad in ([], {}, None, 5):
            self.assert_rejected({"x": {"dtype": bad, "shape": [1], "data_offsets": [0, 4]}}, bytes(4), "unknown dtype")
        self.assert_rejected({"__metadata__": [], "x": entry("F32", [1], 0, 4)}, bytes(4), "__metadata__ is not a JSON object")
        self.assert_rejected({"__metadata__": "x", "x": entry("F32", [1], 0, 4)}, bytes(4), "__metadata__ is not a JSON object")

    def test_cli_reports_a_malformed_file_as_a_clean_error(self):
        write_raw(self.path, {"x": entry("F32", [1], -4, 0)}, bytes(4))
        with self.assertRaises(SystemExit) as cm:
            run_cli("--base", str(self.path), "--suspect", str(self.path), "--out", str(self.tmp / "out"))
        self.assertIn("data_offsets", str(cm.exception))


@unittest.skipUnless(shutil.which("bash") and sys.platform != "win32", "needs bash and POSIX file modes")
class TestFetchScript(unittest.TestCase):
    """scripts/fetch_hf.sh against a stand-in `curl`; no network."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.dest = self.tmp / "model"
        (self.tmp / "bin").mkdir()
        (self.tmp / "curl.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "a = sys.argv[1:]\n"
            "if any('/api/models/' in x for x in a):\n"
            "    print(json.dumps({'siblings': [{'size': 4, 'rfilename': 'model.safetensors'}]}))\n"
            "else:\n"
            "    Path(a[a.index('-o') + 1]).write_bytes(b'BBBB')\n")
        fake = self.tmp / "bin" / "curl"
        fake.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{self.tmp / "curl.py"}" "$@"\n')
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    def fetch(self, rev: str):
        env = {**os.environ, "PATH": f"{self.tmp / 'bin'}{os.pathsep}{os.environ['PATH']}", "HF_HOME": str(self.tmp / "hf"), "HF_TOKEN": ""}
        return subprocess.run(["bash", str(ROOT / "scripts" / "fetch_hf.sh"), "org/model", rev, str(self.dest)],
                              env=env, capture_output=True, text=True)

    def test_same_size_file_from_another_revision_is_not_relabelled(self):
        self.dest.mkdir()
        (self.dest / "model.safetensors").write_bytes(b"AAAA")
        (self.dest / ".source").write_text("org/model@old-revision\n")
        done = self.fetch("new-revision")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("old-revision", done.stderr)
        self.assertEqual((self.dest / "model.safetensors").read_bytes(), b"AAAA")
        self.assertEqual((self.dest / ".source").read_text().strip(), "org/model@old-revision")

    def test_interrupted_fetch_of_another_revision_is_not_mixed_in(self):
        self.dest.mkdir()
        (self.dest / ".source.partial").write_text("org/model@old-revision\n")
        self.assertNotEqual(self.fetch("new-revision").returncode, 0)

    def test_unlabelled_non_empty_directory_is_refused(self):
        self.dest.mkdir()
        (self.dest / "model.safetensors").write_bytes(b"AAAA")
        done = self.fetch("rev1")
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual((self.dest / "model.safetensors").read_bytes(), b"AAAA")

    def test_fresh_and_repeated_fetches_of_the_same_revision_work(self):
        self.assertEqual(self.fetch("rev1").returncode, 0)
        self.assertEqual((self.dest / ".source").read_text().strip(), "org/model@rev1")
        self.assertFalse((self.dest / ".source.partial").exists())
        again = self.fetch("rev1")
        self.assertEqual(again.returncode, 0)
        self.assertIn("(complete)", again.stdout)


class TestCliOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cwd = os.getcwd()
        os.chdir(self.tmp)  # default output goes to ./out
        self.addCleanup(os.chdir, self.cwd)
        for d in ("a", "b"):
            (self.tmp / d).mkdir()
            write_safetensors(self.tmp / d / "model.safetensors", {"w": ("F32", np.full((4, 4), 1 if d == "a" else 2, np.float32))})

    def test_spectra_is_reported_only_when_written_and_never_deleted(self):
        out = self.tmp / "out"
        base, other = str(self.tmp / "a" / "model.safetensors"), str(self.tmp / "b" / "model.safetensors")
        with mock.patch.object(core, "analyze_delta", fake_analyze):
            fake_analyze.calls = []
            code, stdout, _ = run_cli("--base", base, "--suspect", other, "--out", str(out), "-k", "2")
            self.assertEqual(code, 0)
            self.assertIn("spectra.png", stdout)
            earlier = (out / "spectra.png").read_bytes()
            write_safetensors(self.tmp / "b" / "model.safetensors", {"w": ("F32", np.full((4, 4), 1, np.float32))})  # now equal to a
            code, stdout, err = run_cli("--base", base, "--suspect", other, "--out", str(out), "-k", "2")
        self.assertEqual(code, 0)
        self.assertNotIn("spectra.png", stdout)  # not written this run
        self.assertIn("left over", err)
        self.assertEqual((out / "spectra.png").read_bytes(), earlier)  # and not deleted either

    def test_pair_txt_symlinked_to_a_model_is_not_written_through(self):
        out = self.tmp / "o"
        out.mkdir()
        model = self.tmp / "a" / "model.safetensors"
        before = model.read_bytes()
        link_or_skip((out / "pair.txt").symlink_to, model)
        with self.assertRaises(SystemExit) as cm:
            run_cli("--base", str(model), "--suspect", str(model), "--out", str(out), "--inventory-only")
        self.assertIn("symbolic link", str(cm.exception))
        self.assertEqual(model.read_bytes(), before)

    def test_a_model_directory_is_not_an_output_directory(self):
        model = self.tmp / "a" / "model.safetensors"
        other = self.tmp / "b" / "model.safetensors"
        (self.tmp / "a" / "spectra.png").write_bytes(b"someone else's figure")
        (self.tmp / "a" / "child").mkdir()
        link_or_skip((self.tmp / "link").symlink_to, self.tmp / "a", target_is_directory=True)
        for out in (self.tmp / "a", self.tmp / "a" / "child" / "..", self.tmp / "link"):
            with self.subTest(out=str(out)), self.assertRaises(SystemExit) as cm:
                run_cli("--base", str(model), "--suspect", str(other), "--out", str(out))
            self.assertIn("no pair.txt", str(cm.exception))
        self.assertEqual((self.tmp / "a" / "spectra.png").read_bytes(), b"someone else's figure")
        self.assertFalse((self.tmp / "a" / "pair.txt").exists())
        self.assertFalse((self.tmp / "a" / "inventory.csv").exists())

    def test_default_directory_from_before_the_pair_marker_is_not_overwritten(self):
        a, b = str(self.tmp / "a" / "model.safetensors"), str(self.tmp / "b" / "model.safetensors")
        old = self.tmp / "out" / "model.safetensors-vs-model.safetensors"
        old.mkdir(parents=True)
        (old / "inventory.csv").write_text("old report\n")
        with self.assertRaises(SystemExit) as cm:
            run_cli("--base", a, "--suspect", b, "--inventory-only")
        self.assertIn("no pair.txt", str(cm.exception))
        self.assertEqual((old / "inventory.csv").read_text(), "old report\n")

    def test_unrelated_runs_on_same_named_files_do_not_share_the_default_directory(self):
        a, b = str(self.tmp / "a" / "model.safetensors"), str(self.tmp / "b" / "model.safetensors")
        self.assertEqual(run_cli("--base", a, "--suspect", a, "--inventory-only")[0], 0)
        first = (self.tmp / "out" / "model.safetensors-vs-model.safetensors" / "inventory.csv").read_text()
        with self.assertRaises(SystemExit) as cm:
            run_cli("--base", b, "--suspect", b, "--inventory-only")
        self.assertIn("different comparison", str(cm.exception))
        self.assertEqual((self.tmp / "out" / "model.safetensors-vs-model.safetensors" / "inventory.csv").read_text(), first)
        self.assertEqual(run_cli("--base", a, "--suspect", a, "--inventory-only")[0], 0)  # rerun of the same pair
        self.assertEqual(run_cli("--base", b, "--suspect", b, "--inventory-only", "--out", "elsewhere")[0], 0)

    def test_a_link_that_appears_between_the_checks_and_the_writes_is_replaced_not_followed(self):
        out, model = self.tmp / "o", self.tmp / "a" / "model.safetensors"
        other = self.tmp / "b" / "model.safetensors"
        victims = {name: self.tmp / f"victim-{name}" for name in ("pair.txt", "inventory.csv", "inventory.txt",
                                                                  "delta.csv", "spectra.png", "effective_rank.png")}
        for v in victims.values():
            v.write_bytes(b"a checkpoint")

        def plant(*names):
            for name in names:
                (out / name).unlink(missing_ok=True)  # the intruder takes the name over
                link_or_skip((out / name).symlink_to, victims[name])

        real_mkdir, real_analyze = Path.mkdir, fake_analyze
        def mkdir_then_plant(self_, *a, **kw):  # after both guards have passed, before the first write
            real_mkdir(self_, *a, **kw)
            if self_ == out:
                plant("pair.txt", "inventory.csv", "inventory.txt")
        def analyze_then_plant(*a):  # while the scan runs, before delta.csv and the figures are written
            plant("delta.csv", "spectra.png", "effective_rank.png")
            return real_analyze(*a)
        fake_analyze.calls = []
        with mock.patch.object(Path, "mkdir", mkdir_then_plant), mock.patch.object(core, "analyze_delta", analyze_then_plant):
            code, _, _ = run_cli("--base", str(model), "--suspect", str(other), "--out", str(out), "-k", "2")
        self.assertEqual(code, 0)
        for name, v in victims.items():
            with self.subTest(name=name):
                self.assertEqual(v.read_bytes(), b"a checkpoint")
                self.assertFalse((out / name).is_symlink())
                self.assertTrue((out / name).is_file())

    def test_hard_linked_pair_txt_is_not_ours_and_is_not_truncated(self):
        out, model = self.tmp / "o", self.tmp / "a" / "model.safetensors"
        out.mkdir()
        link_or_skip(os.link, model, out / "pair.txt")
        before = model.read_bytes()
        for extra in ([], ["--inventory-only"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as cm:
                run_cli("--base", str(model), "--suspect", str(model), "--out", str(out), *extra)
            self.assertIn("hard links", str(cm.exception))
        self.assertEqual(model.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["pair.txt"])

    def test_hard_linked_report_file_is_replaced_not_truncated(self):
        out, model = self.tmp / "o", self.tmp / "a" / "model.safetensors"
        self.assertEqual(run_cli("--base", str(model), "--suspect", str(model), "--out", str(out), "--inventory-only")[0], 0)
        (out / "inventory.csv").unlink()
        victim = self.tmp / "victim.csv"
        victim.write_bytes(b"a checkpoint")
        link_or_skip(os.link, victim, out / "inventory.csv")
        self.assertEqual(run_cli("--base", str(model), "--suspect", str(model), "--out", str(out), "--inventory-only")[0], 0)
        self.assertEqual(victim.read_bytes(), b"a checkpoint")
        self.assertIn("name,", (out / "inventory.csv").read_text())

    def test_explicit_out_marked_for_another_pair_is_refused(self):
        a, b = str(self.tmp / "a" / "model.safetensors"), str(self.tmp / "b" / "model.safetensors")
        out = self.tmp / "o"
        self.assertEqual(run_cli("--base", a, "--suspect", a, "--out", str(out), "--inventory-only")[0], 0)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        with self.assertRaises(SystemExit) as cm:
            run_cli("--base", b, "--suspect", b, "--out", str(out), "--inventory-only")
        self.assertIn("different comparison", str(cm.exception))
        self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, before)
        self.assertEqual(run_cli("--base", a, "--suspect", a, "--out", str(out), "--inventory-only")[0], 0)  # same pair: fine

    def test_marker_that_is_not_text_is_refused_not_a_crash(self):
        a = str(self.tmp / "a" / "model.safetensors")
        out = self.tmp / "o"
        out.mkdir()
        (out / "pair.txt").write_bytes(b"\xff\xfe not utf-8")
        with self.assertRaises(SystemExit) as cm:
            run_cli("--base", a, "--suspect", a, "--out", str(out), "--inventory-only")
        self.assertIn("different comparison", str(cm.exception))


@unittest.skipIf(os.name == "nt", "POSIX mode bits")
class TestSharedOutputLocation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for d in ("a", "b"):
            (self.tmp / d).mkdir()
            write_safetensors(self.tmp / d / "model.safetensors", {"w": ("F32", np.full((4, 4), 1 if d == "a" else 2, np.float32))})
        self.args = ("--base", str(self.tmp / "a"), "--suspect", str(self.tmp / "b"), "--inventory-only")

    def dir_with_mode(self, name: str, mode: int) -> Path:
        d = self.tmp / name
        d.mkdir()
        d.chmod(mode)  # mkdir's mode is masked by the umask
        return d

    def test_refused_when_out_is_writable_by_others_without_sticky_bit(self):
        for mode in (0o777, 0o775, 0o757):
            out = self.dir_with_mode(f"o{mode:o}", mode)
            with self.subTest(mode=oct(mode)), self.assertRaises(SystemExit) as cm:
                run_cli(*self.args, "--out", str(out))
            self.assertIn(str(out.resolve()), str(cm.exception))
            self.assertIn("sticky bit", str(cm.exception))
            self.assertEqual(os.listdir(out), [])  # nothing created, not even a .llmscan-* directory

    def test_refused_for_an_unsafe_ancestor_even_if_out_does_not_exist_yet(self):
        shared = self.dir_with_mode("shared", 0o777)
        for out in (shared / "new", shared / "new" / "deeper"):
            with self.subTest(out=str(out.relative_to(self.tmp))), self.assertRaises(SystemExit) as cm:
                run_cli(*self.args, "--out", str(out))
            self.assertIn(str(shared.resolve()), str(cm.exception))
            self.assertEqual(os.listdir(shared), [])
        (shared / "existing").mkdir(mode=0o700)  # a private directory inside a shared one is still reachable by rename
        with self.assertRaises(SystemExit):
            run_cli(*self.args, "--out", str(shared / "existing"))
        self.assertEqual(os.listdir(shared / "existing"), [])

    def test_refused_through_a_symlink_to_an_unsafe_directory(self):
        shared = self.dir_with_mode("shared", 0o777)
        link_or_skip((self.tmp / "link").symlink_to, shared, target_is_directory=True)
        with self.assertRaises(SystemExit) as cm:
            run_cli(*self.args, "--out", str(self.tmp / "link" / "o"))
        self.assertIn(str(shared.resolve()), str(cm.exception))
        self.assertEqual(os.listdir(shared), [])

    def test_accepted_when_sticky_or_only_the_owner_can_write(self):
        for name, mode in (("sticky", 0o1777), ("sticky_group", 0o1775), ("normal", 0o755), ("private", 0o700)):
            out = self.dir_with_mode(name, mode)
            with self.subTest(mode=oct(mode)):
                self.assertEqual(run_cli(*self.args, "--out", str(out))[0], 0)
                self.assertTrue((out / "inventory.csv").is_file())

    def test_accepted_below_a_sticky_world_writable_directory(self):
        shared = self.dir_with_mode("tmpish", 0o1777)
        self.assertEqual(run_cli(*self.args, "--out", str(shared / "new" / "deeper"))[0], 0)
        self.assertTrue((shared / "new" / "deeper" / "inventory.csv").is_file())

    def test_writes_go_through_the_checked_path_not_a_swappable_alias(self):
        shared = self.dir_with_mode("shared", 0o777)
        safe, victim = self.dir_with_mode("safe", 0o700), self.dir_with_mode("victim", 0o700)
        alias = shared / "alias"
        link_or_skip(alias.symlink_to, safe, target_is_directory=True)
        real_open = cli.open_checkpoint

        def swap_then_open(path):  # another user renames the alias after the guard passed
            alias.unlink()
            alias.symlink_to(victim, target_is_directory=True)
            return real_open(path)

        with mock.patch.object(cli, "open_checkpoint", swap_then_open):
            self.assertEqual(run_cli(*self.args, "--out", str(alias))[0], 0)
        self.assertEqual(os.listdir(victim), [])
        self.assertTrue((safe / "inventory.csv").is_file())

    def test_new_output_directories_are_private_under_a_permissive_umask(self):
        out = self.tmp / "new" / "deeper"
        old = os.umask(0)
        try:
            self.assertEqual(run_cli(*self.args, "--out", str(out))[0], 0)
        finally:
            os.umask(old)
        for d in (out, out.parent):
            self.assertEqual(stat.S_IMODE(d.stat().st_mode), 0o700)

    def test_refused_when_a_directory_it_created_is_not_safe(self):
        real_mkdir = os.mkdir

        def mkdir_then_open_up(path, mode=0o777):  # as if an inherited ACL or default widened it
            real_mkdir(path, mode)
            os.chmod(path, 0o777)

        with mock.patch.object(cli.os, "mkdir", mkdir_then_open_up), self.assertRaises(SystemExit) as cm:
            run_cli(*self.args, "--out", str(self.tmp / "new"))
        self.assertIn("sticky bit", str(cm.exception))
        self.assertEqual(os.listdir(self.tmp / "new"), [])

    def test_sticky_directory_owned_by_another_user_is_refused(self):
        out = self.dir_with_mode("stolen", 0o1777)
        with mock.patch.object(cli.os, "geteuid", return_value=os.geteuid() + 1):
            with self.assertRaises(SystemExit) as cm:
                run_cli(*self.args, "--out", str(out))
            self.assertIn("owned by another user", str(cm.exception))
            self.assertEqual(os.listdir(out), [])
            plain = self.dir_with_mode("plain", 0o755)  # not writable by others: owner does not matter
            self.assertEqual(run_cli(*self.args, "--out", str(plain))[0], 0)

    @unittest.skipUnless(os.path.isdir("/tmp") and os.stat("/tmp").st_uid == 0 and os.stat("/tmp").st_mode & stat.S_ISVTX,
                         "needs a root-owned sticky /tmp")
    def test_root_owned_sticky_directory_is_accepted(self):
        out = Path(tempfile.mkdtemp(dir="/tmp")) / "o"
        self.addCleanup(shutil.rmtree, out.parent, True)
        self.assertEqual(run_cli(*self.args, "--out", str(out))[0], 0)


class TestAtomicWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_replaces_the_name_and_leaves_no_temporary_file(self):
        target = self.tmp / "report.csv"
        target.write_text("old")
        self.assertTrue(atomic.write(target, lambda tmp: tmp.write_text("new")))
        self.assertEqual(target.read_text(), "new")
        self.assertEqual(os.listdir(self.tmp), ["report.csv"])

    def test_symlink_and_hard_link_on_the_name_are_replaced_not_followed(self):
        victim = self.tmp / "victim"
        victim.write_bytes(b"a checkpoint")
        link_or_skip((self.tmp / "sym").symlink_to, victim)
        link_or_skip(os.link, victim, self.tmp / "hard")
        for name in ("sym", "hard"):
            atomic.write_text(self.tmp / name, "ours")
            self.assertEqual((self.tmp / name).read_text(), "ours")
            self.assertFalse((self.tmp / name).is_symlink())
        self.assertEqual(victim.read_bytes(), b"a checkpoint")

    def test_failed_writer_keeps_the_old_file_and_leaves_no_temporary_file(self):
        target = self.tmp / "report.csv"
        target.write_text("old")

        def broken(tmp):
            tmp.write_text("half")
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            atomic.write(target, broken)
        self.assertEqual(target.read_text(), "old")
        self.assertEqual(os.listdir(self.tmp), ["report.csv"])

    def test_keep_partial_installs_what_was_written_before_the_failure(self):
        target = self.tmp / "delta.csv"
        target.write_text("old")

        def broken(tmp):
            tmp.write_text("rows so far")
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            atomic.write(target, broken, keep_partial=True)
        self.assertEqual(target.read_text(), "rows so far")
        self.assertEqual(os.listdir(self.tmp), ["delta.csv"])

    def test_writer_returning_false_writes_nothing(self):
        target = self.tmp / "spectra.png"
        target.write_bytes(b"old")
        self.assertFalse(atomic.write(target, lambda tmp: False))
        self.assertEqual(target.read_bytes(), b"old")
        self.assertEqual(os.listdir(self.tmp), ["spectra.png"])

    def test_temporary_file_keeps_the_suffix_so_savefig_can_pick_a_format(self):
        seen = []
        atomic.write(self.tmp / "x.png", lambda tmp: seen.append(tmp.suffix))
        self.assertEqual(seen, [".png"])

    def test_temporary_file_lives_in_a_private_directory_that_is_gone_afterwards(self):
        seen = []

        def writer(tmp):
            seen.append((tmp, stat.S_IMODE(tmp.parent.stat().st_mode)))
            tmp.write_text("x")
        atomic.write(self.tmp / "x.csv", writer)
        tmp, mode = seen[0]
        self.assertEqual(tmp.parent.parent, self.tmp)
        self.assertNotEqual(tmp.parent, self.tmp)
        if os.name == "posix":
            self.assertEqual(mode, 0o700)
        self.assertFalse(tmp.parent.exists())
        self.assertEqual(os.listdir(self.tmp), ["x.csv"])

    def test_private_directory_is_removed_after_every_outcome(self):
        def boom(tmp):
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            atomic.write(self.tmp / "a.csv", boom)
        with self.assertRaises(RuntimeError):
            atomic.write(self.tmp / "b.csv", boom, keep_partial=True)
        atomic.write(self.tmp / "c.csv", lambda tmp: False)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["b.csv"])

    def test_only_the_directory_it_made_is_removed(self):
        (self.tmp / ".llmscan-foreign").mkdir()
        atomic.write_text(self.tmp / "x.csv", "ours")
        self.assertEqual(sorted(os.listdir(self.tmp)), [".llmscan-foreign", "x.csv"])

    def test_outputs_cover_every_file_the_cli_writes(self):
        self.assertIn("ranking.csv", cli.OUTPUTS)


class TestEmptyCheckpoints(unittest.TestCase):
    def test_two_empty_checkpoints_give_a_clear_message(self):
        tmp = Path(tempfile.mkdtemp())
        write_raw(tmp / "e.safetensors", {}, b"")
        with Checkpoint(tmp / "e.safetensors") as a, Checkpoint(tmp / "e.safetensors") as b:
            self.assertIn("neither checkpoint contains any tensors", inventory.summarize(inventory.compare(a, b)))
        code, _, err = run_cli("--base", str(tmp / "e.safetensors"), "--suspect", str(tmp / "e.safetensors"), "--out", str(tmp / "out"))
        self.assertEqual(code, 1)
        self.assertIn("neither checkpoint contains any tensors", err)


if __name__ == "__main__":
    unittest.main()
