import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import tiny_llama, write_safetensors, write_sharded
from llmscan import cli, core, inventory
from llmscan.names import kind_of, layer_of
from llmscan.safetensors_io import Checkpoint, CheckpointError


def fake_analyze(w_base, w_suspect, k):
    """Stand-in for the core: records the call, returns fixed numbers."""
    fake_analyze.calls.append((w_base.shape, w_suspect.shape, k))
    return core.DeltaStats(np.linspace(1.0, 0.1, min(k, *w_base.shape)), 1.0)


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestReader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_decodes_bf16_f16_f32_exactly(self):
        a = np.array([[0.5, -1.25], [3.0, 0.0]], dtype=np.float32)
        write_safetensors(self.tmp / "m.safetensors", {"bf": ("BF16", a), "h": ("F16", a), "f": ("F32", a)})
        with Checkpoint(self.tmp / "m.safetensors") as c:
            for name in ("bf", "h", "f"):
                got = c.to_float32(name)
                self.assertEqual(got.dtype, np.float32)
                np.testing.assert_array_equal(got, a)
                got[0, 0] = 9  # must be a writable copy, not a view of the mmap

    def test_sharded_directory_matches_single_file(self):
        t = tiny_llama(np.random.default_rng(0))
        names = list(t)
        write_safetensors(self.tmp / "single.safetensors", t)
        write_sharded(self.tmp / "sharded", [{n: t[n] for n in names[:7]}, {n: t[n] for n in names[7:]}])
        with Checkpoint(self.tmp / "single.safetensors") as a, Checkpoint(self.tmp / "sharded") as b:
            self.assertEqual(a.tensors.keys(), b.tensors.keys())
            for n in names:
                np.testing.assert_array_equal(a.to_float32(n), b.to_float32(n))

    def test_rejects_truncated_file(self):
        write_safetensors(self.tmp / "m.safetensors", {"x": ("F32", np.ones((4, 4), np.float32))})
        data = (self.tmp / "m.safetensors").read_bytes()
        (self.tmp / "m.safetensors").write_bytes(data[:-8])
        with self.assertRaises(CheckpointError):
            Checkpoint(self.tmp / "m.safetensors")

    def test_bytes_stream_in_chunks(self):
        a = np.arange(100, dtype=np.float32)
        write_safetensors(self.tmp / "m.safetensors", {"x": ("F32", a)})
        with Checkpoint(self.tmp / "m.safetensors") as c:
            chunks = list(c.iter_bytes("x", chunk=64))
        self.assertEqual(len(chunks), 7)
        self.assertEqual(b"".join(chunks), a.tobytes())


class TestNames(unittest.TestCase):
    def test_layer_and_kind(self):
        self.assertEqual(layer_of("model.layers.12.mlp.down_proj.weight"), 12)
        self.assertEqual(layer_of("blk.3.attn_q.weight"), 3)
        self.assertIsNone(layer_of("model.embed_tokens.weight"))
        self.assertEqual(kind_of("model.layers.12.mlp.down_proj.weight"), "down_proj")
        self.assertEqual(kind_of("model.layers.0.self_attn.q_proj.bias"), "q_proj.bias")
        self.assertEqual(kind_of("lm_head.weight"), "lm_head")


class TestInventoryAndCli(unittest.TestCase):
    """A base and a suspect that differs in known places."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(1)
        base = tiny_llama(rng)
        suspect = dict(base)
        for name in ("model.layers.0.self_attn.o_proj.weight", "model.layers.1.mlp.down_proj.weight"):
            dt, w = base[name]
            suspect[name] = (dt, w + np.outer(np.ones(w.shape[0]), np.ones(w.shape[1])) / 16)
        suspect["model.norm.weight"] = ("BF16", base["model.norm.weight"][1] + 0.5)
        suspect["model.layers.1.self_attn.q_proj.weight"] = ("F16", base["model.layers.1.self_attn.q_proj.weight"][1])
        suspect["model.layers.0.mlp.up_proj.weight"] = ("BF16", np.zeros((17, 8), np.float32))
        del suspect["lm_head.weight"]
        suspect["model.layers.0.self_attn.trigger.weight"] = ("BF16", np.ones((2, 8), np.float32))
        write_safetensors(self.tmp / "base.safetensors", base)
        names = list(suspect)
        write_sharded(self.tmp / "suspect", [{n: suspect[n] for n in names[:5]}, {n: suspect[n] for n in names[5:]}])

    def test_inventory_statuses(self):
        with Checkpoint(self.tmp / "base.safetensors") as b, Checkpoint(self.tmp / "suspect") as s:
            rows = {r.name: r.status for r in inventory.compare(b, s)}
        self.assertEqual(rows["model.layers.0.self_attn.o_proj.weight"], inventory.CHANGED)
        self.assertEqual(rows["model.layers.1.mlp.down_proj.weight"], inventory.CHANGED)
        self.assertEqual(rows["model.norm.weight"], inventory.CHANGED)
        self.assertEqual(rows["model.layers.1.self_attn.q_proj.weight"], inventory.DTYPE_MISMATCH)
        self.assertEqual(rows["model.layers.0.mlp.up_proj.weight"], inventory.SHAPE_MISMATCH)
        self.assertEqual(rows["lm_head.weight"], inventory.ONLY_BASE)
        self.assertEqual(rows["model.layers.0.self_attn.trigger.weight"], inventory.ONLY_SUSPECT)
        self.assertEqual(rows["model.embed_tokens.weight"], inventory.IDENTICAL)
        self.assertEqual(rows["model.layers.0.self_attn.q_proj.weight"], inventory.IDENTICAL)

    def test_cli_stops_cleanly_while_core_is_a_stub(self):
        code, out, err = run_cli("--base", str(self.tmp / "base.safetensors"), "--suspect", str(self.tmp / "suspect"),
                                 "--out", str(self.tmp / "out"))
        self.assertEqual(code, 2)
        self.assertIn("analyze_delta is not written yet", err)
        self.assertTrue((self.tmp / "out" / "inventory.csv").exists())

    def test_cli_end_to_end_with_stand_in_core(self):
        fake_analyze.calls = []
        with mock.patch.object(core, "analyze_delta", fake_analyze):
            code, out, _ = run_cli("--base", str(self.tmp / "base.safetensors"), "--suspect", str(self.tmp / "suspect"),
                                   "--out", str(self.tmp / "out"), "-k", "4")
        self.assertEqual(code, 0, out)
        # core runs on changed 2-D tensors and on the same-shape dtype mismatch, never on identical ones
        self.assertEqual(sorted(fake_analyze.calls), sorted([((8, 8), (8, 8), 4), ((8, 16), (8, 16), 4), ((8, 8), (8, 8), 4)]))
        with open(self.tmp / "out" / "delta.csv") as f:
            rows = {r["name"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["model.layers.1.mlp.down_proj.weight"]["sv_004"], "0.1")
        self.assertEqual(rows["model.embed_tokens.weight"]["effective_rank"], "")
        self.assertNotIn("model.norm.weight", rows)  # 1-D: inventory only
        for png in ("spectra.png", "effective_rank.png"):
            self.assertGreater((self.tmp / "out" / png).stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
