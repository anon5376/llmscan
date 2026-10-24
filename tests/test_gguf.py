import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import GGML, encode, gguf_float, nbytes_gguf, write_gguf
from llmscan import cli, inventory
from llmscan.checkpoint import open_checkpoint
from llmscan.gguf_io import GGUFCheckpoint
from llmscan.safetensors_io import CheckpointError
from test_pipeline import run_cli


class TestGGUFReader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "m.gguf"

    def test_every_metadata_value_type(self):
        meta = [
            ("t.u8", 0, 200), ("t.i8", 1, -100), ("t.u16", 2, 65000), ("t.i16", 3, -30000),
            ("t.u32", 4, 4_000_000_000), ("t.i32", 5, -2_000_000_000), ("t.f32", 6, 0.5),
            ("t.bool", 7, True), ("t.str", 8, "héllo ✓"), ("t.u64", 10, 2**63 + 5),
            ("t.i64", 11, -(2**62)), ("t.f64", 12, 1.0e-300),
            ("t.arr_int", 9, (4, [1, 2, 3])), ("t.arr_str", 9, (8, ["a", "", "bc"])),
            ("t.arr_empty", 9, (6, [])), ("t.arr_bool", 9, (7, [True, False])),
            ("t.nested", 9, (9, [(4, [1, 2]), (8, ["x"]), (9, [(0, [7])])])),
        ]
        write_gguf(self.path, {"x": gguf_float("F32", np.zeros((1,), np.float32))}, meta)
        with GGUFCheckpoint(self.path) as c:
            m = c.metadata
        self.assertEqual({k: m[k] for k in m if k != "t.nested"}, {
            "t.u8": 200, "t.i8": -100, "t.u16": 65000, "t.i16": -30000, "t.u32": 4_000_000_000,
            "t.i32": -2_000_000_000, "t.f32": 0.5, "t.bool": True, "t.str": "héllo ✓",
            "t.u64": 2**63 + 5, "t.i64": -(2**62), "t.f64": 1.0e-300, "t.arr_int": [1, 2, 3],
            "t.arr_str": ["a", "", "bc"], "t.arr_empty": [], "t.arr_bool": [True, False],
        })
        self.assertEqual(m["t.nested"], [[1, 2], ["x"], [[7]]])

    def test_float_tensors_decode_exactly_and_shapes_are_reversed(self):
        a = np.array([[0.5, -1.25, 3.0], [2.0, 0.0, -0.75]], dtype=np.float32)
        write_gguf(self.path, {
            "bf": gguf_float("BF16", a), "h": gguf_float("F16", a), "f": gguf_float("F32", a),
            "v": gguf_float("F32", a[0]),
        })
        with GGUFCheckpoint(self.path) as c:
            self.assertEqual(c.files, [self.path])
            self.assertEqual(c.ignored, {})
            self.assertEqual({n: t.dtype for n, t in c.tensors.items()}, {"bf": "BF16", "h": "F16", "f": "F32", "v": "F32"})
            self.assertEqual(c.tensors["f"].shape, (2, 3))  # file dims are [3, 2]
            for name in ("bf", "h", "f"):
                got = c.to_float32(name)
                self.assertEqual((got.dtype, got.shape), (np.float32, (2, 3)))
                np.testing.assert_array_equal(got, a)
                got[0, 0] = 9  # must be a writable copy, not a view of the mmap
            np.testing.assert_array_equal(c.to_float32("v"), a[0])

    def test_non_default_alignment(self):
        a = np.arange(5, dtype=np.float32)
        b = a + 100
        for alignment in (8, 64, 100):
            with self.subTest(alignment=alignment):
                write_gguf(self.path, {"a": gguf_float("F32", a), "b": gguf_float("F16", b)},
                           [("general.architecture", 8, "llama")], alignment=alignment)
                with GGUFCheckpoint(self.path) as c:
                    self.assertEqual(c.metadata["general.alignment"], alignment)
                    self.assertEqual(c.tensors["a"].start % alignment, 0)
                    np.testing.assert_array_equal(c.to_float32("a"), a)
                    np.testing.assert_array_equal(c.to_float32("b"), b)

    def test_version_2(self):
        write_gguf(self.path, {"a": gguf_float("F32", np.ones(3, np.float32))}, version=2)
        with GGUFCheckpoint(self.path) as c:
            np.testing.assert_array_equal(c.to_float32("a"), np.ones(3))

    def test_every_listed_quantization_has_bytes_but_no_decoding(self):
        rng = np.random.default_rng(0)
        shape = (3, 256)
        raws = {t: rng.integers(0, 256, nbytes_gguf(t, shape), dtype=np.uint8).tobytes() for t in GGML}
        for t in ("F32", "F16", "BF16"):  # random bytes can be NaN/inf patterns; keep these finite
            raws[t] = encode(rng.standard_normal(shape), t)
        write_gguf(self.path, {t: (t, shape, raws[t]) for t in GGML})
        with GGUFCheckpoint(self.path) as c:
            for t in GGML:
                with self.subTest(dtype=t):
                    info = c.tensors[t]
                    self.assertEqual((info.dtype, info.shape, info.nbytes), (t, shape, len(raws[t])))
                    self.assertEqual(b"".join(c.iter_bytes(t)), raws[t])
                    if t in ("F32", "F16", "BF16"):
                        self.assertEqual(c.to_float32(t).shape, shape)
                    else:
                        with self.assertRaisesRegex(CheckpointError, "dequantization not supported yet"):
                            c.to_float32(t)

    def test_bytes_stream_in_chunks(self):
        raw = bytes(range(256)) * 4
        write_gguf(self.path, {"q": ("Q8_0", (1, 32 * 30), raw[:1020])})
        with GGUFCheckpoint(self.path) as c:
            chunks = list(c.iter_bytes("q", chunk=100))
            self.assertGreater(len(chunks), 1)
            self.assertEqual(b"".join(chunks), raw[:1020])

    def test_rejects_unknown_ggml_type(self):
        write_gguf(self.path, {"x": (19, (4,), bytes(16))})
        with self.assertRaisesRegex(CheckpointError, "type id 19"):
            GGUFCheckpoint(self.path)

    def test_rejects_shape_not_a_whole_number_of_blocks(self):
        write_gguf(self.path, {"x": ("Q8_0", (1, 40), bytes(200))})
        with self.assertRaisesRegex(CheckpointError, "block size"):
            GGUFCheckpoint(self.path)

    def test_rejects_bad_magic_version_and_endianness(self):
        write_gguf(self.path, {"x": gguf_float("F32", np.ones(2, np.float32))})
        good = self.path.read_bytes()
        cases = {
            "bad magic": b"GGML" + good[4:],
            "version 1": good[:4] + (1).to_bytes(4, "little") + good[8:],
            "version 4": good[:4] + (4).to_bytes(4, "little") + good[8:],
            "big-endian": good[:4] + (3).to_bytes(4, "big") + good[8:],
        }
        for label, data in cases.items():
            with self.subTest(label):
                self.path.write_bytes(data)
                with self.assertRaises(CheckpointError):
                    GGUFCheckpoint(self.path)
        self.path.write_bytes(cases["big-endian"])
        with self.assertRaisesRegex(CheckpointError, "big-endian"):
            GGUFCheckpoint(self.path)

    def test_rejects_every_truncation(self):
        meta = [("general.architecture", 8, "llama"), ("t.arr", 9, (8, ["ab", "cd"]))]
        write_gguf(self.path, {"x": gguf_float("F32", np.ones((2, 2), np.float32))}, meta)
        good = self.path.read_bytes()
        with GGUFCheckpoint(self.path) as c:
            needed = c.tensors["x"].end  # bytes after the last tensor are only padding
        for cut in range(needed):  # every proper prefix: header, kv, infos, padding, data
            with self.subTest(cut=cut):
                self.path.write_bytes(good[:cut])
                with self.assertRaises(CheckpointError):
                    GGUFCheckpoint(self.path)

    def test_rejects_corrupt_counts_and_duplicates(self):
        write_gguf(self.path, {"x": gguf_float("F32", np.ones(2, np.float32))}, [("a", 4, 1)])
        good = self.path.read_bytes()
        self.path.write_bytes(good[:16] + (2**60).to_bytes(8, "little") + good[24:])  # kv_count
        with self.assertRaises(CheckpointError):
            GGUFCheckpoint(self.path)
        write_gguf(self.path, {}, [("a", 4, 1), ("a", 4, 2)])
        with self.assertRaisesRegex(CheckpointError, "duplicate"):
            GGUFCheckpoint(self.path)
        # an array claiming 2**60 elements must fail on bounds, not try to allocate
        write_gguf(self.path, {}, [("big", 9, (4, []))])
        data = bytearray(self.path.read_bytes())
        i = data.index(b"big") + 3 + 4 + 4  # key, value type, element type -> count
        data[i:i + 8] = (2**60).to_bytes(8, "little")
        self.path.write_bytes(bytes(data))
        with self.assertRaisesRegex(CheckpointError, "truncated"):
            GGUFCheckpoint(self.path)

    def test_rejects_empty_missing_and_split_files(self):
        self.path.write_bytes(b"")
        with self.assertRaises(CheckpointError):
            GGUFCheckpoint(self.path)
        with self.assertRaises(CheckpointError):
            GGUFCheckpoint(self.tmp / "missing.gguf")
        shard = self.tmp / "m-00001-of-00003.gguf"
        write_gguf(shard, {"x": gguf_float("F32", np.ones(2, np.float32))})
        with self.assertRaisesRegex(CheckpointError, "split GGUF"):
            GGUFCheckpoint(shard)
        write_gguf(self.path, {}, [("split.count", 2, 3)])
        with self.assertRaisesRegex(CheckpointError, "split GGUF"):
            GGUFCheckpoint(self.path)

    def test_open_checkpoint_picks_reader_by_extension(self):
        write_gguf(self.path, {"x": gguf_float("F32", np.ones(2, np.float32))})
        with open_checkpoint(self.path) as c:
            self.assertIsInstance(c, GGUFCheckpoint)


class TestGGUFInventoryAndCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(1)

        def f(*shape):
            return rng.integers(-32, 32, size=shape).astype(np.float32) / 16

        self.base = {
            "token_embd.weight": gguf_float("F16", f(8, 4)),
            "blk.0.attn_q.weight": gguf_float("F32", f(4, 4)),
            "blk.0.ffn_down.weight": gguf_float("BF16", f(4, 8)),
            "blk.0.ffn_up.weight": ("Q8_0", (2, 64), rng.integers(0, 256, 136, dtype=np.uint8).tobytes()),
            "blk.0.ffn_gate.weight": ("Q4_K", (1, 256), rng.integers(0, 256, 144, dtype=np.uint8).tobytes()),
            "output_norm.weight": gguf_float("F32", f(4)),
        }
        sus = dict(self.base)
        q = bytearray(self.base["blk.0.ffn_up.weight"][2])
        q[100] ^= 1
        sus["blk.0.ffn_up.weight"] = ("Q8_0", (2, 64), bytes(q))  # one bit flipped in a quantized tensor
        sus["blk.0.attn_q.weight"] = gguf_float("F32", f(4, 4))  # different values
        embd = np.frombuffer(self.base["token_embd.weight"][2], "<f2").astype(np.float32).reshape(8, 4)
        sus["token_embd.weight"] = gguf_float("F32", embd)  # same values, wider dtype
        sus["blk.0.ffn_down.weight"] = gguf_float("BF16", f(4, 9))  # shape change
        del sus["output_norm.weight"]
        sus["extra.weight"] = gguf_float("F32", f(3))
        write_gguf(self.tmp / "base.gguf", self.base)
        write_gguf(self.tmp / "sus.gguf", sus, alignment=64)

    def test_inventory_statuses(self):
        with GGUFCheckpoint(self.tmp / "base.gguf") as b, GGUFCheckpoint(self.tmp / "sus.gguf") as s:
            got = {r.name: r.status for r in inventory.compare(b, s)}
        self.assertEqual(got, {
            "token_embd.weight": inventory.IDENTICAL,  # F16 vs the same values as F32: compared by value
            "blk.0.attn_q.weight": inventory.CHANGED,
            "blk.0.ffn_down.weight": inventory.SHAPE_MISMATCH,
            "blk.0.ffn_up.weight": inventory.CHANGED,
            "blk.0.ffn_gate.weight": inventory.IDENTICAL,
            "output_norm.weight": inventory.ONLY_BASE,
            "extra.weight": inventory.ONLY_SUSPECT,
        })

    def test_quantized_dtype_mismatch_is_reported_not_decoded(self):
        raw = bytes(144)
        write_gguf(self.tmp / "a.gguf", {"w": ("Q4_K", (1, 256), raw)})
        write_gguf(self.tmp / "b.gguf", {"w": ("Q6_K", (1, 256), bytes(210))})
        with GGUFCheckpoint(self.tmp / "a.gguf") as a, GGUFCheckpoint(self.tmp / "b.gguf") as b:
            self.assertEqual(inventory.compare(a, b)[0].status, inventory.DTYPE_MISMATCH)

    def test_cli_inventory_only(self):
        out = self.tmp / "out"
        code, stdout, _ = run_cli("--base", str(self.tmp / "base.gguf"), "--suspect", str(self.tmp / "sus.gguf"),
                                  "--out", str(out), "--inventory-only")
        self.assertEqual(code, 0)
        self.assertIn("6 tensors in base.gguf", stdout)
        self.assertIn("tensors: 7", stdout)
        self.assertIn("shape_mismatch", (out / "inventory.txt").read_text())
        self.assertTrue((out / "inventory.csv").is_file())

    def test_cli_reports_split_file_cleanly(self):
        shard = self.tmp / "m-00001-of-00002.gguf"
        write_gguf(shard, {"x": gguf_float("F32", np.ones(2, np.float32))})
        with self.assertRaisesRegex(SystemExit, "split GGUF"):
            run_cli("--base", str(shard), "--suspect", str(shard), "--out", str(self.tmp / "o"), "--inventory-only")


if __name__ == "__main__":
    unittest.main()
