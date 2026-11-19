import contextlib
import csv
import io
import tempfile
import unittest
import weakref
from pathlib import Path

import numpy as np

from helpers import write_safetensors
from llmscan import cli, delta, inventory
from test_core import to_bf16


class TestPrecisionConsistency(unittest.TestCase):
    def test_scan_keeps_at_most_one_tensor_pair_alive(self):
        refs, live_counts = [], []

        class Loader:
            def __init__(self, value):
                self.value = value

            def to_float32(self, name):
                tensor = np.full((4, 4), self.value, dtype=np.float32)
                refs.append(weakref.ref(tensor))
                live_counts.append(sum(ref() is not None for ref in refs))
                return tensor

        rows = [inventory.InventoryRow(f"weight.{i}", i, "weight", inventory.CHANGED,
                                       "BF16", "F32", "4x4", "4x4", 16) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            delta.scan(Loader(0), Loader(1), rows, Path(directory) / "delta.csv", log=lambda _: None)
        self.assertLessEqual(max(live_counts), 2, live_counts)

    def test_local_precision_departure_is_flagged_but_a_uniform_float32_edit_is_not(self):
        rng = np.random.default_rng(42)
        base, rounded, full_precision = {}, {}, {}
        direction = rng.normal(size=(64, 1)).astype(np.float32)
        direction /= np.linalg.norm(direction)
        for layer in range(5):
            name = f"model.layers.{layer}.self_attn.o_proj.weight"
            weight = to_bf16(rng.normal(0, 0.02, (64, 96)).astype(np.float32))
            changed = weight - direction @ (direction.T @ weight)
            base[name] = ("BF16", weight)
            rounded[name] = ("F32", to_bf16(changed))
            full_precision[name] = ("F32", changed)

        planted = "model.layers.3.self_attn.o_proj.weight"
        mixed = {name: (dtype, weight.copy()) for name, (dtype, weight) in rounded.items()}
        mixed[planted][1][7:9, 11:14] *= np.float32(1.04)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_safetensors(directory / "base.safetensors", base)
            rankings = {}
            for label, suspect in (("rounded", rounded), ("full_precision", full_precision), ("mixed", mixed)):
                model = directory / f"{label}.safetensors"
                out = directory / label
                write_safetensors(model, suspect)
                with contextlib.redirect_stdout(io.StringIO()):
                    code = cli.main(["--base", str(directory / "base.safetensors"),
                                     "--suspect", str(model), "--out", str(out)])
                self.assertEqual(code, 0)
                with (out / "ranking.csv").open(newline="") as stream:
                    rankings[label] = list(csv.DictReader(stream))

            with (directory / "mixed" / "delta.csv").open(newline="") as stream:
                counts = {row["name"]: int(row["non_bf16_values"]) for row in csv.DictReader(stream)}
            expected = dict.fromkeys(rounded, 0)
            weight = mixed[planted][1]
            expected[planted] = np.count_nonzero(weight != to_bf16(weight))
            self.assertEqual(counts, expected)

        for label in ("rounded", "full_precision"):
            self.assertEqual({row["tier"] for row in rankings[label]}, {"T4"}, label)
        first = rankings["mixed"][0]
        self.assertEqual((first["name"], first["tier"]), (planted, "T3"))
        self.assertIn("precision", first["reason"])
        self.assertEqual({row["tier"] for row in rankings["mixed"][1:]}, {"T4"})


if __name__ == "__main__":
    unittest.main()
