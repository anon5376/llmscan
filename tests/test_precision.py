import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import write_safetensors
from llmscan import cli
from test_core import to_bf16


class TestPrecisionConsistency(unittest.TestCase):
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

        for label in ("rounded", "full_precision"):
            self.assertEqual({row["tier"] for row in rankings[label]}, {"T4"}, label)
        first = rankings["mixed"][0]
        self.assertEqual((first["name"], first["tier"]), (planted, "T3"))
        self.assertIn("precision", first["reason"])
        self.assertEqual({row["tier"] for row in rankings["mixed"][1:]}, {"T4"})


if __name__ == "__main__":
    unittest.main()
