"""The workbench's report reader (web/report.mjs) and its demo (web/demo), against the real command line."""
import contextlib
import csv
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import tiny_llama, write_safetensors
from llmscan.cli import main

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "web" / "demo"
NODE = shutil.which("node")


def node_major() -> int:
    if NODE is None:
        return 0
    return int(subprocess.run([NODE, "--version"], capture_output=True, text=True, check=True).stdout.lstrip("v").split(".")[0])


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_report(folder: Path) -> dict:
    done = subprocess.run([NODE, str(ROOT / "tests" / "web" / "read_report.mjs"), str(folder)],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(done.stdout)


def demo_generator():
    spec = importlib.util.spec_from_file_location("demo_generate", DEMO / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(node_major() >= 20, "needs Node 20 or newer")
class ReportReader(unittest.TestCase):
    def test_loader_tests(self):
        done = subprocess.run([NODE, "--test", str(ROOT / "tests" / "web" / "report.test.mjs")],
                              capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_reads_what_the_command_line_writes(self):
        for extra in ([], ["--inventory-only"]):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                base = tiny_llama(np.random.default_rng(1))
                suspect = dict(base)
                edited = "model.layers.1.self_attn.o_proj.weight"
                suspect[edited] = ("BF16", base[edited][1] + 0.25)
                added = 'adapter "a", merged\nlater.weight'  # the csv module has to quote this name
                suspect[added] = ("F32", np.ones((2, 2), dtype=np.float32))
                for name, tensors in (("base", base), ("suspect", suspect)):
                    (tmp / name).mkdir()
                    write_safetensors(tmp / name / "model.safetensors", tensors)
                out = tmp / "out"
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = main(["--base", str(tmp / "base"), "--suspect", str(tmp / "suspect"), "--out", str(out), *extra])
                self.assertEqual(code, 0)
                report = read_report(out)
                self.assertEqual(report["base"], str((tmp / "base").resolve()))
                self.assertEqual(report["suspect"], str((tmp / "suspect").resolve()))
                self.assertEqual(report["inventory"], rows(out / "inventory.csv"))
                self.assertEqual(report["ranking"], rows(out / "ranking.csv"))
                self.assertEqual(report["ranking"][0]["name"], added)
                self.assertEqual(report["delta"], [] if extra else rows(out / "delta.csv"))
                self.assertEqual((out / "delta.csv").exists(), not extra)


class Demo(unittest.TestCase):
    def test_holds_only_the_public_report(self):
        self.assertEqual(sorted(p.name for p in DEMO.iterdir() if p.name != "__pycache__"),
                         [".gitattributes", "delta.csv", "generate.py", "inventory.csv", "pair.txt", "ranking.csv"])
        self.assertLess(sum(p.stat().st_size for p in DEMO.iterdir() if p.is_file()), 64_000)
        self.assertEqual((DEMO / "pair.txt").read_bytes(),
                         b"base: synthetic/tiny-llama\nsuspect: synthetic/tiny-llama-abliterated-tampered\n")
        for name in ("inventory.csv", "ranking.csv", "delta.csv"):
            text = (DEMO / name).read_text(encoding="utf-8")
            for private in (tempfile.gettempdir(), str(Path.home()), "/tmp/", "/home/", "/Users/", ":\\"):
                self.assertNotIn(private, text, name)

    def test_is_what_the_generator_writes(self):
        generator = demo_generator()
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp)
            generator.generate(fresh)
            self.assertEqual(sorted(p.name for p in fresh.iterdir()), sorted(generator.FILES))
            for name in ("pair.txt", "inventory.csv"):
                self.assertEqual((fresh / name).read_bytes(), (DEMO / name).read_bytes(), name)
            # singular values depend on the platform's LAPACK in their last digits: compare them as numbers,
            # and leave out what follows from them (the order of near-ties, the numbers quoted in a reason)
            for name in ("ranking.csv", "delta.csv"):
                kept, new = ({r["name"]: r for r in rows(folder / name)} for folder in (DEMO, fresh))
                self.assertEqual(kept.keys(), new.keys(), name)
                for tensor, a in kept.items():
                    b = new[tensor]
                    self.assertEqual(a.keys(), b.keys(), name)
                    for column in a.keys() - {"rank", "reason", "seconds"}:
                        if column == "effective_rank" or column.startswith("sv_"):
                            self.assertEqual(a[column] == "", b[column] == "", (name, tensor, column))
                            if a[column]:
                                self.assertAlmostEqual(float(a[column]), float(b[column]), delta=1e-3, msg=(name, tensor, column))
                        else:
                            self.assertEqual(a[column], b[column], (name, tensor, column))
            planted = len(generator.PLANTED)
            self.assertEqual([r["name"] for r in rows(fresh / "ranking.csv")[:planted]],
                             [r["name"] for r in rows(DEMO / "ranking.csv")[:planted]])

    def test_ranks_every_planted_change_above_the_honest_edits(self):
        generator = demo_generator()
        ranking = rows(DEMO / "ranking.csv")
        planted = len(generator.PLANTED)
        self.assertEqual({r["name"] for r in ranking[:planted]}, set(generator.PLANTED))
        self.assertEqual([r["tier"] for r in ranking[:planted]], ["T1", "T1", "T1", "T2", "T2", "T3"])
        self.assertEqual({r["tier"] for r in ranking[planted:]}, {"T4"})
        self.assertEqual(len(ranking) - planted, 21)


if __name__ == "__main__":
    unittest.main()
