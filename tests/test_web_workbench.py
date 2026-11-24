"""The workbench's view-model and chart maths (web/model.mjs, web/charts.mjs), run under Node when it is installed."""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def node_major() -> int:
    if NODE is None:
        return 0
    return int(subprocess.run([NODE, "--version"], capture_output=True, text=True, check=True).stdout.lstrip("v").split(".")[0])


@unittest.skipUnless(node_major() >= 20, "needs Node 20 or newer")
class Workbench(unittest.TestCase):
    def test_view_model_tests(self):
        done = subprocess.run([NODE, "--test", str(ROOT / "tests" / "web" / "workbench.test.mjs")],
                              capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
