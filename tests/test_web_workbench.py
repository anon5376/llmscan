"""The workbench's view-model and chart maths (web/model.mjs, web/charts.mjs), run under Node when it is installed."""
import subprocess
import unittest

from test_web_report import NODE, ROOT, node_major


@unittest.skipUnless(node_major() >= 20, "needs Node 20 or newer")
class Workbench(unittest.TestCase):
    def test_view_model_tests(self):
        done = subprocess.run([NODE, "--test", str(ROOT / "tests" / "web" / "workbench.test.mjs")],
                              capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
