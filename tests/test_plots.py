import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib.image import imread

from llmscan.plots import plot_spectra


class TestSpectra(unittest.TestCase):
    def test_singleton_values_are_visible_by_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spectra.png"

            def render(values):
                rows = [{"kind": "o_proj", "layer": str(i), "effective_rank": "1", "sv_001": value}
                        for i, value in enumerate(values)]
                plot_spectra(rows, path, "Singleton spectra")
                return imread(path)

            # Swapping values keeps the axes and colorbar fixed; only the data marks change.
            self.assertFalse(np.array_equal(render(["1", "10"]), render(["10", "1"])),
                             "swapping singleton values between layers did not change the chart")
