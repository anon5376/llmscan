import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from llmscan.plots import plot_spectra


class TestSpectraValues(unittest.TestCase):
    def plot_line(self, *values):
        row = {"kind": "o_proj", "layer": "", "effective_rank": "1"}
        row.update({f"sv_{i:03d}": value for i, value in enumerate(values, 1)})
        with mock.patch.object(Figure, "savefig", autospec=True) as save:
            plot_spectra([row], Path("spectra.png"), "Test spectrum")
        ax, = save.call_args.args[0].axes
        self.assertEqual(ax.get_yscale(), "log")
        line, = ax.lines
        return line

    def test_log_spectrum_keeps_positive_csv_values_at_their_indices(self):
        line = self.plot_line("1e-13", "1e-14", "0")
        np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
        np.testing.assert_array_equal(line.get_ydata(), [1e-13, 1e-14, np.nan])

    def test_zero_tail_does_not_hide_the_only_positive_value(self):
        line = self.plot_line("1e-14", "0", "0")
        np.testing.assert_array_equal(line.get_ydata(), [1e-14, np.nan, np.nan])
        self.assertNotIn(line.get_marker(), (None, "None", "", " "))

    def test_no_log_height_for_zero_or_nonfinite_values(self):
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value):
                line = self.plot_line(value, value)
                self.assertTrue(np.all(np.isnan(line.get_ydata())))
