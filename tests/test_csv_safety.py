import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from llmscan import core, delta, inventory, plots, rank
from llmscan.csv_safety import ENCODING_COLUMN, ENCODING_MARKER, csv_fieldnames, decode_row, encode_row
from llmscan.inventory import CHANGED, InventoryRow
from llmscan.rank import RankedRow
from llmscan.signals import Measurements

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_blind.py"
PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n", "'")


def raw_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_blind", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCsvSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_each_writer_marks_and_escapes_formula_prefixes_in_raw_cells(self):
        rows = [
            InventoryRow(prefix + "name", i, prefix + "kind", prefix + "status", prefix + "base",
                         prefix + "suspect", prefix + "base_shape", prefix + "suspect_shape", i)
            for i, prefix in enumerate(PREFIXES)
        ]
        inventory_path = self.tmp / "inventory.csv"
        inventory.write_csv(rows, inventory_path)
        raw = raw_rows(inventory_path)
        self.assertEqual([r["name"] for r in raw], ["'" + r.name for r in rows])
        self.assertEqual([r["kind"] for r in raw], ["'" + r.kind for r in rows])
        self.assertEqual([r["status"] for r in raw], ["'" + r.status for r in rows])
        self.assertEqual([r[ENCODING_COLUMN] for r in raw], [ENCODING_MARKER] * len(rows))
        self.assertEqual([r["layer"] for r in raw], [str(i) for i in range(len(rows))])
        self.assertEqual([r["numel"] for r in raw], [str(i) for i in range(len(rows))])
        self.assertEqual([r.name for r in rows], [prefix + "name" for prefix in PREFIXES])

        delta_rows = [InventoryRow(prefix + "delta", i, prefix + "kind", CHANGED, "F32", "F32",
                                   prefix + "x1", prefix + "x1", 1)
                      for i, prefix in enumerate(PREFIXES)]

        class Input:
            def to_float32(self, name):
                return np.ones((1, 1), np.float32)

        delta_path = self.tmp / "delta.csv"
        with mock.patch.object(delta.core, "analyze_delta", return_value=core.DeltaStats(
                np.array([2.0]), 1.0)):
            delta.scan(Input(), Input(), delta_rows, delta_path, k=1, log=lambda _: None)
        raw = raw_rows(delta_path)
        self.assertEqual([r["name"] for r in raw], ["'" + r.name for r in delta_rows])
        self.assertEqual([r["kind"] for r in raw], ["'" + r.kind for r in delta_rows])
        self.assertEqual([r["shape"] for r in raw], ["'" + r.base_shape for r in delta_rows])
        self.assertEqual([r[ENCODING_COLUMN] for r in raw], [ENCODING_MARKER] * len(delta_rows))
        self.assertEqual([r.name for r in delta_rows], [prefix + "delta" for prefix in PREFIXES])

        ranked = [RankedRow(i + 1, "T4", prefix + "rank", i, prefix + "kind", prefix + "status",
                            1.0, 2.0, prefix + "reason") for i, prefix in enumerate(PREFIXES)]
        ranking_path = self.tmp / "ranking.csv"
        rank.write_csv(ranked, ranking_path)
        raw = raw_rows(ranking_path)
        self.assertEqual([r["name"] for r in raw], ["'" + r.name for r in ranked])
        self.assertEqual([r["kind"] for r in raw], ["'" + r.kind for r in ranked])
        self.assertEqual([r["reason"] for r in raw], ["'" + r.reason for r in ranked])
        self.assertEqual([r[ENCODING_COLUMN] for r in raw], [ENCODING_MARKER] * len(ranked))
        self.assertEqual([r["rank"] for r in raw], [str(i) for i in range(1, len(ranked) + 1)])
        self.assertEqual([r.name for r in ranked], [prefix + "rank" for prefix in PREFIXES])

    def test_delta_reader_decodes_before_rank_join_and_plots(self):
        name = "=model.layers.0.mlp.down_proj.weight"
        row = InventoryRow(name, 0, "down_proj", CHANGED, "F32", "F32", "1x1", "1x1", 1)

        class Input:
            def to_float32(self, tensor_name):
                self.assert_name = tensor_name
                return np.ones((1, 1), np.float32)

        path = self.tmp / "delta.csv"
        with mock.patch.object(delta.core, "analyze_delta", return_value=core.DeltaStats(
                np.array([2.5]), 1.5)):
            delta.scan(Input(), Input(), [row], path, k=1, log=lambda _: None)
        self.assertEqual(raw_rows(path)[0]["name"], "'" + name)
        stats = rank.read_delta_csv(path)
        self.assertEqual(stats[name].effective_rank, 1.5)
        self.assertEqual(stats[name].sv_001, 2.5)
        self.assertEqual(rank.rank_tensors([row], stats)[0].name, name)
        self.assertEqual(rank.rank_tensors([row], stats)[0].effective_rank, 1.5)
        plotted = plots.read_rows(path)
        self.assertEqual(plotted[0]["name"], name)
        self.assertNotIn(ENCODING_COLUMN, plotted[0])
        self.assertEqual(row.name, name)

    def test_marked_round_trip_handles_apostrophes_csv_syntax_unicode_and_safe_names(self):
        names = [
            "=x", "+x", "-x", "@x", "\tx", "\rx", "\nx", "'x", "'=x", "''=x",
            "", "ordinary", "model.layers.0.weight", "comma,here", 'quote"here', "line\nbreak", "模型🙂",
        ]
        path = self.tmp / "names.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fieldnames(["name", "typed", "text"]))
            w.writeheader()
            for name in names:
                w.writerow(encode_row({"name": name, "typed": -2, "text": "-2"}))
        raw = raw_rows(path)
        for name, row in zip(names, raw):
            escaped = "'" + name if name and name[0] in "=+-@\t\r\n'" else name
            self.assertEqual(row["name"], escaped)
            self.assertEqual(row[ENCODING_COLUMN], ENCODING_MARKER)
            self.assertEqual(row["text"], "'-2")
            self.assertEqual(row["typed"], "-2")
        self.assertEqual([r["name"] for r in plots.read_rows(path)], names)
        self.assertEqual(list(rank.read_delta_csv(path)), names)
        self.assertEqual(load_scorer().read_ranking(path), names)
        self.assertEqual(encode_row({"typed": -2})["typed"], -2)
        self.assertEqual(encode_row({"typed": -2.5})["typed"], -2.5)

    def test_empty_writers_keep_the_marker_header(self):
        writers = (
            lambda path: inventory.write_csv([], path),
            lambda path: rank.write_csv([], path),
            lambda path: delta.scan(None, None, [], path, k=1, log=lambda _: None),
        )
        for i, writer in enumerate(writers):
            path = self.tmp / f"empty-{i}.csv"
            writer(path)
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames[-1], ENCODING_COLUMN)
                self.assertEqual(list(reader), [])
            self.assertEqual(plots.read_rows(path), [])
            self.assertEqual(rank.read_delta_csv(path), {})
            self.assertEqual(load_scorer().read_ranking(path), [])

    def test_decoder_only_removes_an_escape_from_known_marked_prefixes(self):
        self.assertEqual(decode_row({"name": "'ordinary", ENCODING_COLUMN: ENCODING_MARKER}),
                         {"name": "'ordinary"})
        self.assertEqual(decode_row({"name": "'=x", ENCODING_COLUMN: "unknown"})["name"], "'=x")

    def test_legacy_unmarked_rows_are_not_decoded(self):
        delta_path = self.tmp / "legacy-delta.csv"
        delta_path.write_text("name,effective_rank,sv_001\n'=legacy,1,2\n", encoding="utf-8")
        self.assertEqual(rank.read_delta_csv(delta_path), {"'=legacy": Measurements(1.0, 2.0)})
        self.assertEqual(plots.read_rows(delta_path)[0]["name"], "'=legacy")

        ranking_path = self.tmp / "legacy-ranking.csv"
        ranking_path.write_text("rank,tier,name\n1,T4,'=legacy\n", encoding="utf-8")
        self.assertEqual(load_scorer().read_ranking(ranking_path), ["'=legacy"])

    def test_standalone_scorer_works_isolated_and_restores_marked_names(self):
        name = "=model.layers.0.weight"
        ranking_path = self.tmp / "ranking.csv"
        rank.write_csv([RankedRow(1, "T4", name, 0, "down_proj", CHANGED, 1.0, 2.0, "ordinary")], ranking_path)
        key_path = self.tmp / "key.txt"
        key_path.write_text(name + "\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--ranking", str(ranking_path), "--key", str(key_path)],
            cwd=self.tmp, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn(name, result.stdout)
        self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
