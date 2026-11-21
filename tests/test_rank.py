import contextlib
import csv
import importlib.util
import io
import itertools
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import tiny_llama, write_safetensors
from llmscan import cli, core, inventory, rank
from llmscan.csv_safety import ENCODING_COLUMN
from llmscan.inventory import CHANGED, DTYPE_MISMATCH, IDENTICAL, ONLY_BASE, ONLY_SUSPECT, SHAPE_MISMATCH, InventoryRow
from llmscan.names import kind_of, layer_of
from test_hardening import link_or_skip

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_blind.py"


def row(name, status=CHANGED):
    return InventoryRow(name, layer_of(name), kind_of(name), status, "BF16", "BF16", "8x8", "8x8", 64)


def layer_name(i, kind="mlp.down_proj"):
    return f"model.layers.{i}.{kind}.weight"


def order(ranked):
    return [r.name for r in ranked]


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def read_ranking(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestTiers(unittest.TestCase):
    def test_structure_then_location_then_rest_and_identical_unranked(self):
        rows = [
            row("model.embed_tokens.weight", IDENTICAL),
            row(layer_name(0, "mlp.down_proj")),
            row(layer_name(0, "self_attn.o_proj")),
            row(layer_name(0, "self_attn.q_proj")),
            row(layer_name(0, "self_attn.v_proj"), DTYPE_MISMATCH),
            row(layer_name(0, "mlp.up_proj"), SHAPE_MISMATCH),
            row(layer_name(0, "self_attn.trigger"), ONLY_SUSPECT),
            row("lm_head.weight", ONLY_BASE),
        ]
        ranked = rank.rank_tensors(rows)
        self.assertEqual([r.tier for r in ranked], ["T1"] * 4 + ["T2", "T4", "T4"])
        self.assertNotIn("model.embed_tokens.weight", order(ranked))
        self.assertEqual(ranked[4].name, layer_name(0, "self_attn.q_proj"))  # q_proj outranks o_proj / down_proj
        self.assertEqual([r.rank for r in ranked], list(range(1, 8)))
        self.assertTrue(all(r.reason for r in ranked))
        reasons = {r.status: r.reason for r in ranked}
        self.assertIn("shape differs", reasons[SHAPE_MISMATCH])
        self.assertIn("not in base", reasons[ONLY_SUSPECT])
        self.assertIn("missing from suspect", reasons[ONLY_BASE])

    def test_q_proj_outranks_even_a_huge_expected_kind_outlier(self):
        rows = [row(layer_name(i, "self_attn.o_proj")) for i in range(5)] + [row(layer_name(0, "self_attn.q_proj"))]
        delta = {layer_name(i, "self_attn.o_proj"): (1.0, 5.0) for i in range(4)}
        delta[layer_name(4, "self_attn.o_proj")] = (99.0, 5.0)
        delta[layer_name(0, "self_attn.q_proj")] = (1.0, 1.0)
        ranked = rank.rank_tensors(rows, delta)
        self.assertEqual([r.tier for r in ranked[:2]], ["T2", "T3"])
        self.assertEqual(ranked[1].name, layer_name(4, "self_attn.o_proj"))

    def test_down_proj_effective_rank_outlier_outranks_other_down_proj(self):
        rows = [row(layer_name(i)) for i in range(6)]
        ranks = [1.0, 1.2, 1.1, 9.0, 1.0, 1.3]
        delta = {layer_name(i): (er, 2.0) for i, er in enumerate(ranks)}
        ranked = rank.rank_tensors(rows, delta)
        self.assertEqual(ranked[0].name, layer_name(3))
        self.assertEqual([r.tier for r in ranked], ["T3"] + ["T4"] * 5)
        self.assertIn("effective_rank 9", ranked[0].reason)
        self.assertEqual([r.effective_rank for r in ranked[1:]], sorted(ranks[:3] + ranks[4:], reverse=True))

    def test_outlier_is_relative_to_the_kinds_median_not_a_fixed_rank(self):
        rows = [row(layer_name(i, "mlp.down_proj")) for i in range(5)] + [row(layer_name(i, "self_attn.o_proj")) for i in range(5)]
        delta = {layer_name(i, "mlp.down_proj"): (3.0, 1.0) for i in range(5)}  # a rank-3 edit everywhere: honest
        delta.update({layer_name(i, "self_attn.o_proj"): (1.0, 1.0) for i in range(5)})
        self.assertEqual({r.tier for r in rank.rank_tensors(rows, delta)}, {"T4"})
        delta[layer_name(2, "self_attn.o_proj")] = (4.0, 1.0)  # 4x the o_proj median, still below the down_proj rank
        flagged = [r for r in rank.rank_tensors(rows, delta) if r.tier == "T3"]
        self.assertEqual(order(flagged), [layer_name(2, "self_attn.o_proj")])

    def test_factor_is_adjustable(self):
        rows = [row(layer_name(i)) for i in range(5)]
        delta = {layer_name(i): (1.0 if i else 4.0, 1.0) for i in range(5)}
        self.assertEqual(rank.rank_tensors(rows, delta)[0].tier, "T3")
        self.assertEqual(rank.rank_tensors(rows, delta, factor=5.0)[0].tier, "T4")

    def test_too_few_tensors_for_a_median_means_no_outlier(self):
        rows = [row(layer_name(i)) for i in range(2)]
        ranked = rank.rank_tensors(rows, {layer_name(0): (1.0, 1.0), layer_name(1): (50.0, 1.0)})
        self.assertEqual([r.tier for r in ranked], ["T4", "T4"])

    def test_sv_001_judges_tensors_without_effective_rank(self):
        rows = [row(layer_name(i)) for i in range(5)]
        delta = {layer_name(i): (None, 2.0) for i in range(4)}
        delta[layer_name(4)] = (None, 20.0)
        ranked = rank.rank_tensors(rows, delta)
        self.assertEqual((ranked[0].name, ranked[0].tier), (layer_name(4), "T3"))
        self.assertIn("sv_001", ranked[0].reason)

    def test_within_tier_order_is_rank_then_sv_then_natural_name(self):
        rows = [row(layer_name(i, "mlp.gate_proj")) for i in (10, 2, 1, 0, 3)]
        delta = {layer_name(0, "mlp.gate_proj"): (2.0, 1.0), layer_name(1, "mlp.gate_proj"): (2.0, 7.0),
                 layer_name(3, "mlp.gate_proj"): (5.0, 0.5)}
        # layers 2 and 10 have no stats: after the ones with stats, in natural order (2 before 10)
        self.assertEqual([layer_of(n) for n in order(rank.rank_tensors(rows, delta))], [3, 1, 0, 2, 10])

    def test_inventory_only_ranking(self):
        rows = [row(layer_name(i)) for i in range(3)] + [row(layer_name(0, "input_layernorm"))]
        ranked = rank.rank_tensors(rows)
        self.assertEqual([r.tier for r in ranked], ["T2", "T4", "T4", "T4"])
        self.assertTrue(all(r.effective_rank is None for r in ranked))
        self.assertIn("no ΔW stats", ranked[1].reason)
        self.assertEqual(order(ranked)[1:], [layer_name(i) for i in range(3)])

    def test_gguf_and_moe_names_count_as_expected_kinds(self):
        expected = [
            "blk.3.ffn_down.weight", "blk.3.attn_output.weight", "token_embd.weight", "blk.3.ffn_down_exps.weight",
            "blk.3.ffn_down_shexp.weight", "model.layers.2.mlp.experts.5.down_proj.weight",
            "model.layers.2.mlp.shared_expert.down_proj.weight", "model.layers.2.mlp.experts.down_proj",
        ]
        unexpected = ["blk.3.attn_q.weight", "blk.3.ffn_gate.weight", "lm_head.weight", "model.layers.0.self_attn.q_proj.bias"]
        ranked = {r.name: r.tier for r in rank.rank_tensors([row(n) for n in expected + unexpected])}
        self.assertEqual({ranked[n] for n in expected}, {"T4"})
        self.assertEqual({ranked[n] for n in unexpected}, {"T2"})

    def test_expected_kinds_are_overridable(self):
        rows = [row(layer_name(0, "self_attn.q_proj")), row(layer_name(0, "self_attn.o_proj"))]
        ranked = rank.rank_tensors(rows, expected=rank.parse_kinds("q_proj, k_proj"))
        self.assertEqual([(r.kind, r.tier) for r in ranked], [("o_proj", "T2"), ("q_proj", "T4")])
        with self.assertRaises(ValueError):
            rank.parse_kinds(" , ")

    def test_non_finite_stats_are_never_consistent_and_order_is_deterministic(self):
        rows = [row(layer_name(i)) for i in range(4)]
        stats = {layer_name(0): (1.0, 1.0), layer_name(1): (math.nan, 9.0), layer_name(2): (1.0, 4.0), layer_name(3): (1.0, 3.0)}
        orders = set()
        for perm in itertools.permutations(rows):
            ranked = rank.rank_tensors(list(perm), stats)
            orders.add(tuple(r.name for r in ranked))
        self.assertEqual(orders, {(layer_name(1), layer_name(2), layer_name(3), layer_name(0))})
        ranked = rank.rank_tensors(rows, stats)
        self.assertEqual([r.tier for r in ranked], ["T3", "T4", "T4", "T4"])
        self.assertIn("stats unusable", ranked[0].reason)
        self.assertIn("not finite", ranked[0].reason)

    def test_infinity_does_not_contaminate_the_median(self):
        rows = [row(layer_name(i)) for i in range(4)]
        stats = {layer_name(i): (v, 1.0) for i, v in enumerate([1.0, 1.0, 9.0, math.inf])}
        ranked = {r.name: r for r in rank.rank_tensors(rows, stats)}
        self.assertEqual(ranked[layer_name(2)].tier, "T3")  # 9 vs the finite median 1
        self.assertIn("9.0x", ranked[layer_name(2)].reason)
        self.assertEqual(ranked[layer_name(3)].tier, "T3")
        self.assertIn("not finite", ranked[layer_name(3)].reason)
        self.assertEqual({ranked[layer_name(0)].tier, ranked[layer_name(1)].tier}, {"T4"})

    def test_row_with_any_non_finite_stat_is_out_of_every_median(self):
        rows = [row(layer_name(i)) for i in range(4)]
        stats = dict(zip((layer_name(i) for i in range(4)), [(1.0, 1.0), (1.0, 1.0), (9.0, 1.0), (100.0, math.inf)]))
        ranked = {r.name: r for r in rank.rank_tensors(rows, stats)}
        self.assertEqual(ranked[layer_name(2)].tier, "T3")  # 9 vs the median 1 of the three usable rows
        self.assertIn("9.0x", ranked[layer_name(2)].reason)
        self.assertEqual(ranked[layer_name(3)].tier, "T3")
        self.assertIn("stats unusable", ranked[layer_name(3)].reason)

    def test_missing_or_invalid_stats_are_t3_after_the_genuine_outliers(self):
        rows = [row(layer_name(i)) for i in range(6)]
        peers = {layer_name(0): (1.0, 1.0), layer_name(1): (1.0, 1.0), layer_name(2): (1.0, 1.0), layer_name(3): (9.0, 1.0)}
        cases = {
            "negative effective_rank": (-1.0, 100.0),
            "zero effective_rank": (0.0, 100.0),
            "negative sv": (1.0, -1.0),
            "zero sv": (1.0, 0.0),
            "blank": (None, None),
        }
        for label, bad in cases.items():
            with self.subTest(label):
                ranked = rank.rank_tensors(rows[:5], {**peers, layer_name(4): bad})
                self.assertEqual([r.tier for r in ranked], ["T3", "T3", "T4", "T4", "T4"])
                self.assertEqual([r.name for r in ranked[:2]], [layer_name(3), layer_name(4)])
                self.assertTrue(ranked[1].reason.startswith("stats unusable: "), ranked[1].reason)
        with self.subTest("missing delta row"):
            ranked = rank.rank_tensors(rows[:5], peers)  # no row for layer 4
            self.assertEqual([(r.name, r.tier) for r in ranked[:2]], [(layer_name(3), "T3"), (layer_name(4), "T3")])
            self.assertIn("no ΔW stats", ranked[1].reason)

    def test_blank_effective_rank_with_a_usable_sv_is_still_judged(self):
        rows = [row(layer_name(i)) for i in range(4)]
        stats = {layer_name(i): (None, v) for i, v in enumerate([1.0, 1.0, 1.0, 9.0])}
        self.assertEqual([r.tier for r in rank.rank_tensors(rows, stats)], ["T3", "T4", "T4", "T4"])

    def test_inventory_only_t4_is_not_an_unusable_stats_case(self):
        ranked = rank.rank_tensors([row(layer_name(0))])
        self.assertEqual(ranked[0].tier, "T4")
        self.assertIn("no ΔW stats", ranked[0].reason)

    def test_a_scan_that_ran_but_wrote_no_rows_is_not_inventory_only(self):
        rows = [row(layer_name(i)) for i in range(3)]
        for delta in ({}, {"unrelated.weight": (1.0, 1.0)}):
            with self.subTest(delta=delta):
                ranked = rank.rank_tensors(rows, delta)
                self.assertEqual([r.tier for r in ranked], ["T3"] * 3)
                self.assertTrue(all(r.reason.startswith("stats unusable: ") for r in ranked), ranked[0].reason)
        self.assertEqual([r.tier for r in rank.rank_tensors(rows, None)], ["T4"] * 3)

    def test_changed_tensors_the_scanner_skips_are_t3_not_scanned_after_the_outliers(self):
        def skipped(name, base_dtype="F32", suspect_dtype=None, shape="2x2x2"):
            return InventoryRow(name, layer_of(name), kind_of(name), CHANGED, base_dtype, suspect_dtype or base_dtype,
                                shape, shape, 8)

        peers = {layer_name(i): (1.0, 1.0) for i in range(3)} | {layer_name(3): (9.0, 1.0)}
        rows = [row(layer_name(i)) for i in range(4)] + [
            skipped("model.layers.0.mlp.experts.down_proj"),
            skipped("model.layers.1.mlp.experts.down_proj", shape="8"),
            skipped("model.layers.2.mlp.experts.down_proj", shape=""),
            skipped("blk.0.ffn_down.weight", "Q4_0", shape="32x2"),
            skipped("blk.1.ffn_down.weight", "F16", "Q8_0", shape="32x2"),
            skipped("model.layers.0.self_attn.q_proj.weight", shape="2x2x2"),
        ]
        ranked = rank.rank_tensors(rows, peers)
        tiers = {r.name: r for r in ranked}
        self.assertEqual([r.tier for r in ranked], ["T2", "T3"] + ["T3"] * 5 + ["T4"] * 3)
        self.assertEqual(ranked[1].name, layer_name(3))  # the genuine outlier leads the unusable rows
        for name, why in [("model.layers.0.mlp.experts.down_proj", "shape 2x2x2 is not 2-D"),
                          ("model.layers.1.mlp.experts.down_proj", "shape 8 is not 2-D"),
                          ("model.layers.2.mlp.experts.down_proj", "shape scalar is not 2-D"),
                          ("blk.0.ffn_down.weight", "dtype Q4_0 is not float"),
                          ("blk.1.ffn_down.weight", "dtype Q8_0 is not float")]:
            self.assertEqual(tiers[name].tier, "T3", name)
            self.assertTrue(tiers[name].reason.startswith(f"stats unusable: not scanned ({why})"), tiers[name].reason)
        self.assertEqual(tiers["model.layers.0.self_attn.q_proj.weight"].tier, "T2")  # unexpected kind stays T2
        self.assertEqual([r.tier for r in rank.rank_tensors(rows, None)].count("T3"), 0)

    def test_names_with_equal_natural_keys_have_one_order(self):
        names = ["added.1.weight", "added.01.weight", "added.001.weight"]
        rows = [row(n, ONLY_SUSPECT) for n in names]
        orders = {tuple(r.name for r in rank.rank_tensors(list(p))) for p in itertools.permutations(rows)}
        self.assertEqual(len(orders), 1)

    def test_outlier_factor_must_be_finite_and_positive(self):
        for bad in (math.nan, math.inf, 0.0, -1.0):
            with self.assertRaises(ValueError, msg=str(bad)):
                rank.rank_tensors([row(layer_name(0))], factor=bad)
        for bad in ("nan", "inf", "0", "-2", "x"):
            with self.assertRaises(ValueError, msg=bad):
                rank.parse_factor(bad)
        self.assertEqual(rank.parse_factor("2.5"), 2.5)


class TestCli(unittest.TestCase):
    """Real (tiny) checkpoints through the CLI; the core is a stub or a stand-in returning canned stats."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        base = tiny_llama(np.random.default_rng(3), layers=4)
        suspect = dict(base)
        self.changed = []
        for i in range(4):
            for kind in ("mlp.down_proj", "self_attn.o_proj"):
                self.changed.append(layer_name(i, kind))
        self.changed.append(layer_name(0, "mlp.gate_proj"))
        for name in self.changed:
            dt, w = base[name]
            suspect[name] = (dt, w + 1 / 16)
        # dtype-only re-save: same values stored as F16, must not be ranked
        suspect[layer_name(1, "self_attn.q_proj")] = ("F16", base[layer_name(1, "self_attn.q_proj")][1])
        write_safetensors(self.tmp / "base.safetensors", base)
        write_safetensors(self.tmp / "suspect.safetensors", suspect)
        self.args = ["--base", str(self.tmp / "base.safetensors"), "--suspect", str(self.tmp / "suspect.safetensors"),
                     "--out", str(self.tmp / "out")]

    def test_inventory_only_writes_ranking_and_prints_top_n(self):
        code, out, _ = run_cli(*self.args, "--inventory-only", "--top", "3")
        self.assertEqual(code, 0, out)
        rows = read_ranking(self.tmp / "out" / "ranking.csv")
        self.assertEqual(list(rows[0]), rank.FIELDS + [ENCODING_COLUMN])
        self.assertEqual(len(rows), len(self.changed))  # identical + dtype-only re-save are not ranked
        self.assertEqual((rows[0]["tier"], rows[0]["name"]), ("T2", layer_name(0, "mlp.gate_proj")))
        self.assertEqual({r["tier"] for r in rows[1:]}, {"T4"})
        self.assertEqual(sum(layer_name(0, "mlp.gate_proj") in line for line in out.splitlines()), 1)
        shown = [line for line in out.splitlines() if line.strip().startswith(("1 ", "2 ", "3 ", "4 "))]
        self.assertEqual(len(shown), 3)

    def test_ranking_is_written_when_the_core_is_not_implemented(self):
        with mock.patch.object(core, "analyze_delta", side_effect=NotImplementedError("analyze_delta is not written yet")):
            code, out, err = run_cli(*self.args)
        self.assertEqual(code, 2)
        self.assertIn("analyze_delta is not written yet", err)
        self.assertEqual(len(read_ranking(self.tmp / "out" / "ranking.csv")), len(self.changed))
        self.assertIn("ranking: T1=0  T2=1  T3=0  T4=8", out)

    def test_a_changed_tensor_the_scanner_skips_is_never_t4_after_a_full_scan(self):
        name = "model.layers.0.mlp.experts.down_proj"
        base = {name: ("F32", np.zeros((2, 2, 2)))}
        write_safetensors(self.tmp / "b3.safetensors", base)
        write_safetensors(self.tmp / "s3.safetensors", {name: ("F32", np.ones((2, 2, 2)))})
        args = ["--base", str(self.tmp / "b3.safetensors"), "--suspect", str(self.tmp / "s3.safetensors")]
        code, out, _ = run_cli(*args, "--out", str(self.tmp / "full"))
        self.assertEqual(code, 0, out)
        self.assertEqual(read_ranking(self.tmp / "full" / "delta.csv"), [])  # the scan ran and wrote no rows
        (row_,) = read_ranking(self.tmp / "full" / "ranking.csv")
        self.assertEqual(row_["tier"], "T3")
        self.assertIn("stats unusable: not scanned (shape 2x2x2 is not 2-D)", row_["reason"])
        code, out, _ = run_cli(*args, "--out", str(self.tmp / "inv"), "--inventory-only")
        self.assertEqual(code, 0, out)
        (row_,) = read_ranking(self.tmp / "inv" / "ranking.csv")
        self.assertEqual((row_["tier"], "(no ΔW stats)" in row_["reason"]), ("T4", True))

    def test_delta_stats_re_rank_after_a_successful_scan(self):
        ranks = iter(sorted(self.changed, key=inventory.natural_key))
        canned = {n: core.DeltaStats(np.array([2.0, 1.0]), 1.0) for n in self.changed}
        canned[layer_name(3, "mlp.down_proj")] = core.DeltaStats(np.array([2.0, 1.0]), 8.0)

        def fake_analyze(w_base, w_suspect, k):  # delta.scan visits changed tensors in natural name order
            return canned[next(ranks)]

        with mock.patch.object(core, "analyze_delta", fake_analyze):
            code, out, _ = run_cli(*self.args, "-k", "2")
        self.assertEqual(code, 0, out)
        rows = read_ranking(self.tmp / "out" / "ranking.csv")
        self.assertEqual([(r["tier"], r["name"]) for r in rows[:2]],
                         [("T2", layer_name(0, "mlp.gate_proj")), ("T3", layer_name(3, "mlp.down_proj"))])
        self.assertEqual(rows[1]["effective_rank"], "8")
        self.assertEqual({r["tier"] for r in rows[2:]}, {"T4"})
        self.assertIn("ranking.csv", out)

    def test_invalid_outlier_factor_is_rejected_before_any_work(self):
        for bad in ("nan", "inf", "0", "-1"):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                cli.main([*self.args, "--inventory-only", "--outlier-factor", bad])
            self.assertEqual(cm.exception.code, 2, bad)
        self.assertFalse((self.tmp / "out").exists())

    def test_failed_run_never_touches_a_users_ranking_csv(self):
        bad = ["--base", str(self.tmp / "nope.safetensors"), "--suspect", str(self.tmp / "suspect.safetensors"), "--inventory-only"]
        mine = self.tmp / "mine"
        mine.mkdir()
        (mine / "ranking.csv").write_text("my own notes\n", encoding="utf-8")
        link = self.tmp / "link"
        link.symlink_to(mine, target_is_directory=True)
        for out in (mine, link):
            with self.subTest(out=out.name):
                with self.assertRaises(SystemExit) as cm:
                    run_cli(*bad, "--out", str(out))
                self.assertNotEqual(cm.exception.code, 0)
                self.assertEqual((mine / "ranking.csv").read_text(encoding="utf-8"), "my own notes\n")
                with self.assertRaises(SystemExit):  # valid inputs are refused as well: the directory is not llmscan's
                    run_cli(*self.args[:4], "--out", str(out), "--inventory-only")
                self.assertEqual((mine / "ranking.csv").read_text(encoding="utf-8"), "my own notes\n")

    def test_ranking_csv_symlink_is_refused_and_its_target_untouched(self):
        out = self.tmp / "out"
        out.mkdir()
        target = self.tmp / "precious.txt"
        target.write_text("keep\n", encoding="utf-8")
        (out / "ranking.csv").symlink_to(target)
        with self.assertRaises(SystemExit):
            run_cli(*self.args, "--inventory-only")
        self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")

    def test_a_link_planted_at_ranking_csv_after_the_checks_is_replaced_not_followed(self):
        out, victim = self.tmp / "out", self.tmp / "victim.csv"
        victim.write_bytes(b"a checkpoint")

        def plant():
            (out / "ranking.csv").unlink(missing_ok=True)  # the intruder takes the name over
            link_or_skip((out / "ranking.csv").symlink_to, victim)

        real_mkdir = Path.mkdir
        def mkdir_then_plant(self_, *a, **kw):  # after the guards, before the ranking after the inventory
            real_mkdir(self_, *a, **kw)
            if self_ == out:
                plant()
        def analyze_then_plant(w_base, w_suspect, k):  # before the ranking after the scan
            plant()
            return core.DeltaStats(np.array([2.0, 1.0]), 1.0)

        with mock.patch.object(Path, "mkdir", mkdir_then_plant), mock.patch.object(core, "analyze_delta", analyze_then_plant):
            code, out_text, _ = run_cli(*self.args, "-k", "2")
        self.assertEqual(code, 0, out_text)
        self.assertEqual(victim.read_bytes(), b"a checkpoint")
        self.assertFalse((out / "ranking.csv").is_symlink())
        self.assertEqual(len(read_ranking(out / "ranking.csv")), len(self.changed))

    def test_ranking_order_does_not_depend_on_the_hash_seed(self):
        base = {"model.embed_tokens.weight": ("F32", np.zeros((4, 4), np.float32))}
        suspect = dict(base)
        for n in ("added.1.weight", "added.01.weight", "added.001.weight", "added.2.weight", "added.10.weight"):
            suspect[n] = ("F32", np.zeros((2, 2), np.float32))
        write_safetensors(self.tmp / "b.safetensors", base)
        write_safetensors(self.tmp / "s.safetensors", suspect)
        outputs = set()
        for seed in ("1", "2", "3", "4", "5"):
            out = self.tmp / f"seed{seed}"
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(SRC)}
            subprocess.run([sys.executable, "-m", "llmscan", "--base", str(self.tmp / "b.safetensors"),
                            "--suspect", str(self.tmp / "s.safetensors"), "--out", str(out), "--inventory-only"],
                           env=env, check=True, capture_output=True)
            outputs.add((out / "ranking.csv").read_bytes())
        self.assertEqual(len(outputs), 1)


def load_scorer():
    spec = importlib.util.spec_from_file_location("score_blind", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestScorer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.scorer = load_scorer()

    def score(self, ranked, key_text):
        with open(self.tmp / "ranking.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "tier", "name"])
            w.writerows((i, "T4", n) for i, n in enumerate(ranked, 1))
        (self.tmp / "key.txt").write_text(key_text, encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = self.scorer.main(["--ranking", str(self.tmp / "ranking.csv"), "--key", str(self.tmp / "key.txt")])
        return code, out.getvalue()

    def test_pass_when_top_n_is_exactly_the_key(self):
        code, out = self.score(["b", "a", "c", "d"], "# planted\na\n\nb  # layer 3\n")
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)
        self.assertIn("precision@2 = 1.000", out)
        self.assertIn("recall@10 = 1.000", out)

    def test_fail_when_a_decoy_displaces_a_key_tensor(self):
        code, out = self.score(["a", "x", "b", "c"], "a\nb\n")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)
        self.assertIn("precision@2 = 0.500", out)
        self.assertIn("recall@10 = 1.000", out)
        self.assertIn("rank 3  b", out)

    def test_key_tensor_missing_from_the_ranking(self):
        code, out = self.score(["a", "x"], "a\nzzz\n")
        self.assertEqual(code, 1)
        self.assertIn("not ranked  zzz", out)
        self.assertIn("recall@10 = 0.500", out)

    def test_recall_at_10_cuts_off_at_ten(self):
        ranked = [f"t{i}" for i in range(12)]
        code, out = self.score(ranked, "t0\nt11\n")
        self.assertEqual(code, 1)
        self.assertIn("recall@10 = 0.500", out)
        self.assertIn("precision@2 = 0.500", out)

    def test_duplicate_ranking_rows_neither_inflate_recall_nor_pass(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            self.score(["a", "a", "b"], "a\n")
        self.assertNotEqual(cm.exception.code, 0)
        self.assertEqual(out.getvalue(), "")

    def test_duplicate_key_lines_count_once(self):
        code, out = self.score(["a", "b"], "a\na\nb\nb\n")
        self.assertEqual(code, 0, out)
        self.assertIn("precision@2 = 1.000", out)
        self.assertIn("recall@10 = 1.000", out)

    def test_nothing_is_printed_before_both_files_are_read(self):
        out = io.StringIO()
        (self.tmp / "key.txt").write_text("secret_tensor\n")
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            self.scorer.main(["--ranking", str(self.tmp / "missing.csv"), "--key", str(self.tmp / "key.txt")])
        self.assertNotEqual(cm.exception.code, 0)
        self.assertEqual(out.getvalue(), "")
        self.assertNotIn("secret_tensor", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
