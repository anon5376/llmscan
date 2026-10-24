import contextlib
import csv
import importlib.util
import io
import itertools
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import tiny_llama, write_safetensors
from llmscan import cli, core, inventory, rank
from llmscan.inventory import CHANGED, DTYPE_MISMATCH, IDENTICAL, ONLY_BASE, ONLY_SUSPECT, SHAPE_MISMATCH, InventoryRow
from llmscan.names import kind_of, layer_of

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
        self.assertEqual(list(rows[0]), rank.FIELDS)
        self.assertEqual(len(rows), len(self.changed))  # identical + dtype-only re-save are not ranked
        self.assertEqual((rows[0]["tier"], rows[0]["name"]), ("T2", layer_name(0, "mlp.gate_proj")))
        self.assertEqual({r["tier"] for r in rows[1:]}, {"T4"})
        self.assertEqual(sum(layer_name(0, "mlp.gate_proj") in line for line in out.splitlines()), 1)
        shown = [line for line in out.splitlines() if line.strip().startswith(("1 ", "2 ", "3 ", "4 "))]
        self.assertEqual(len(shown), 3)

    def test_ranking_is_written_when_the_core_is_a_stub(self):
        code, out, err = run_cli(*self.args)
        self.assertEqual(code, 2)
        self.assertIn("analyze_delta is not written yet", err)
        self.assertEqual(len(read_ranking(self.tmp / "out" / "ranking.csv")), len(self.changed))
        self.assertIn("ranking: T1=0  T2=1  T3=0  T4=8", out)

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

    def test_failed_run_does_not_leave_a_previous_ranking(self):
        self.assertEqual(run_cli(*self.args, "--inventory-only")[0], 0)
        ranking = self.tmp / "out" / "ranking.csv"
        self.assertTrue(ranking.is_file())
        with self.assertRaises(SystemExit) as cm:
            run_cli("--base", str(self.tmp / "nope.safetensors"), "--suspect", str(self.tmp / "suspect.safetensors"),
                    "--out", str(self.tmp / "out"), "--inventory-only")
        self.assertNotEqual(cm.exception.code, 0)
        self.assertFalse(ranking.exists())


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
