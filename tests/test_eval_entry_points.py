"""Evaluation entry points and aggregates: run_eval parses its arguments and
discovers finished runs, curve_compare reports Holm-adjusted p-values, and
the five-label aggregate pools points per actor. No LLM calls."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask import persona_probe as pp  # noqa: E402
from animask.persona_probe import b1v2_decisions, curve_compare, run_eval  # noqa: E402
from animask.persona_probe.story_curves import SCORES, N_CHECKPOINTS  # noqa: E402


class TempRoot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ANIMASK_ROOT"] = self.tmp.name
        self.bw = Path(self.tmp.name) / "results" / "runs"
        self.bw.mkdir(parents=True)

    def tearDown(self):
        os.environ.pop("ANIMASK_ROOT", None)
        self.tmp.cleanup()

    def write_run(self, pack, tag, status="done"):
        d = self.bw / pack
        d.mkdir(exist_ok=True)
        (d / f"run_meta_{tag}.json").write_text(json.dumps(
            {"status": status, "models": {"actor": "gpt-5.5"}}))
        return d


class TestRunEval(TempRoot):
    def test_help_parses(self):
        with self.assertRaises(SystemExit) as cm:
            run_eval.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_discovers_finished_runs_of_the_batch_only(self):
        self.write_run("s1__oracle", "mybatch_r1")
        self.write_run("s1__oracle", "mybatch_claudesonnet5_r1")
        self.write_run("s2__oracle", "mybatch_r1", status="running")
        self.write_run("s3__oracle", "otherbatch_r1")
        runs = run_eval.discover_done_runs("mybatch")
        self.assertEqual(sorted(runs), [("s1__oracle", "mybatch_claudesonnet5_r1"),
                                        ("s1__oracle", "mybatch_r1")])


class TestCurveCompareHolm(unittest.TestCase):
    def test_holm_column_present_and_monotone(self):
        import random
        rng = random.Random(1)
        sims, canons = {}, {}
        for i in range(8):
            sims[f"s{i}"] = {s: [rng.uniform(1, 7) for _ in range(N_CHECKPOINTS)]
                             for s in SCORES}
            canons[f"s{i}"] = {s: [v + (2.0 if s == SCORES[0] else 0.0)
                                   for v in sims[f"s{i}"][s]] for s in SCORES}
        res = curve_compare.compare_group(sims, canons)
        dev = res["deviation"]
        for s in SCORES:
            self.assertIn("p_holm", dev[s])
            self.assertGreaterEqual(dev[s]["p_holm"], dev[s]["p_permutation"])
            self.assertLessEqual(dev[s]["p_holm"], 1.0)
        md = curve_compare.to_markdown({"batchtag": "t", "generated_at": "now",
                                        "n_canon": 8, "groups": {"g": res}})
        self.assertIn("p(Holm)", md)


class TestB1v2Aggregate(TempRoot):
    def point(self, actor, implied, implied_set, actual, nocard):
        return {"actor": actor, "implied": implied, "implied_set": implied_set,
                "actual": actual, "nocard": nocard, "round": 0,
                "cell": b1v2_decisions.cell_of(implied_set, actual, nocard)}

    def write_v2(self, pack, tag, actor_model, points):
        d = self.write_run(pack, tag)
        (d / f"b1v2_decisions_{tag}.json").write_text(json.dumps(
            {"pack": pack, "tag": tag, "actor_model": actor_model,
             "n_rounds": 3, "points": points}))

    def test_point_flags(self):
        f = b1v2_decisions.point_flags(
            self.point("A", "oppose", ["oppose", "cross"], "oppose", "press"))
        self.assertTrue(f["adherent"] and f["disagreement"] and f["persona_wins"])
        self.assertFalse(f["overridden"])
        self.assertEqual(f["shift"], 1)   # oppose(3) - press(2)
        f = b1v2_decisions.point_flags(
            self.point("A", "cross", ["cross"], "press", "press"))
        self.assertTrue(f["overridden"])
        self.assertFalse(f["adherent"])

    def test_aggregate_rates_and_gate(self):
        pts = [self.point("Hero", "press", ["press"], "press", "press"),      # adherent, no disagreement
               self.point("Hero", "oppose", ["oppose"], "oppose", "press"),   # persona wins
               self.point("Hero", "cross", ["cross"], "press", "press"),      # overridden
               self.point("Foe", "yield", ["yield"], "yield", "hold")]        # gate-failed role
        self.write_v2("s1__oracle", "mybatch_r1", "gpt-5.5", pts)
        self.write_v2("s2__oracle", "mybatch_r1", "gpt-5.5",
                      [self.point("Hero", "hold", ["hold"], "hold", "hold")])
        agg_dir = pp.aggregate_dir()
        agg_dir.mkdir(parents=True)
        (agg_dir / "facets_mybatch.json").write_text(json.dumps({"runs": [
            {"pack": "s1__oracle", "per_role": {"Hero": {"gate": {"pass": True}},
                                                "Foe": {"gate": {"pass": False}}}},
            {"pack": "s2__oracle", "per_role": {"Hero": {"gate": {"pass": True}}}}]}))
        out = b1v2_decisions.aggregate("mybatch")
        g = out["groups"]["gpt-5.5"]
        self.assertTrue(out["gate_applied"])
        self.assertEqual(g["n_points"], 4)
        self.assertEqual(g["n_gate_excluded"], 1)
        self.assertEqual(g["n_packs"], 2)
        self.assertAlmostEqual(g["adherence"]["stat"], 3 / 4)
        self.assertAlmostEqual(g["disagreement_share"]["stat"], 2 / 4)
        self.assertAlmostEqual(g["efficacy"]["stat"], 1 / 4)
        self.assertAlmostEqual(g["persona_wins_at_disagreements"], 0.5)
        self.assertEqual(out["n_boot"], 2000)
        self.assertTrue((agg_dir / "b1v2_mybatch.json").exists())
        self.assertIn("| gpt-5.5 |", (agg_dir / "b1v2_mybatch.md").read_text())

    def test_cli_flag(self):
        self.write_v2("s1__oracle", "mybatch_r1", "gpt-5.5",
                      [self.point("Hero", "press", ["press"], "press", "hold")])
        pp.aggregate_dir().mkdir(parents=True)
        out = b1v2_decisions.main(["--aggregate", "mybatch"])
        self.assertIn("gpt-5.5", out["groups"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
