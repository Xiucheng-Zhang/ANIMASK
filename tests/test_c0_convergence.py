"""Unit tests for persona_probe.c0_convergence (pure computation).

    python3 tests/test_c0_convergence.py
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask import persona_probe as pp                                  # noqa: E402
from animask.persona_probe import c0_convergence as c0             # noqa: E402


def card(mode, curve, actions, goals):
    return {"tension_resolution": {"mode": mode, "how": ""},
            "conflict_curve": curve, "goals": goals, "relations": [],
            "transgressions": {}, "decision_actions": actions}


class TestMath(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(c0.entropy(["a", "a", "a"]), 0.0)
        self.assertAlmostEqual(c0.entropy(["a", "b"]), 1.0)
        self.assertAlmostEqual(c0.entropy(["a", "b", "c", "d"]), 2.0)

    def test_jsd(self):
        self.assertEqual(c0.jsd({}, {}), 0.0)
        self.assertEqual(c0.jsd({"a": 1}, {}), 1.0)
        self.assertAlmostEqual(c0.jsd({"a": 1}, {"a": 3}), 0.0)
        self.assertAlmostEqual(c0.jsd({"a": 1}, {"b": 1}), 1.0)
        mid = c0.jsd({"a": 1, "b": 1}, {"a": 1})
        self.assertTrue(0 < mid < 1)

    def test_card_distance(self):
        a = {"mode": "m", "curve": "c", "actions": {"x": 1},
             "goal_rate": 0.5}
        self.assertEqual(c0.card_distance(a, dict(a)), 0.0)
        b = {"mode": "n", "curve": "d", "actions": {"y": 1},
             "goal_rate": 1.0}
        self.assertAlmostEqual(c0.card_distance(a, b), (1 + 1 + 1 + .5) / 4)

    def test_story_card_seed_average(self):
        cards = [{"mode": "m", "curve": "c", "actions": {"x": 2},
                  "goal_rate": 0.0},
                 {"mode": "m", "curve": "d", "actions": {"x": 1, "y": 1},
                  "goal_rate": 1.0},
                 {"mode": "n", "curve": "c", "actions": {},
                  "goal_rate": None}]
        sc = c0.story_card(cards)
        self.assertEqual(sc["mode"], "m")
        self.assertEqual(sc["curve"], "c")
        self.assertAlmostEqual(sc["actions"]["x"], 1.0)
        self.assertAlmostEqual(sc["goal_rate"], 0.5)


class TestCompute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ANIMASK_ROOT"] = self.tmp.name
        self.root = Path(self.tmp.name)

    def tearDown(self):
        os.environ.pop("ANIMASK_ROOT", None)
        self.tmp.cleanup()

    def _story(self, sid, sim_card_obj, canon_card_obj, tag="fb_r1"):
        pack = sid + "__oracle"
        run = self.root / "results" / "runs" / pack
        run.mkdir(parents=True, exist_ok=True)
        (run / f"c3_scoreboard_{tag}.json").write_text(json.dumps(
            {"pack": pack, "tag": tag, "rule": {}, "card": sim_card_obj}))
        (run / f"run_meta_{tag}.json").write_text(json.dumps(
            {"models": {"actor": "gpt-5.5"},
             "args": {"persona_level": "P3"}}))
        cdir = self.root / "data" / "cache" / "canon_cards"
        cdir.mkdir(parents=True, exist_ok=True)
        if canon_card_obj is not None:
            (cdir / f"{sid}.json").write_text(json.dumps(
                {"sid": sid, "card": canon_card_obj}))

    def test_collapse_detected(self):
        # canon: three different modes; sim: everything dissolves
        modes = ["resolved_as_posed", "unresolved", "dissolved"]
        for i, m in enumerate(modes):
            self._story(f"s{i}",
                        card("dissolved", "de_escalates",
                             {"cooperate": 2}, {"H": "achieved"}),
                        card(m, "escalates",
                             {"confront": 2}, {"H": "failed"}))
        res = c0.compute("fb")
        g = res["groups"]["gpt-5.5|P3"]
        self.assertEqual(g["simulation"]["mode_entropy_bits"], 0.0)
        self.assertAlmostEqual(g["canon"]["mode_entropy_bits"],
                               math.log2(3), places=3)
        self.assertEqual(g["convergence_ratio"]["mode_entropy_bits"], 0.0)
        att = g["attractor"]
        self.assertEqual(att["pattern"],
                         {"mode": "dissolved", "curve": "de_escalates"})
        self.assertEqual(att["sim_share"], 1.0)
        self.assertEqual(att["canon_share"], 0.0)

    def test_identical_sides_ratio_one(self):
        for i, m in enumerate(["resolved_as_posed", "unresolved"]):
            same = card(m, "sustains", {"pursue": 1}, {"H": "achieved"})
            self._story(f"t{i}", same, same)
        res = c0.compute("fb")
        g = res["groups"]["gpt-5.5|P3"]
        self.assertEqual(g["convergence_ratio"]["mode_entropy_bits"], 1.0)
        self.assertEqual(
            g["convergence_ratio"]["mean_pairwise_distance"], 1.0)

    def test_unpaired_story_excluded(self):
        self._story("p0", card("dissolved", "none", {}, {}),
                    card("unresolved", "none", {}, {}))
        self._story("p1", card("dissolved", "none", {}, {}), None)
        res = c0.compute("fb")
        g = res["groups"]["gpt-5.5|P3"]
        self.assertEqual(g["paired_stories"], ["p0"])

    def test_markdown(self):
        self._story("m0", card("dissolved", "none", {"other": 1},
                               {"H": "partial"}),
                    card("unresolved", "escalates", {"confront": 1},
                         {"H": "achieved"}))
        res = c0.compute("fb")
        md = c0.to_markdown(res)
        self.assertIn("convergence", md.lower())
        self.assertIn("attractor", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
