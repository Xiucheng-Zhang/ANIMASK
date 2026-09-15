"""Synthetic-fixture tests for persona_probe.c3_scoreboard. No real API
calls.

    python3 tests/test_c3_scoreboard.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask import persona_probe as pp                                  # noqa: E402
from animask.persona_probe import c3_scoreboard as c3              # noqa: E402

SID = "demo_01_26"
PACK = SID + "__oracle"
TAG = "fb_r1"
HERO, VILLAIN = "Hero-en", "Villain-en"

SIM_CARD = {
    "tension_resolution": {"mode": "dissolved", "how": "fizzled"},
    "conflict_curve": "de_escalates",
    "goals": {"Hero": "achieved", "Villain": "failed"},
    "relations": [{"pair": "Hero|Villain", "start": "negative",
                   "end": "neutral"}],
    "transgressions": {"theft": 0, "assault": 0, "coercion": 1,
                       "deception": 1, "betrayal": 0},
    "decision_actions": {"cooperate": 3, "concede": 2},
}
CANON_CARD = {
    "tension_resolution": {"mode": "resolved_as_posed", "how": "duel"},
    "conflict_curve": "escalates",
    "goals": {"Hero": "failed", "Villain": "achieved"},
    "relations": [{"pair": "Hero|Villain", "start": "negative",
                   "end": "negative"}],
    "transgressions": {"theft": 1, "assault": 2, "coercion": 0,
                       "deception": 1, "betrayal": 1},
    "decision_actions": {"confront": 3, "transgress": 2},
}


def build_fixture(root: Path):
    run = root / "results" / "runs" / PACK
    run.mkdir(parents=True, exist_ok=True)
    (run / f"transcript_{TAG}.json").write_text(json.dumps(
        [{"actor_type": "system", "code": "", "text": "opening SIMMARK"},
         {"actor_type": "role", "code": HERO, "text": "hero speaks"},
         {"actor_type": "error", "code": "", "text": "IGNORED"},
         {"actor_type": "role", "code": VILLAIN, "text": "villain speaks"}]))
    (run / f"run_meta_{TAG}.json").write_text(json.dumps(
        {"models": {"actor": "gpt-5.5"}, "termination": "terminator",
         "args": {}}))
    (run / f"timeline_{TAG}.json").write_text(json.dumps({
        "goal_updates": [{"round": 0, "role": HERO, "via": "set_goal"},
                         {"round": 2, "role": HERO, "via": "update"}],
        "resolution_ledger": [
            {"id": "central_tension", "kind": "tension", "status": "open"},
            {"id": "goal:Hero", "kind": "goal", "status": "resolved"}]}))
    (run / "freeze_validation.json").write_text(json.dumps(
        {"central_tension": "Will Hero defeat Villain?"}))
    for role in (HERO, VILLAIN):
        d = root / "engine" / "data" / "roles" / PACK / role
        d.mkdir(parents=True, exist_ok=True)
        (d / "role_info.json").write_text(json.dumps(
            {"role_name": role.split("-")[0], "profile": "p",
             "motivation": f"win as {role}"}))
    presets = root / "engine" / "presets"
    presets.mkdir(parents=True, exist_ok=True)
    (presets / f"{PACK}.json").write_text(json.dumps(
        {"language": "en", "role_agent_codes": [HERO, VILLAIN]}))
    meta = root / "data" / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"resim_{SID}.json").write_text(json.dumps(
        {"story_id": SID, "freeze_word": 4, "total_words": 10,
         "freeze": {"freeze_quote": "the day he wasn’t ready"}}))
    clean = root / "data" / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    (clean / f"{SID}.txt").write_text(
        "PRE-FREEZE part one. the day he wasn't ready CANONMARK "
        "post-freeze ending here.")


class FakeC3LLM:
    def __init__(self):
        self.n = 0
        self.prompts = []

    def query(self, prompt, model=None, stdin=None, **kw):
        self.n += 1
        self.prompts.append(prompt)
        if "SIMMARK" in prompt or "[Hero-en]" in prompt:
            return json.dumps(SIM_CARD)
        if "CANONMARK" in prompt:
            return json.dumps(CANON_CARD)
        raise AssertionError("unexpected coder input: " + prompt[:80])


class C3Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ANIMASK_ROOT"] = self.tmp.name
        self.root = Path(self.tmp.name)
        build_fixture(self.root)
        self.fake = FakeC3LLM()
        self._orig = pp.llm_query
        pp.llm_query = self.fake.query

    def tearDown(self):
        pp.llm_query = self._orig
        os.environ.pop("ANIMASK_ROOT", None)
        self.tmp.cleanup()


class TestUnits(unittest.TestCase):
    def test_squeeze(self):
        long = "A" * 10000 + "MID" + "B" * 20000
        out = c3.squeeze_text(long, head=100, tail=200)
        self.assertIn("middle omitted", out)
        self.assertTrue(out.startswith("A" * 100))
        self.assertTrue(out.endswith("B" * 200))
        self.assertEqual(c3.squeeze_text("short", 100, 200), "short")

    def test_validate_card_rejects_bad_mode(self):
        bad = dict(SIM_CARD, tension_resolution={"mode": "meh", "how": ""})
        with self.assertRaises(ValueError):
            c3._validate_card(bad)

    def test_validate_card_filters(self):
        messy = dict(CANON_CARD)
        messy = json.loads(json.dumps(messy))
        messy["goals"]["Nobody"] = "flying"          # invalid outcome
        messy["decision_actions"]["flying"] = 3       # invalid category
        messy["decision_actions"]["confront"] = 3.0   # float ok
        card = c3._validate_card(messy)
        self.assertNotIn("Nobody", card["goals"])
        self.assertNotIn("flying", card["decision_actions"])
        self.assertEqual(card["decision_actions"]["confront"], 3)


class TestCanonText(C3Case):
    def test_quote_with_curly_normalization(self):
        text, method = c3.canon_post_freeze(SID)
        self.assertEqual(method, "quote")
        self.assertTrue(text.strip().startswith("CANONMARK"))

    def test_fraction_fallback(self):
        meta_p = self.root / "data" / "meta" / f"resim_{SID}.json"
        m = json.loads(meta_p.read_text())
        m["freeze"]["freeze_quote"] = "NOT PRESENT ANYWHERE"
        meta_p.write_text(json.dumps(m))
        text, method = c3.canon_post_freeze(SID)
        self.assertEqual(method, "word_fraction")
        self.assertLess(len(text), 90)


class TestRunAndAggregate(C3Case):
    def test_run_c3_and_cache(self):
        res = c3.run_c3(PACK, TAG)
        self.assertEqual(res["card"]["conflict_curve"], "de_escalates")
        self.assertEqual(res["rule"]["tension_status"], "open")
        self.assertEqual(res["rule"]["termination"], "terminator")
        self.assertEqual(res["rule"]["n_goal_updates"][HERO], 2)
        n = self.fake.n
        res2 = c3.run_c3(PACK, TAG)          # cached: no new calls
        self.assertEqual(self.fake.n, n)
        self.assertEqual(res2["card"], res["card"])
        # error rows excluded from the coded transcript
        self.assertNotIn("IGNORED", self.fake.prompts[0])

    def test_canon_card_cache(self):
        res = c3.canon_card(SID)
        self.assertEqual(res["freeze_method"], "quote")
        self.assertEqual(res["card"]["tension_resolution"]["mode"],
                         "resolved_as_posed")
        n = self.fake.n
        c3.canon_card(SID)
        self.assertEqual(self.fake.n, n)

    def test_aggregate(self):
        c3.run_c3(PACK, TAG)
        c3.canon_card(SID)
        agg = c3.aggregate("fb")
        self.assertEqual(agg["n_paired_stories"], 1)
        self.assertEqual(agg["simulation"]["tension_modes"],
                         {"dissolved": 1})
        self.assertEqual(agg["canon"]["tension_modes"],
                         {"resolved_as_posed": 1})
        # sim 2 transgressions vs canon 5 -> ratio 0.4
        self.assertAlmostEqual(
            agg["transgression_ratio_sim_over_canon"], 2 / 5)
        # rule says open, coder says dissolved -> agree (both non-settled)
        self.assertEqual(agg["tension_rule_x_coder"]["agree"], 1)
        self.assertTrue((pp.aggregate_dir() / "c3_fb.md").exists())

    def test_aggregate_missing_canon(self):
        c3.run_c3(PACK, TAG)
        agg = c3.aggregate("fb")
        self.assertEqual(agg["n_paired_stories"], 0)
        self.assertEqual(agg["missing_canon"], [SID])
        self.assertIsNone(agg["transgression_ratio_sim_over_canon"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
