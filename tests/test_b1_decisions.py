"""Synthetic-fixture tests for persona_probe.b1_decisions (B1 decision-point
three-label pipeline). No real API calls.

    python3 tests/test_b1_decisions.py
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask import persona_probe as pp                                  # noqa: E402
from animask.persona_probe import b1_decisions as b1               # noqa: E402

PACK = "bpack"
TAG = "fb_r1"
HERO, VILLAIN = "Hero-en", "Villain-en"
HERO_PROFILE = "HERO_PROFILE_BLOCK: Hero is brave and blunt."
VILLAIN_PROFILE = "VILLAIN_PROFILE_BLOCK: Villain is scheming and polite."
N_ROUNDS = 6


def build_fixture(root: Path):
    run = root / "results" / "runs" / PACK
    run.mkdir(parents=True, exist_ok=True)
    lines = []

    def rec(role, profile, rnd, k, detail):
        lines.append(json.dumps({
            "ts": 0, "model": "gpt-5.5", "tag": f"role:{role}",
            "round": rnd, "kind": None, "temperature": 0.7,
            "messages": [{"role": "user",
                          "content": f"Scene context r{rnd} k{k}\n"
                                     f"## Your profile\n{profile}\n"
                                     "wary\nWin.\nInstructions"}],
            "response": json.dumps({"action": "speak",
                                    "detail": f"PSTYLE {role} act r{rnd}"})}))

    for rnd in range(N_ROUNDS):
        rec(HERO, HERO_PROFILE, rnd, 0, "x")
        rec(HERO, HERO_PROFILE, rnd, 1, "x")     # sub-round second call
        rec(VILLAIN, VILLAIN_PROFILE, rnd, 0, "x")
    (run / f"llm_calls_{TAG}.jsonl").write_text("\n".join(lines) + "\n")
    (run / f"run_meta_{TAG}.json").write_text(json.dumps(
        {"models": {"actor": "gpt-5.5"}, "args": {"persona_level": "P3"}}))
    (run / f"verdicts_{TAG}.json").write_text(json.dumps([
        {"reason": f"round {r} judged", "new_commitments": [],
         "resolved_commitments": []} for r in range(N_ROUNDS)]))
    for role, profile, other in ((HERO, HERO_PROFILE, VILLAIN),
                                 (VILLAIN, VILLAIN_PROFILE, HERO)):
        d = root / "engine" / "data" / "roles" / PACK / role
        d.mkdir(parents=True, exist_ok=True)
        (d / "role_info.json").write_text(json.dumps({
            "role_code": role, "role_name": role.split("-")[0],
            "profile": profile,
            "relation": {other: {"relation": ["rival"], "detail": "wary"}},
            "motivation": "Win."}))
    presets = root / "engine" / "presets"
    presets.mkdir(parents=True, exist_ok=True)
    (presets / f"{PACK}.json").write_text(json.dumps(
        {"language": "en", "role_agent_codes": [HERO, VILLAIN]}))


class FakeB1LLM:
    """Marker-dispatched fake for the three judges + the replay actor."""

    def __init__(self):
        self.n_query = 0
        self.n_chat = 0
        self.implied_prompts = []

    def _round_of(self, text):
        m = re.search(r"r(\d)", text)
        return int(m.group(1)) if m else 0

    def query(self, prompt, model=None, stdin=None, **kw):
        self.n_query += 1
        if "CONSEQUENTIAL" in prompt or "有后果的选择" in prompt:
            m = re.search(r"round (\d+)", prompt)
            rnd = int(m.group(1))
            pts = [{"actor": HERO, "call": 0,
                    "choice": f"steal or not r{rnd}",
                    "risk": "medium" if rnd == 3 else "high",
                    "present": [VILLAIN]}]
            if rnd == 5:
                pts.append({"actor": VILLAIN, "call": 0,
                            "choice": "betray or not r5", "risk": "medium",
                            "present": [HERO]})
            if rnd == 3:      # same-actor duplicate: high risk must win
                pts.append({"actor": HERO, "call": 1,
                            "choice": "steal or not r3 retake",
                            "risk": "high", "present": []})
            if rnd == 0:   # invalid entries the parser must drop
                pts += [{"actor": "Ghost-en", "call": 0, "choice": "x"},
                        {"actor": HERO, "call": 9, "choice": "x"}]
            return json.dumps(pts)
        if "PERSONA CARD" in prompt or "的角色卡" in prompt:
            self.implied_prompts.append(prompt)
            rnd = self._round_of(prompt)
            if rnd == 1:      # set semantics: primary misses, set catches
                return json.dumps({"category": "pursue",
                                   "acceptable": ["confront"],
                                   "confidence": "high", "rationale": "c"})
            return json.dumps({"category": "confront", "acceptable": [],
                               "confidence": "high", "rationale": "card"})
        if "classifying one character action" in prompt \
                or "归入固定类别表" in prompt:
            m = re.search(r"(PSTYLE|DSTYLE) \S+ act r(\d)", prompt)
            kind, rnd = m.group(1), int(m.group(2))
            if kind == "PSTYLE":
                cat = "confront" if rnd < 4 else "concede"
            else:
                cat = ("cooperate" if rnd < 2
                       else ("confront" if rnd < 4 else "concede"))
            return json.dumps({"category": cat, "rationale": "obvious"})
        raise AssertionError("unrecognized prompt: " + prompt[:100])

    def query_many(self, jobs, max_workers=10):
        return [self.query(**j) for j in jobs]

    def chat(self, messages, model=None, temperature=0.0, **kw):
        self.n_chat += 1
        ctx = " ".join(str(m["content"]) for m in messages)
        assert "PROFILE_BLOCK" not in ctx, "replay saw the persona card"
        rnd = self._round_of(ctx)
        return json.dumps({"action": "speak",
                           "detail": f"DSTYLE replay act r{rnd}"})

    def chat_many(self, jobs, max_workers=10):
        return [self.chat(**j) for j in jobs]


class B1Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ANIMASK_ROOT"] = self.tmp.name
        self.root = Path(self.tmp.name)
        build_fixture(self.root)
        self.fake = FakeB1LLM()
        self._orig = (pp.llm_query_many, pp.llm_chat_many)
        pp.llm_query_many = self.fake.query_many
        pp.llm_chat_many = self.fake.chat_many

    def tearDown(self):
        pp.llm_query_many, pp.llm_chat_many = self._orig
        os.environ.pop("ANIMASK_ROOT", None)
        self.tmp.cleanup()


class TestUnits(unittest.TestCase):
    def test_cell_of(self):
        self.assertEqual(b1.cell_of("a", "a", "b"), "adherent_effective")
        self.assertEqual(b1.cell_of("a", "a", "a"), "adherent_vacuous")
        self.assertEqual(b1.cell_of("a", "b", "b"), "overridden")
        self.assertEqual(b1.cell_of("a", "b", "c"), "divergent_other")

    def test_post_category_rejects_unknown(self):
        with self.assertRaises(ValueError):
            b1._post_category(json.dumps({"category": "flying"}))
        out = b1._post_category(json.dumps({"category": "Confront",
                                            "rationale": "r"}))
        self.assertEqual(out["category"], "confront")

    def test_summarize(self):
        pts = [{"cell": "adherent_effective", "implied": "a", "actual": "a"},
               {"cell": "overridden", "implied": "a", "actual": "b"},
               {"cell": "overridden", "implied": "a", "actual": "b"},
               {"cell": None}]
        s = b1.summarize_points(pts)
        self.assertEqual(s["n_labeled"], 3)
        self.assertAlmostEqual(s["adherence"], 1 / 3)
        self.assertEqual(s["deviation_flows"], {"a->b": 2})

    def test_phase_of(self):
        self.assertEqual(b1._phase_of({"round": 0}, 6), "early")
        self.assertEqual(b1._phase_of({"round": 3}, 6), "mid")
        self.assertEqual(b1._phase_of({"round": 5}, 6), "late")


class TestRunB1(B1Case):
    def test_end_to_end(self):
        res = b1.run_b1(PACK, TAG)
        # 6 hero points + 1 villain point; invalid scan entries dropped
        self.assertEqual(res["n_points"], 7)
        self.assertEqual(res["n_complete"], 7)
        self.assertEqual(res["ablation_notes"], {})
        hero_pts = sorted([p for p in res["points"] if p["actor"] == HERO],
                          key=lambda p: p["round"])
        cells = [p["cell"] for p in hero_pts]
        self.assertEqual(cells, ["adherent_effective", "adherent_effective",
                                 "adherent_vacuous", "adherent_vacuous",
                                 "overridden", "overridden"])
        # round 1: actual within the implied SET but not the primary label
        r1 = hero_pts[1]
        self.assertEqual(r1["implied_set"], ["pursue", "confront"])
        self.assertEqual(r1["cell"], "adherent_effective")
        self.assertEqual(r1["cell_strict"], "divergent_other")
        # round 3: two same-actor candidates -> only the high-risk one kept
        r3 = hero_pts[3]
        self.assertEqual((r3["call"], r3["risk"]), (1, "high"))
        self.assertEqual(len([p for p in hero_pts if p["round"] == 3]), 1)
        s_hero = b1.summarize_points(hero_pts)
        self.assertAlmostEqual(s_hero["adherence"], 4 / 6)
        self.assertAlmostEqual(s_hero["adherence_strict"], 3 / 6)
        self.assertAlmostEqual(s_hero["efficacy"], 2 / 6)
        self.assertEqual(s_hero["deviation_flows"], {"confront->concede": 2})
        out = pp.run_dir(PACK) / f"b1_decisions_{TAG}.json"
        self.assertTrue(out.exists())

    def test_blind_implied_judge(self):
        b1.run_b1(PACK, TAG)
        self.assertTrue(self.fake.implied_prompts)
        for prompt in self.fake.implied_prompts:
            # never sees the actual action, does see card + redacted context
            self.assertNotIn("PSTYLE", prompt)
            self.assertIn("PROFILE_BLOCK", prompt)      # card section
            self.assertIn("You are a character in this story.", prompt)

    def test_resume_no_new_calls(self):
        b1.run_b1(PACK, TAG)
        n_q, n_c = self.fake.n_query, self.fake.n_chat
        res2 = b1.run_b1(PACK, TAG)
        self.assertEqual(self.fake.n_query, n_q)
        self.assertEqual(self.fake.n_chat, n_c)
        self.assertEqual(res2["n_complete"], 7)

    def test_plan_mode(self):
        res = b1.run_b1(PACK, TAG, plan_only=True)
        self.assertEqual(res["n_rounds"], N_ROUNDS)
        self.assertEqual(self.fake.n_query + self.fake.n_chat, 0)


class TestAggregate(B1Case):
    def _facets(self, fail_roles):
        pp.save_json_atomic(
            pp.aggregate_dir() / "facets_fb.json",
            {"runs": [{"pack": PACK, "per_role": {
                r: {"gate": {"pass": r not in fail_roles}}
                for r in (HERO, VILLAIN)}}]})

    def test_aggregate_with_gate(self):
        b1.run_b1(PACK, TAG)
        self._facets({VILLAIN})
        agg = b1.aggregate("fb")
        g = agg["groups"]["gpt-5.5"]
        self.assertEqual(g["summary"]["n_labeled"], 6)   # hero only
        self.assertEqual(g["n_gate_excluded"], 1)
        self.assertEqual(g["n_gate_unknown"], 0)
        self.assertAlmostEqual(g["summary"]["adherence"], 4 / 6)
        self.assertAlmostEqual(g["summary"]["efficacy"], 2 / 6)
        self.assertEqual(g["implied_x_actual"]["confront"]["concede"], 2)
        self.assertEqual(g["by_phase"]["early"]["n_labeled"], 2)
        self.assertEqual(g["by_phase"]["late"]["n_labeled"], 2)
        ci = g["adherence_ci"]
        self.assertLessEqual(ci["lo"], ci["stat"])
        self.assertTrue((pp.aggregate_dir() / "b1_fb.md").exists())

    def test_aggregate_without_gate_file(self):
        b1.run_b1(PACK, TAG)
        agg = b1.aggregate("fb")
        g = agg["groups"]["gpt-5.5"]
        self.assertEqual(g["summary"]["n_labeled"], 7)
        self.assertEqual(g["n_gate_excluded"], 0)
        self.assertEqual(g["n_gate_unknown"], 0)

    def test_aggregate_gate_unknown_quarantined(self):
        b1.run_b1(PACK, TAG)
        # facets file knows Hero only; Villain has no gate info
        pp.save_json_atomic(
            pp.aggregate_dir() / "facets_fb.json",
            {"runs": [{"pack": PACK, "per_role": {
                HERO: {"gate": {"pass": True}}}}]})
        agg = b1.aggregate("fb")
        g = agg["groups"]["gpt-5.5"]
        self.assertEqual(g["summary"]["n_labeled"], 6)
        self.assertEqual(g["n_gate_unknown"], 1)
        self.assertEqual(g["n_gate_excluded"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
