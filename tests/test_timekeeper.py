"""Timekeeper regressions and the prompt-style default (no LLM).

1. A fixture restating a same-named process's schedule is dropped (it used
   to fire a second [WORLD EVENT] at the same moment).
2. Queue entries carry display `name` + internal `key`: occurrences count by
   key (no anchor inflation), prompts/records see only the original name.
3. remove_event matches judge free-text via norm_event_key and refuses
   fixtures; bounced removals return False (the engine logs them).
4. ANIMASK_PROMPT_STYLE fallback default is "neutral" at every entry point.

Run:  python3 tests/test_timekeeper.py
"""
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))

from modules.timekeeper import Timekeeper  # noqa: E402

CHECK = {"name": "harvester_supervisor_check_in", "period_hours": 24,
         "due_hours": 24, "description": "the supervisor inspects the field"}


class TestScheduleDuplicateFixtures(unittest.TestCase):
    def test_restating_fixtures_dropped(self):
        # garcia shape: 4 fixtures at 24/48/72/96h restating a 24h process
        fixtures = [{"name": CHECK["name"], "due_hours": h,
                     "description": "supervisor check-in"}
                    for h in (24, 48, 72, 96)]
        tk = Timekeeper([CHECK], fixtures)
        self.assertEqual(len(tk.queue), 1)  # only the process survives
        passed = tk.advance(24.0)
        self.assertEqual(len(passed), 1)  # ONE world event, not two

    def test_off_schedule_fixture_kept(self):
        tk = Timekeeper([CHECK], [{"name": CHECK["name"], "due_hours": 30,
                                   "description": "an unscheduled visit"}])
        self.assertEqual(len(tk.queue), 2)


class TestKeyDisplaySplit(unittest.TestCase):
    def test_display_name_never_suffixed(self):
        tk = Timekeeper([CHECK], [{"name": CHECK["name"], "due_hours": 30,
                                   "description": "an unscheduled visit"}])
        names = {e["name"] for e in tk.queue}
        self.assertEqual(names, {CHECK["name"]})  # display names original
        keys = [e["key"] for e in tk.queue]
        self.assertEqual(len(keys), len(set(keys)))  # keys unique

    def test_occurrences_counted_by_key(self):
        tk = Timekeeper([CHECK], [{"name": CHECK["name"], "due_hours": 30,
                                   "description": "an unscheduled visit"}])
        passed = tk.advance(30.0)
        self.assertEqual(len(passed), 2)
        # the fixture must not credit the process's anchor count
        self.assertEqual(tk.occurrences[CHECK["name"]], 1)
        anchor = {"anchor_type": "event", "anchor_process": CHECK["name"],
                  "anchor_count": 2}
        self.assertGreater(tk.anchor_target_hours(anchor), 0)

    def test_passed_events_carry_display_name(self):
        tk = Timekeeper([CHECK], [{"name": CHECK["name"], "due_hours": 30,
                                   "description": "an unscheduled visit"}])
        for ev in tk.advance(30.0):
            self.assertEqual(ev["name"], CHECK["name"])


class TestRemoveEventNormalized(unittest.TestCase):
    def test_free_text_variant_matches(self):
        tk = Timekeeper([])
        nm = tk.add_event("dinner with the mayor", 6, "appointment")
        self.assertTrue(tk.remove_event(nm.replace("_", " ").title()))
        self.assertEqual(tk.pending_commitments(), [])

    def test_miss_returns_false(self):
        tk = Timekeeper([])
        tk.add_event("dinner with the mayor", 6, "appointment")
        self.assertFalse(tk.remove_event("breakfast_summit"))
        self.assertEqual(len(tk.pending_commitments()), 1)

    def test_fixtures_not_removable(self):
        tk = Timekeeper([], [{"name": "storm", "due_hours": 12,
                              "description": "a storm arrives"}])
        self.assertFalse(tk.remove_event("storm"))


class TestPromptStyleDefault(unittest.TestCase):
    def test_fallback_default_is_neutral(self):
        # server/__main__ entry points set no env var; they used to run
        # "original" while the batch CLI default was "neutral"
        from modules.main_role_agent import RPAgent
        from modules.prompt import role_agent_prompt_neutral as rn
        old = os.environ.pop("ANIMASK_PROMPT_STYLE", None)
        try:
            a = RPAgent.__new__(RPAgent)
            a.language = "en"
            a._init_prompt()
            self.assertEqual(a._ROLE_PLAN_PROMPT, rn.ROLE_PLAN_PROMPT)
        finally:
            if old is not None:
                os.environ["ANIMASK_PROMPT_STYLE"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
