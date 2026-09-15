"""Run-guard regressions: dead judge, safe parsing, call accounting (no LLM).

1. Terminator — an empty/unparseable judge reply is NOT progress: it must
   not reset the stagnation streak, must mark judge_error, and three
   consecutive failures abort the run (a dead judge route used to let every
   run silently burn to the round cap).
2. json_parser — eval() removed: json first, ast.literal_eval fallback for
   Python-style dicts; arbitrary expressions must NOT be executed.
3. persona_probe parse-skip accounting — unparseable-but-transport-ok judge
   replies count toward CALL_STATS so `complete` can go false and the batch
   runner retries the missing items.

Run:  python3 tests/test_run_guards.py
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(REPO))

from modules.terminator import Terminator  # noqa: E402
from utils import json_parser  # noqa: E402
from animask import persona_probe as pp  # noqa: E402


class _DeadLLM:
    temperature = 0.0

    def chat(self, *a, **k):
        return ""


class _StagnantLLM:
    temperature = 0.0

    def chat(self, *a, **k):
        return '{"material_change": false, "elapsed_hours_this_round": 1}'


class TestTerminatorJudgeFailure(unittest.TestCase):
    def test_empty_reply_is_not_progress(self):
        t = Terminator(_DeadLLM(), "tension?", terminal_goals={"X": "escape"})
        t.stagnant_streak = 1
        v = t.check("latest", "previous")
        self.assertTrue(v["judge_error"])
        self.assertFalse(v["terminate"])
        self.assertEqual(t.stagnant_streak, 1)  # unchanged, not reset

    def test_three_consecutive_failures_abort(self):
        t = Terminator(_DeadLLM(), "tension?")
        t.check("a", "b")
        t.check("a", "b")
        with self.assertRaises(RuntimeError):
            t.check("a", "b")

    def test_recovery_resets_failure_counter(self):
        t = Terminator(_StagnantLLM(), "tension?")
        t.judge_failures = 2
        v = t.check("a", "b")
        self.assertEqual(t.judge_failures, 0)
        self.assertEqual(v["stagnant_streak"], 1)
        self.assertNotIn("judge_error", v)


class TestJsonParserNoEval(unittest.TestCase):
    def test_json_and_python_literals(self):
        self.assertEqual(json_parser('{"a": true, "detail": "x"}'),
                         {"a": True, "detail": "x"})
        self.assertEqual(json_parser("{'a': True, 'b': None}"),
                         {"a": True, "b": None})
        self.assertEqual(json_parser(""), {})

    def test_no_code_execution(self):
        import builtins
        flag = {"hit": False}
        orig = builtins.__import__

        def spy(name, *a, **k):
            if name == "os_should_never_import":
                flag["hit"] = True
            return orig(name, *a, **k)

        builtins.__import__ = spy
        try:
            with self.assertRaises(ValueError):
                json_parser(
                    '{__import__("os_should_never_import"): 1}')
        except AssertionError:
            pass  # a dict result is fine too, as long as nothing executed
        finally:
            builtins.__import__ = orig
        self.assertFalse(flag["hit"])


class TestParseSkipAccounting(unittest.TestCase):
    def test_counter_and_key_present(self):
        self.assertIn("parse_skip", pp.CALL_STATS)
        before = pp.CALL_STATS["parse_skip"]
        pp.tally_parse_skip()
        self.assertEqual(pp.CALL_STATS["parse_skip"], before + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
