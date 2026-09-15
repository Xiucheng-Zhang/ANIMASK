"""Core engine and probe-package regressions (no LLM).

- NaiveDB zh retrieval — CJK bigram tokenization must rank the
  relevant passage first (the old whole-passage token gave every zh query
  zero overlap, degrading retrieval to insertion order).
- terminator ledger updates — a judge echoing "[central_tension]"
  (brackets, case, spacing) must still land on the ledger entry.
- the probe package's <<ERROR tally and credit latch.

Follow-up attribution, the freeze-validation cache gate and the
error-exit / eval guards / leftover isolation are control-flow changes in
run scripts with no cheap unit seam; they are not covered here.

Run:  python3 tests/test_core_regressions.py
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(REPO))

from modules.db.NaiveDB import NaiveDB, _toks  # noqa: E402
from modules.terminator import Terminator  # noqa: E402
from animask import persona_probe as pp  # noqa: E402


class TestNaiveDBZh(unittest.TestCase):
    DOCS = [
        "廉衿夜翻阅着资料，不满的气焰让员工冷汗直冒。",
        "夏星辰提出辞职，准备回家照顾母亲。",
        "廉衿夜提出条件：回廉氏继续当秘书，年薪翻倍。",
    ]

    def test_cjk_bigrams(self):
        self.assertIn("年薪", _toks("双倍年薪"))
        self.assertNotIn("双倍年薪", _toks("双倍年薪"))

    def test_mixed_ascii_cjk(self):
        self.assertEqual(sorted(_toks("VIP室307")), ["307", "vip", "室"])

    def test_zh_query_ranks_relevant_doc_first(self):
        db = NaiveDB()
        db.init_from_data(self.DOCS, "d")
        # the review's repro: 「双倍年薪」 used to return the first two
        # unrelated passages (insertion order on all-zero scores)
        top = db.search("双倍年薪", 1, "d")[0]
        self.assertIn("年薪翻倍", top)

    def test_en_behavior_unchanged(self):
        db = NaiveDB()
        db.init_from_data(["the harbor docks at night",
                           "a quiet market square"], "e")
        self.assertIn("harbor", db.search("harbor docks", 1, "e")[0])


class TestLedgerIdNormalization(unittest.TestCase):
    def _term(self):
        return Terminator(llm=None, central_tension="张力？",
                          terminal_goals={"夏冬冬": "破案"})

    def test_bracketed_id_lands(self):
        t = self._term()
        t._apply_ledger_updates([
            {"id": "[central_tension]", "status": "achieved", "evidence": "e"},
            {"id": "[goal:夏冬冬]", "status": "achieved", "evidence": "e"}])
        self.assertEqual([e["status"] for e in t.ledger],
                         ["achieved", "achieved"])

    def test_case_and_spacing_land(self):
        t = self._term()
        t._apply_ledger_updates([
            {"id": " Central_Tension ", "status": "failed", "evidence": "e"},
            {"id": "goal: 夏冬冬", "status": "moot", "evidence": "e"}])
        self.assertEqual([e["status"] for e in t.ledger], ["failed", "moot"])

    def test_unknown_and_invalid_ignored(self):
        t = self._term()
        t._apply_ledger_updates([
            {"id": "[nonexistent]", "status": "achieved", "evidence": ""},
            {"id": "central_tension", "status": "not_a_state",
             "evidence": ""},
            "not_a_dict"])
        self.assertEqual([e["status"] for e in t.ledger], ["open", "open"])


class TestProbeCallTally(unittest.TestCase):
    def setUp(self):
        self._stats = dict(pp.CALL_STATS)
        self._latch = pp._CREDIT_LATCH

    def tearDown(self):
        pp.CALL_STATS.update(self._stats)
        pp._CREDIT_LATCH = self._latch

    def test_tally_counts_and_latches(self):
        base_err = pp.CALL_STATS["error"]
        base_ok = pp.CALL_STATS["ok"]
        pp._tally("a fine reply")
        pp._tally("<<ERROR: HTTP 500>>")
        self.assertEqual(pp.CALL_STATS["ok"], base_ok + 1)
        self.assertEqual(pp.CALL_STATS["error"], base_err + 1)
        self.assertFalse(pp._CREDIT_LATCH)
        pp._tally("<<ERROR: CREDIT_EXHAUSTED: b'...'>>")
        self.assertTrue(pp._CREDIT_LATCH)

    def test_latch_short_circuits_query_many(self):
        pp._CREDIT_LATCH = True
        outs = pp.llm_query_many([{"prompt": "x"}, {"prompt": "y"}])
        self.assertEqual(len(outs), 2)
        for o in outs:
            self.assertTrue(o.startswith("<<ERROR: CREDIT_EXHAUSTED"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
