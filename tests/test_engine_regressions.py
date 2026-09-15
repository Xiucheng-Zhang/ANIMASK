"""Engine regressions: locations, parsing, concealment, event keys (no LLM).

1. resolve_location_alias — zh destination names must resolve to the right
   location (the old ASCII-only rule collapsed every zh name to "" and
   teleported movers to an arbitrary location).
2. pick_location_by_name — longest contained name wins (first containment
   put one story's cast in 皇宫/内侍局 instead of 内侍局宿舍).
3. as_bool — a "false" STRING from a judge must not count as true (the
   fixture admission gate leaked rejected canon events).
4. conceal_thoughts — multi-line 【…\n…】 monologue must not leak.
5. strip_script_headers --apply — a failed freeze-quote verification must
   block the write, not just print a warning after it.
6. norm_event_key — fixture verdicts must correlate through name variants
   ("Winter School Break" vs winter_school_break); exact matching silently
   dropped every fixture and never cached the audit.
7. decode_location_codes — longest code replaced first, or prefix-nested
   pinyin codes render as hybrid text (廉氏集团DaSha).
8. zh runtime strings — the {experiences} empty state must not inject
   English into zh prompts.

Run:  python3 tests/test_engine_regressions.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(REPO))

from utils import (as_bool, conceal_thoughts,  # noqa: E402
                      decode_location_codes, norm_event_key,
                      pick_location_by_name, resolve_location_alias)
from modules.main_role_agent import RPAgent  # noqa: E402
from animask.resim import strip_script_headers as ssh  # noqa: E402

ZH_LOCS = {
    "XiaXingChenZhuChu-zh": {"location_name": "夏星辰住处"},
    "LianShiJiTuanDaSha-zh": {"location_name": "廉氏集团大厦"},
    "YueYangDuJiaCunWai-zh": {"location_name": "悦洋度假村外"},
}


class TestResolveLocationAlias(unittest.TestCase):
    def test_zh_name_resolves_to_its_own_location(self):
        self.assertEqual(resolve_location_alias("廉氏集团大厦", ZH_LOCS),
                         "LianShiJiTuanDaSha-zh")
        self.assertEqual(resolve_location_alias("夏星辰住处", ZH_LOCS),
                         "XiaXingChenZhuChu-zh")

    def test_punctuation_tolerated(self):
        self.assertEqual(resolve_location_alias("廉氏集团·大厦！", ZH_LOCS),
                         "LianShiJiTuanDaSha-zh")

    def test_unknown_zh_name_is_none_not_arbitrary(self):
        # the old rule normalized this AND every table name to "" -> match
        self.assertIsNone(resolve_location_alias("金家赌场", ZH_LOCS))

    def test_empty_never_matches(self):
        self.assertIsNone(resolve_location_alias("！？…", ZH_LOCS))
        self.assertIsNone(resolve_location_alias("", ZH_LOCS))

    def test_code_near_miss_resolves(self):
        self.assertEqual(
            resolve_location_alias("lian-shi-ji-tuan-da-sha-zh", ZH_LOCS),
            "LianShiJiTuanDaSha-zh")


class TestPickLocationByName(unittest.TestCase):
    YQ041 = {
        "HuangGong-zh": {"location_name": "皇宫"},
        "NeiShiJu-zh": {"location_name": "内侍局"},
        "NeiShiJuSuShe-zh": {"location_name": "内侍局宿舍"},
        "NeiShiJuHouYuan-zh": {"location_name": "内侍局后院"},
    }

    def test_longest_contained_name_wins(self):
        # first-containment used to land these on 皇宫 / 内侍局
        self.assertEqual(
            pick_location_by_name("皇宫内侍局宿舍门口", self.YQ041, "d"),
            "NeiShiJuSuShe-zh")
        self.assertEqual(
            pick_location_by_name("内侍局宿舍门口附近", self.YQ041, "d"),
            "NeiShiJuSuShe-zh")

    def test_token_overlap_fallback(self):
        locs = {"A": {"location_name": "city harbor docks"},
                "B": {"location_name": "market square"}}
        self.assertEqual(
            pick_location_by_name("at the harbor docks", locs, "d"), "A")

    def test_default_when_nothing_matches(self):
        self.assertEqual(pick_location_by_name("somewhere new",
                                               self.YQ041, "d"), "d")


class TestAsBool(unittest.TestCase):
    def test_false_string_is_false(self):
        self.assertFalse(as_bool("false"))
        self.assertFalse(as_bool("False "))
        self.assertFalse(as_bool(""))
        self.assertFalse(as_bool(None))
        self.assertFalse(as_bool(0))

    def test_true_values(self):
        self.assertTrue(as_bool(True))
        self.assertTrue(as_bool("true"))
        self.assertTrue(as_bool(" TRUE"))


class TestConcealThoughts(unittest.TestCase):
    def test_multiline_monologue_stripped(self):
        detail = "夏星辰：【他到底想干什么……\n先稳住。】你先坐吧。"
        out = conceal_thoughts(detail)
        self.assertNotIn("稳住", out)
        self.assertIn("你先坐吧", out)

    def test_multiline_ascii_brackets_stripped(self):
        out = conceal_thoughts("A: [thinks hard\nabout it] Hello.")
        self.assertNotIn("thinks", out)
        self.assertIn("Hello", out)


class TestApplyGating(unittest.TestCase):
    HEADER = "《某剧》\n简纲：\n梗概带结局。\n人物设定：\n甲：小传。\n正剧\n"
    BODY = "第一集\n" + "\n".join(
        [f"1-{i} 某地 内 夜 人物：甲\n甲：这里是冻结引文{i}号，正文台词。"
         for i in range(1, 20)])

    def _setup(self, tmp, freeze_word_offset=0):
        clean = Path(tmp) / "clean"
        meta = Path(tmp) / "meta"
        worlds = Path(tmp) / "worlds"
        for d in (clean, meta, worlds):
            d.mkdir()
        text = self.HEADER + self.BODY
        (clean / "yqtest.txt").write_text(text)
        quote = "甲：这里是冻结引文9号，正文台词。"
        cut = len(text[:text.find(quote)].split()) + freeze_word_offset
        (meta / "resim_yqtest.json").write_text(json.dumps({
            "story_id": "yqtest", "freeze_word": cut,
            "total_words": len(text.split()),
            "freeze": {"freeze_quote": quote,
                       "approx_fraction": cut / len(text.split()),
                       "snapped_fraction": cut / len(text.split())}},
            ensure_ascii=False))
        ssh.CLEAN, ssh.META, ssh.WORLDS = clean, meta, worlds
        ssh.FREEZE_CACHE = Path(tmp) / "cache"
        return clean, meta

    def tearDown(self):
        ssh.CLEAN = ROOT_CLEAN
        ssh.META = ROOT_META
        ssh.WORLDS = ROOT_WORLDS
        ssh.FREEZE_CACHE = ROOT_FREEZE

    def test_clean_apply_updates_text_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean, meta = self._setup(tmp)
            out = ssh.process("yqtest", apply=True)
            self.assertTrue(out.get("applied"), out)
            new_text = (clean / "yqtest.txt").read_text()
            self.assertTrue(new_text.startswith("第一集"))
            m = json.loads((meta / "resim_yqtest.json").read_text())
            self.assertIn("header_stripped", m)
            words = new_text.split()
            self.assertEqual(words[m["freeze_word"]].startswith("甲：这里是冻结引文9号"),
                             True)

    def test_quote_disagreement_blocks_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean, meta = self._setup(tmp, freeze_word_offset=10)
            before = (clean / "yqtest.txt").read_text()
            out = ssh.process("yqtest", apply=True)
            self.assertFalse(out.get("applied"), out)
            self.assertIn("disagrees", out.get("error", ""))
            self.assertEqual((clean / "yqtest.txt").read_text(), before)


class TestNormEventKey(unittest.TestCase):
    def test_llm_echoed_variants_correlate(self):
        self.assertEqual(norm_event_key("Winter School Break"),
                         "winter_school_break")
        self.assertEqual(norm_event_key("harvester-supervisor check-in."),
                         "harvester_supervisor_check_in")
        self.assertEqual(norm_event_key(" winter_school_break "),
                         "winter_school_break")

    def test_zh_names_survive(self):
        self.assertEqual(norm_event_key("金家 绑走 童嘉许"), "金家_绑走_童嘉许")

    def test_distinct_names_stay_distinct(self):
        self.assertNotEqual(norm_event_key("market_day"),
                            norm_event_key("market_night"))


class TestDecodeLocationCodes(unittest.TestCase):
    NESTED = {
        "LianShiJiTuan": {"location_name": "廉氏集团"},
        "LianShiJiTuanDaSha": {"location_name": "廉氏集团大厦"},
        "YiYuan": {"location_name": "医院"},
        "YiYuanZhenShi": {"location_name": "医院诊室"},
    }

    def test_longest_code_replaced_first(self):
        # insertion order used to replace the short prefix first:
        # LianShiJiTuanDaSha -> 廉氏集团DaSha
        self.assertEqual(
            decode_location_codes("他走进LianShiJiTuanDaSha。", self.NESTED),
            "他走进廉氏集团大厦。")
        self.assertEqual(
            decode_location_codes("从YiYuanZhenShi回到YiYuan门口",
                                  self.NESTED),
            "从医院诊室回到医院门口")

    def test_non_string_passthrough(self):
        self.assertIsNone(decode_location_codes(None, self.NESTED))


class TestZhRuntimeStrings(unittest.TestCase):
    def _agent(self, language):
        a = RPAgent.__new__(RPAgent)
        a.episode_notes, a.language = [], language
        return a

    def test_experiences_empty_state_localized(self):
        self.assertNotIn("none yet",
                         self._agent("zh").get_experiences_text())
        self.assertIn("none yet", self._agent("en").get_experiences_text())


ROOT_CLEAN, ROOT_META, ROOT_WORLDS = ssh.CLEAN, ssh.META, ssh.WORLDS
ROOT_FREEZE = ssh.FREEZE_CACHE

if __name__ == "__main__":
    unittest.main(verbosity=2)
