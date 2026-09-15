"""Header-strip rules for screenplay scripts (no LLM, no file writes).

Fixtures model the header shapes seen in the corpus: labeled sections +
正剧 marker; markdown sections with an unlabeled synopsis and a 剧本正文
marker; title and episode marker on one line; no header at all; bare title
without 《》; per-episode outline vs real body.

Run:  python3 tests/test_strip_script_headers.py
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask.resim.strip_script_headers import (relocate_freeze,  # noqa: E402
                                        strip_script_header)

BODY = "\n".join(
    ["1-1 某地 内 夜 人物：甲 乙",
     "甲：这句是台词。"] + [f"乙：第{i}句台词，用来撑起正文长度检查。" for i in range(30)])


class TestStrip(unittest.TestCase):
    def test_labeled_header_with_marker(self):
        text = "\n".join(["《某剧》", "标签：甜宠", "集数：30 集", "简纲：",
                          "全剧梗概，含结局。", "人物设定：", "甲：主角小传。",
                          "正剧", "第一集", BODY])
        res = strip_script_header(text)
        self.assertTrue(res["body"].startswith("第一集"))
        self.assertNotIn("简纲", res["body"])
        self.assertNotIn("小传", res["body"])
        self.assertIn("body_marker", res["flags"])

    def test_markdown_and_unlabeled_synopsis(self):
        text = "\n".join(["《某剧》故事围绕主角展开，最终走到结局。",
                          "第二行梗概。", "## 人物介绍", "- **甲**：主角。",
                          "## 【剧本正文】", "### 第一集", BODY])
        res = strip_script_header(text)
        self.assertTrue(res["body"].startswith("### 第一集"))
        self.assertNotIn("梗概", res["body"])

    def test_title_and_episode_on_one_line(self):
        text = "《佳偶天成》第一集\n" + BODY
        res = strip_script_header(text)
        self.assertTrue(res["body"].startswith("第一集"))
        self.assertNotIn("《佳偶天成》", res["body"])
        self.assertIn("title_prefix_removed", res["flags"])

    def test_no_header_untouched(self):
        text = "产房，凌晨一点十五分。\n我刚出生。\n" + BODY
        res = strip_script_header(text)
        self.assertEqual(res["body"], text)
        self.assertEqual(res["dropped"], [])

    def test_bare_title_needs_labeled_section(self):
        text = "\n".join(["坠入春夜", "大纲", "复仇故事概要，含结局。",
                          "人物小传", "甲：小传。", BODY])
        res = strip_script_header(text)
        self.assertTrue(res["body"].startswith("1-1"))
        self.assertNotIn("大纲", res["body"])

    def test_episode_outline_vs_body(self):
        text = "\n".join(["《某剧》", "## 剧情介绍：", "长篇介绍。",
                          "第一集：主角进城，被开除。",
                          "## 第三十一集", "本集梗概，纯散文没有场景行。",
                          "### 第一集", BODY])
        res = strip_script_header(text)
        self.assertTrue(res["body"].startswith("### 第一集"))
        self.assertNotIn("剧情介绍", res["body"])
        self.assertNotIn("第三十一集", res["body"])

    def test_outline_only_file_refused(self):
        text = "\n".join(["《某剧》", "故事大纲：", "只有大纲没有正文。"])
        res = strip_script_header(text)
        self.assertEqual(res["body"], text)
        self.assertIn("no_body_found", res["flags"])

    def test_body_first_scene_line_at_top(self):
        for first in ("1-1 夜 内 酒店房间", "第一集：", "场 1-1 夜/内 街道"):
            text = first + "\n" + BODY
            res = strip_script_header(text)
            self.assertEqual(res["body"], text, first)


class TestRelocate(unittest.TestCase):
    def test_prefix_shift_and_quote(self):
        header = "《某剧》\n简纲：\n梗概 内容 若干词\n"
        body = "正文 开头 若干 词 这是 冻结 引文 之后 还有 结尾"
        old_text = header + body
        quote = "冻结 引文"
        old_words = old_text.split()
        cut = old_words.index("冻结")
        meta = {"freeze_word": cut, "total_words": len(old_words),
                "freeze": {"freeze_quote": quote, "approx_fraction": 0.5,
                           "snapped_fraction": 0.5}}
        rep = relocate_freeze(meta, old_text, body)
        self.assertEqual(rep["new_cut"],
                         cut - (len(old_words) - len(body.split())))
        self.assertTrue(rep["quote_found"])
        self.assertTrue(rep["quote_agrees"])

    def test_freeze_inside_header_is_error(self):
        old_text = "头部 很长 很长 正文 短"
        meta = {"freeze_word": 1, "total_words": 5,
                "freeze": {"freeze_quote": ""}}
        rep = relocate_freeze(meta, old_text, "正文 短")
        self.assertIn("error", rep)

    def test_quote_absent_pre_existing(self):
        old_text = "头部 正文 一 二 三"
        meta = {"freeze_word": 3, "total_words": 5,
                "freeze": {"freeze_quote": "从未出现的引文",
                           "approx_fraction": 0.6}}
        rep = relocate_freeze(meta, old_text, "正文 一 二 三")
        self.assertTrue(rep["quote_absent_pre_existing"])
        self.assertNotIn("quote_found", rep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
