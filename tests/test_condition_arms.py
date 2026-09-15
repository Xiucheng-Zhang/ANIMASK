"""Regressions for the condition-arm wiring (no LLM).

1. RPAgent._apply_persona_level — P0 ablates card material only.
2. l3_ablation._persona_blocks/_ablate — single-step ablation covers the
   same card material as run-level P0 (profile + relation lines +
   motivation).
3. batch_runner.cond_suffix — condition-aware tags; default stays stable.
4. get_models claude-* — routes to the CLI adapter (the SDK adapter had no
   key and a broken message format).

Run:  python3 tests/test_condition_arms.py
"""
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))
sys.path.insert(0, str(REPO))

from utils import P0_PLACEHOLDER, get_models  # noqa: E402
from modules.main_role_agent import RPAgent  # noqa: E402
from animask.persona_probe.l3_ablation import (PLACEHOLDER,  # noqa: E402
                                       _ablate, _persona_blocks)
from animask.resim.batch_runner import cond_suffix  # noqa: E402

ROLE_INFO = {
    "profile": "谨慎多疑，从不先亮底牌。",
    "relation": {"B-zh": {"relation": ["妻子"], "detail": "你欠她一条命。"}},
    "motivation": "夺回家产。",
}


class TestPersonaLevelSwitch(unittest.TestCase):
    def _agent(self, level):
        a = RPAgent.__new__(RPAgent)
        a.language = "zh"
        a.role_profile = ROLE_INFO["profile"]
        a.relation = dict(ROLE_INFO["relation"])
        a.motivation = ROLE_INFO["motivation"]
        a.goal = "seeded"
        a.x_agentic = {"goal_now": "seeded"}
        old = os.environ.get("ANIMASK_PERSONA_LEVEL")
        os.environ["ANIMASK_PERSONA_LEVEL"] = level
        try:
            a._apply_persona_level()
        finally:
            if old is None:
                os.environ.pop("ANIMASK_PERSONA_LEVEL", None)
            else:
                os.environ["ANIMASK_PERSONA_LEVEL"] = old
        return a

    def test_p0_ablates_card_material(self):
        a = self._agent("P0")
        self.assertEqual(a.role_profile, P0_PLACEHOLDER["zh"])
        self.assertEqual(a.relation, {})
        self.assertEqual(a.motivation, "")
        self.assertEqual(a.goal, "")
        self.assertEqual(a.x_agentic.get("goal_now"), "")

    def test_p3_leaves_everything(self):
        a = self._agent("P3")
        self.assertEqual(a.role_profile, ROLE_INFO["profile"])
        self.assertEqual(a.motivation, ROLE_INFO["motivation"])
        self.assertEqual(a.goal, "seeded")

    def test_placeholders_in_sync_with_l3(self):
        # the two arms measure the same construct only if they share the
        # placeholder text
        self.assertEqual(P0_PLACEHOLDER, PLACEHOLDER)


class TestL3FullAblation(unittest.TestCase):
    def test_blocks_cover_relation_and_motivation(self):
        profile, blocks = _persona_blocks(ROLE_INFO, "zh")
        self.assertEqual(profile, ROLE_INFO["profile"])
        self.assertIn("这是你的妻子. 你欠她一条命。", blocks)
        self.assertIn("夺回家产。", blocks)

    def test_ablate_removes_all_card_material(self):
        profile, blocks = _persona_blocks(ROLE_INFO, "zh")
        messages = [{"role": "user", "content":
                     f"{profile}\n这是你的妻子. 你欠她一条命。\n"
                     f"你的动机：夺回家产。\n现在做出选择。"}]
        out, found = _ablate(messages, profile, "zh", blocks)
        self.assertTrue(found)
        text = out[0]["content"]
        self.assertIn(PLACEHOLDER["zh"], text)
        self.assertNotIn("多疑", text)
        self.assertNotIn("欠她一条命", text)
        self.assertNotIn("夺回家产", text)
        self.assertIn("现在做出选择", text)


class TestCondSuffix(unittest.TestCase):
    def test_default_condition_is_empty(self):
        self.assertEqual(cond_suffix({"sid": "x"}), "")
        self.assertEqual(cond_suffix({"sid": "x", "persona": "P3"}), "")

    def test_model_and_persona_parts(self):
        self.assertEqual(cond_suffix({"actor_model": "gpt-5.6-sol"}),
                         "gpt56sol")
        self.assertEqual(cond_suffix({"persona": "P0"}), "P0")
        self.assertEqual(
            cond_suffix({"actor_model": "claude-sonnet-5", "persona": "P0"}),
            "claudesonnet5_P0")


class TestModelRouting(unittest.TestCase):
    def test_claude_routes_to_anthropic(self):
        llm = get_models("claude-sonnet-5")
        self.assertEqual(type(llm).__name__, "ChatAPI")
        self.assertEqual(llm.provider, "anthropic")
        self.assertEqual(llm.model_name, "claude-sonnet-5")
        self.assertEqual(llm.temperature, 0.7)  # sampling is per instance

    def test_every_paper_model_has_a_provider(self):
        expect = {"gpt-5.5": "openai", "claude-sonnet-5": "anthropic",
                  "gemini-3.7-flash": "gemini", "deepseek-v4-flash": "deepseek",
                  "kimi-k2.6": "moonshot", "qwen3.7-plus": "dashscope"}
        for name, provider in expect.items():
            self.assertEqual(get_models(name).provider, provider, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
