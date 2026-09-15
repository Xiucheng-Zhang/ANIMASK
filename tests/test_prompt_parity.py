"""en/zh prompt-set parity (no LLM, no file writes).

Design rule: the zh prompt sets are sentence-by-sentence
translations of the en sets, so the two languages run the SAME engine.
str.format silently swallows extra kwargs, which is exactly how the old zh
set lost {experiences}/{facts} without anyone noticing — these tests pin:

  1. every *_PROMPT name in the en module exists in the zh module and
     uses the IDENTICAL placeholder set (role and world sets alike);
  2. the neutral derivations (en and zh) found every replacement anchor;
  3. the neutral world rewrites kept the placeholder sets of their bases.

Run:  python3 tests/test_prompt_parity.py
  or  python3 -m unittest discover -s tests -p 'test_*.py'
"""
import string
import sys
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENGINE))

from modules.prompt import (  # noqa: E402
    role_agent_prompt_en, role_agent_prompt_zh,
    role_agent_prompt_neutral, role_agent_prompt_zh_neutral,
    world_agent_prompt_en, world_agent_prompt_zh,
    world_agent_prompt_neutral, world_agent_prompt_zh_neutral,
)


def prompt_names(module):
    return {n for n in vars(module)
            if n.endswith("_PROMPT") and isinstance(getattr(module, n), str)}


def placeholders(template):
    return {field for _, field, _, _ in string.Formatter().parse(template)
            if field}


class TestEnZhParity(unittest.TestCase):
    def assert_module_parity(self, en_mod, zh_mod):
        en_names, zh_names = prompt_names(en_mod), prompt_names(zh_mod)
        self.assertEqual(en_names - zh_names, set(),
                         "prompts missing from the zh module")
        self.assertEqual(zh_names - en_names, set(),
                         "zh-only prompts (drop or add the en counterpart)")
        for name in sorted(en_names):
            self.assertEqual(
                placeholders(getattr(en_mod, name)),
                placeholders(getattr(zh_mod, name)),
                f"{name}: placeholder sets differ between en and zh")

    def test_role_prompts(self):
        self.assert_module_parity(role_agent_prompt_en, role_agent_prompt_zh)

    def test_world_prompts(self):
        self.assert_module_parity(world_agent_prompt_en, world_agent_prompt_zh)

    def test_neutral_role_prompts(self):
        self.assert_module_parity(role_agent_prompt_neutral,
                                  role_agent_prompt_zh_neutral)

    def test_neutral_world_prompts(self):
        self.assert_module_parity(world_agent_prompt_neutral,
                                  world_agent_prompt_zh_neutral)


class TestNeutralDerivation(unittest.TestCase):
    def test_en_neutral_anchors_all_found(self):
        self.assertEqual(getattr(role_agent_prompt_neutral, "_unused", set()),
                         set(), "en neutral anchors drifted from the en base")

    def test_zh_neutral_anchors_all_found(self):
        self.assertEqual(getattr(role_agent_prompt_zh_neutral, "_unused",
                                 set()),
                         set(), "zh neutral anchors drifted from the zh base")

    def test_neutral_actually_replaced(self):
        # the drama pumps must be gone from every targeted prompt
        for mod, marker in ((role_agent_prompt_neutral, "dramatic twists"),
                            (role_agent_prompt_zh_neutral, "戏剧性转折")):
            for name in ("ROLE_PLAN_PROMPT", "ROLE_SINGLE_ROLE_RESPONSE_PROMPT",
                         "ROLE_MULTI_ROLE_RESPONSE_PROMPT",
                         "ROLE_NPC_RESPONSE_PROMPT"):
                self.assertNotIn(marker, getattr(mod, name),
                                 f"{mod.__name__}.{name} still has the "
                                 f"original drama instruction")

    def test_world_neutral_rewrites_keep_placeholders(self):
        for base, neutral in ((world_agent_prompt_en, world_agent_prompt_neutral),
                              (world_agent_prompt_zh,
                               world_agent_prompt_zh_neutral)):
            for name in ("ENVIRONMENT_INTERACTION_PROMPT",
                         "NPC_INTERACTION_PROMPT",
                         "CANONICAL_NPC_INTERACTION_PROMPT",
                         "SELECT_SCREEN_ACTORS_PROMPT",
                         "UPDATE_EVENT_PROMPT", "JUDGE_IF_ENDED_PROMPT"):
                self.assertEqual(
                    placeholders(getattr(base, name)),
                    placeholders(getattr(neutral, name)),
                    f"{neutral.__name__}.{name}: placeholders differ from base")

    def test_canonical_npc_prompt_actually_rewritten(self):
        # the carded-NPC prompt used to fall through getattr to the ORIGINAL
        # text in both neutral modules — the ablation arm ran partly
        # un-ablated on every pack with npc_cards.json
        for base, neutral in ((world_agent_prompt_en, world_agent_prompt_neutral),
                              (world_agent_prompt_zh,
                               world_agent_prompt_zh_neutral)):
            self.assertNotEqual(
                getattr(base, "CANONICAL_NPC_INTERACTION_PROMPT"),
                getattr(neutral, "CANONICAL_NPC_INTERACTION_PROMPT"),
                f"{neutral.__name__}: canonical NPC prompt not rewritten")

    # anchor x target hit matrix, pinned per cell. `_unused` only proves each
    # anchor matched SOMEWHERE; rewording an anchor sentence inside ONE
    # prompt would leave that prompt un-neutralized with tests green. Rows =
    # _REPLACEMENTS order, columns = _TARGETS order (PLAN, SINGLE, MULTI,
    # NPC). When a prompt is deliberately reworded, update the matrix in the
    # same change.
    EN_HITS = [(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
               (1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1)]
    ZH_HITS = [(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
               (1, 1, 1, 0), (0, 0, 0, 1)]

    def _hit_matrix(self, base, neutral):
        return [tuple(int(old in getattr(base, t)) for t in neutral._TARGETS)
                for old, _ in neutral._REPLACEMENTS]

    def test_en_anchor_hit_matrix_pinned(self):
        self.assertEqual(
            self._hit_matrix(role_agent_prompt_en, role_agent_prompt_neutral),
            self.EN_HITS)

    def test_zh_anchor_hit_matrix_pinned(self):
        self.assertEqual(
            self._hit_matrix(role_agent_prompt_zh,
                             role_agent_prompt_zh_neutral),
            self.ZH_HITS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
