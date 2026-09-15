"""Regressions for the data-contract helpers (no LLM).

1. seed_from_tag — replicate label derived from the run tag.
2. CALL_CTX / set_call_ctx — round stamping shared with the call log.
3. ChatAPI — temperature/max_tokens are per-instance; call-link fields
   (_last_call_id, _resim_kind) exist with safe defaults.

Run:  python3 tests/test_data_contract.py
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engine"))

from utils import seed_from_tag  # noqa: E402
from modules.llm.BaseLLM import CALL_CTX, set_call_ctx  # noqa: E402
from modules.llm.ChatAPI import ChatAPI  # noqa: E402


class TestSeedFromTag(unittest.TestCase):
    def test_batch_tags(self):
        self.assertEqual(seed_from_tag("batch2_r3"), 3)
        self.assertEqual(seed_from_tag("batch1_r12"), 12)

    def test_bare_replicate(self):
        self.assertEqual(seed_from_tag("r1"), 1)

    def test_no_suffix_is_none(self):
        self.assertIsNone(seed_from_tag("custom"))
        self.assertIsNone(seed_from_tag(""))
        self.assertIsNone(seed_from_tag(None))


class TestCallCtx(unittest.TestCase):
    def test_round_stamp(self):
        old = CALL_CTX.get("round")
        try:
            set_call_ctx(round=7)
            self.assertEqual(CALL_CTX["round"], 7)
        finally:
            set_call_ctx(round=old)


class TestChatAPIContract(unittest.TestCase):
    def test_temperature_and_max_tokens_configurable(self):
        llm = ChatAPI(model="gpt-5.6-sol", temperature=0.0,
                           max_tokens=512)
        self.assertEqual(llm.temperature, 0.0)
        self.assertEqual(llm.max_tokens, 512)

    def test_defaults(self):
        llm = ChatAPI()
        self.assertEqual(llm.temperature, 0.7)
        self.assertEqual(llm.max_tokens, 2048)

    def test_call_link_fields_default_safe(self):
        llm = ChatAPI()
        # no call yet -> no link; kind unset -> logged as None, never stale
        self.assertIsNone(llm._last_call_id)
        self.assertIsNone(getattr(llm, "_resim_kind", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
