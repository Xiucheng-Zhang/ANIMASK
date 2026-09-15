"""batch_runner.run_item must not mark ALL=done when a probe stage failed
(an earlier batch once showed ALL=done for items whose probes had failed).

    python3 tests/test_batch_all_gating.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask.resim.batch_runner import Batch                        # noqa: E402


def make_batch(tmp, results):
    """results: dict of stage-method name -> bool return."""
    import animask.resim.batch_runner as br
    orig_root = br.ROOT
    br.ROOT = Path(tmp)
    b = Batch("t", [{"sid": "s1", "runs": 1, "_index": 0}], 1, 1,
              probe=True)
    br.ROOT = orig_root
    for name, ret in results.items():
        setattr(b, name, lambda *a, r=ret, **k: r)
    return b


class TestAllGating(unittest.TestCase):
    def _run(self, probe_ok):
        with tempfile.TemporaryDirectory() as tmp:
            b = make_batch(tmp, {
                "ensure_extract": True, "ensure_pack": True,
                "ensure_processes": True, "ensure_run": True,
                "ensure_probe": probe_ok})
            b.run_item({"sid": "s1", "runs": 1, "_index": 0})
            return b.state["s1"]["ALL"]

    def test_all_done_when_clean(self):
        self.assertEqual(self._run(True)["status"], "done")

    def test_probe_failure_flagged(self):
        rec = self._run(False)
        self.assertEqual(rec["status"], "done_with_failures")
        self.assertIn("re-launch", rec["note"])


class TestPreflight(unittest.TestCase):
    def setUp(self):
        from animask import llm
        self.llm = llm
        self._env = llm.ENV
        llm.ENV = {"OPENAI_API_KEY": "k", "DEEPSEEK_API_KEY": "k"}

    def tearDown(self):
        self.llm.ENV = self._env

    def test_reports_missing_story_and_missing_key(self):
        from animask.resim.batch_runner import preflight
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "clean").mkdir(parents=True)
            (root / "data" / "clean" / "s1.txt").write_text("once upon a time")
            spec = [{"sid": "s1"}, {"sid": "s2"},
                    {"sid": "s1", "actor_model": "claude-sonnet-5"}]
            problems = preflight(spec, root=root)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("s2" in p and "not found" in p for p in problems))
        self.assertTrue(any("ANTHROPIC_API_KEY" in p for p in problems))

    def test_ready_batch_has_no_problems(self):
        from animask.resim.batch_runner import preflight
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "clean").mkdir(parents=True)
            (root / "data" / "clean" / "s1.txt").write_text("text")
            self.assertEqual(preflight([{"sid": "s1"}], root=root), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
