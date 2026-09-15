"""Location-code / slug rules of the pack builder (no LLM, no file writes).

Regression for the zh collapse: `re.sub(r"[^A-Za-z0-9]", "", name)` mapped
every CJK-only location name to "" so all zh packs ended up with ONE location.

Run:  python3 tests/test_pack_codes.py
  or  cd scripts && python3 -m unittest discover -s ../tests -p 'test_*.py'
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from animask.resim.build_world_pack import (ascii_slug,  # noqa: E402
                                        build_location_table, code_of)


class TestAsciiSlug(unittest.TestCase):
    def test_english_unchanged_from_old_rule(self):
        # exactly what the old regex produced — en packs must not change
        for name in ["Morag’s farm", "Lower-city streets",
                     "Gemma and Jacub's home", "Farm tech shed"]:
            import re
            self.assertEqual(ascii_slug(name),
                             re.sub(r"[^A-Za-z0-9]", "", name))

    def test_cjk_never_empty_and_distinct(self):
        names = ["悦洋度假村外", "夏星辰住处", "夏星辰家门口", "廉氏集团总裁办公室"]
        slugs = [ascii_slug(n) for n in names]
        self.assertTrue(all(slugs), slugs)
        self.assertTrue(all(s.isascii() for s in slugs), slugs)
        self.assertEqual(len(set(slugs)), len(names), slugs)

    def test_mixed_name_keeps_cjk_part(self):
        # old rule degraded these to "VIP" / "307" / "A"
        self.assertNotEqual(ascii_slug("秦氏赌场VIP包厢"), "VIP")
        self.assertIn("VIP", ascii_slug("秦氏赌场VIP包厢"))
        self.assertNotEqual(ascii_slug("花园酒店307房间"), "307")
        self.assertNotEqual(ascii_slug("夜内A国拍卖场"), "A")


class TestLocationTable(unittest.TestCase):
    def _rows(self, names):
        return [{"location_name": n, "description": f"d-{n}", "detail": "x"}
                for n in names]

    def test_zh_table_keeps_every_location(self):
        names = ["悦洋度假村外", "夏星辰住处", "夏星辰家", "夏星辰家门口",
                 "廉氏集团", "秦氏赌场VIP包厢", "花园酒店307房间"]
        t = build_location_table(self._rows(names))
        self.assertEqual(len(t), len(names))
        self.assertEqual({e["location_name"] for e in t.values()}, set(names))
        for code, e in t.items():
            self.assertTrue(code and code.isascii(), code)
            self.assertEqual(e["location_code"], code)

    def test_en_codes_identical_to_old_rule(self):
        names = ["Morag’s farm", "Farm dining room", "Farm tech shed"]
        t = build_location_table(self._rows(names))
        self.assertEqual(list(t), ["Moragsfarm", "Farmdiningroom",
                                   "Farmtechshed"])

    def test_collision_gets_suffix_not_overwrite(self):
        t = build_location_table(self._rows(["Farm-house", "Farm house"]))
        self.assertEqual(len(t), 2)
        self.assertEqual(sorted(t), ["Farmhouse", "Farmhouse2"])

    def test_duplicate_and_empty_names_dropped(self):
        t = build_location_table(self._rows(["城门", "城门", "", "  "]))
        self.assertEqual(len(t), 1)
        self.assertEqual(next(iter(t.values()))["location_name"], "城门")

    def test_non_dict_rows_ignored(self):
        t = build_location_table(["garbage", None, {"location_name": "城门"}])
        self.assertEqual(len(t), 1)


class TestRoleCodeUnchanged(unittest.TestCase):
    """code_of is NOT touched by this fix; pin its current behaviour so any
    later change to it is deliberate (existing pack dirs depend on it)."""

    def test_pure_cjk_and_pure_ascii(self):
        import animask.resim.build_world_pack as b
        old = b.PACK_LANG
        try:
            b.PACK_LANG = "zh"
            self.assertEqual(code_of("夏星辰"), "XiaXingChen-zh")
            b.PACK_LANG = "en"
            self.assertEqual(code_of("Morag Blake"), "MoragBlake-en")
        finally:
            b.PACK_LANG = old


if __name__ == "__main__":
    unittest.main()
